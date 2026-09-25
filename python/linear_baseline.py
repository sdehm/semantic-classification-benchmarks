"""Train a logistic-regression head over frozen ModernBERT mean-pooled embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from common import (
    load_json,
    load_records,
    load_stratified_records,
    resolve_device,
    serialize_probabilities,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/banking77/records.parquet"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/modernbert-base.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/linear/validation.predictions.parquet"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("runs/linear/embeddings"))
    parser.add_argument("--c", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    measure_latency: bool,
) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    latencies: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        started = perf_counter()
        with torch.inference_mode():
            output = model(**encoded)
            pooled = torch.nn.functional.normalize(
                mean_pool(output.last_hidden_state, encoded["attention_mask"]),
                p=2,
                dim=1,
            )
        elapsed_ms = round((perf_counter() - started) * 1_000 / len(batch))
        vectors.append(pooled.cpu().float().numpy())
        if measure_latency:
            latencies.extend([elapsed_ms] * len(batch))
    return np.vstack(vectors), latencies


def embedding_cache_path(
    records: pl.DataFrame,
    model_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    measure_latency: bool,
    cache_dir: Path,
) -> Path:
    fingerprint = hashlib.sha256()
    cache_inputs = {
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "max_length": model_config["max_length"],
        "device": str(device),
        "batch_size": batch_size,
        "measure_latency": measure_latency,
    }
    fingerprint.update(json.dumps(cache_inputs, sort_keys=True).encode())
    label_column = "gold_label" if "gold_label" in records.columns else "gold_labels_json"
    for record in records.select("id", "text", label_column).iter_rows():
        fingerprint.update(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode())
        fingerprint.update(b"\n")
    return cache_dir / f"{records['split'][0]}-{fingerprint.hexdigest()}.npz"


def load_or_embed(
    records: pl.DataFrame,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    model_config: dict[str, Any],
    batch_size: int,
    measure_latency: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, list[int], bool, int]:
    cache_path = embedding_cache_path(
        records, model_config, device, batch_size, measure_latency, cache_dir
    )
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            vectors = cached["vectors"]
            latencies = cached["latencies"].tolist()
        expected_latency_count = records.height if measure_latency else 0
        if vectors.shape[0] != records.height or len(latencies) != expected_latency_count:
            raise ValueError(f"embedding cache has unexpected row count: {cache_path}")
        return vectors, latencies, True, 0

    started = perf_counter()
    vectors, latencies = embed(
        records.get_column("text").to_list(),
        tokenizer,
        model,
        device,
        model_config["max_length"],
        batch_size,
        measure_latency,
    )
    elapsed_ms = round((perf_counter() - started) * 1_000)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vectors=vectors,
            latencies=np.asarray(latencies, dtype=np.int64),
        )
    os.replace(temporary_path, cache_path)
    return vectors, latencies, False, elapsed_ms


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, metadata_path)
    return metadata_path


def main() -> None:
    args = parse_args()
    if args.train_batch_size < 1:
        raise ValueError(f"train batch size must be positive, got {args.train_batch_size}")
    if args.inference_batch_size < 1:
        raise ValueError(f"inference batch size must be positive, got {args.inference_batch_size}")
    model_config = load_json(args.model_config)
    classifier_config = model_config["logistic_regression"]
    regularization_c = (
        args.c if args.c is not None else classifier_config["regularization_c"]
    )
    max_iter = args.max_iter if args.max_iter is not None else classifier_config["max_iter"]
    seed = args.seed if args.seed is not None else classifier_config["seed"]
    if regularization_c <= 0:
        raise ValueError(f"regularization C must be positive, got {regularization_c}")
    if max_iter < 1:
        raise ValueError(f"max iterations must be positive, got {max_iter}")
    train = load_stratified_records(args.dataset, "train", args.train_limit)
    evaluation = load_records(args.dataset, args.split, args.limit)
    if train.get_column("gold_label").n_unique() < 2:
        raise ValueError("training selection must contain at least two labels")
    device = resolve_device(args.device)
    plan = {
        "classifier": "frozen-modernbert-logistic-regression",
        "model": f"{model_config['id']}@{model_config['revision']}",
        "train_records": train.height,
        "evaluation_records": evaluation.height,
        "split": args.split,
        "device": str(device),
        "train_batch_size": args.train_batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_length": model_config["max_length"],
        "regularization_c": regularization_c,
        "max_iter": max_iter,
        "seed": seed,
        "embedding_cache_dir": str(args.embedding_cache_dir),
        "output": str(args.output),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"], revision=model_config["revision"])
    model = AutoModel.from_pretrained(model_config["id"], revision=model_config["revision"]).to(device)
    model.eval()
    model_load_milliseconds = round((perf_counter() - started) * 1_000)

    training_embeddings, _, training_cache_hit, training_embedding_milliseconds = load_or_embed(
        train,
        tokenizer,
        model,
        device,
        model_config,
        args.train_batch_size,
        False,
        args.embedding_cache_dir,
    )
    started = perf_counter()
    classifier = LogisticRegression(
        C=regularization_c,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    ).fit(training_embeddings, train.get_column("gold_label").to_list())
    fit_milliseconds = round((perf_counter() - started) * 1_000)

    evaluation_embeddings, latencies, evaluation_cache_hit, evaluation_embedding_milliseconds = load_or_embed(
        evaluation,
        tokenizer,
        model,
        device,
        model_config,
        args.inference_batch_size,
        True,
        args.embedding_cache_dir,
    )
    probabilities = classifier.predict_proba(evaluation_embeddings)
    rows: list[dict[str, Any]] = []
    for index, row_id in enumerate(evaluation.get_column("id").to_list()):
        label_probabilities = {
            label: float(score)
            for label, score in zip(classifier.classes_, probabilities[index], strict=True)
        }
        predicted_label = classifier.classes_[int(probabilities[index].argmax())]
        rows.append(
            {
                "id": row_id,
                "predicted_label": predicted_label,
                "label_probabilities_json": serialize_probabilities(label_probabilities),
                "confidence": float(probabilities[index].max()),
                "has_confidence": True,
                "latency_ms": latencies[index],
                "model": f"{model_config['id']}@{model_config['revision']}+logreg",
            }
        )
    write_predictions(args.output, rows)
    plan.update(
        {
            "dry_run": False,
            "model_load_milliseconds": model_load_milliseconds,
            "training_embedding_cache_hit": training_cache_hit,
            "training_embedding_milliseconds": training_embedding_milliseconds,
            "fit_milliseconds": fit_milliseconds,
            "evaluation_embedding_cache_hit": evaluation_cache_hit,
            "evaluation_embedding_milliseconds": evaluation_embedding_milliseconds,
        }
    )
    plan["metadata"] = str(args.output.with_suffix(".metadata.json"))
    write_metadata(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
