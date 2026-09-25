"""Fine-tune ModernBERT sentence embeddings and classify against label prototypes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses, modules
from torch.utils.data import DataLoader

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
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/modernbert-sentence-transformer.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sentence-transformer/validation-v1.predictions.parquet"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("runs/sentence-transformer/validation-v1.model"),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, metadata_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_label_distinct_pairs(
    records: Any, batch_size: int, seed: int
) -> tuple[list[InputExample], int]:
    texts_by_label: dict[str, list[str]] = defaultdict(list)
    for text, label in records.select("text", "gold_label").iter_rows():
        texts_by_label[label].append(text)

    pairs_by_label: dict[str, list[InputExample]] = {}
    for label, texts in texts_by_label.items():
        if len(texts) < 2:
            raise ValueError(f"label {label!r} has fewer than two training examples")
        label_rng = random.Random(f"{seed}:{label}")
        label_rng.shuffle(texts)
        pairs_by_label[label] = [
            InputExample(texts=[text, texts[(index + 1) % len(texts)]])
            for index, text in enumerate(texts)
        ]

    rng = random.Random(seed)
    for pairs in pairs_by_label.values():
        rng.shuffle(pairs)
    ordered: list[InputExample] = []
    while True:
        labels = [label for label, pairs in pairs_by_label.items() if pairs]
        if len(labels) < batch_size:
            break
        rng.shuffle(labels)
        ordered.extend(pairs_by_label[label].pop() for label in labels[:batch_size])

    if len(ordered) < batch_size:
        raise ValueError("training selection does not contain one complete contrastive batch")
    total_pairs = sum(len(pairs) for pairs in pairs_by_label.values()) + len(ordered)
    return ordered, total_pairs - len(ordered)


def create_model(
    model_config: dict[str, Any], device: torch.device
) -> SentenceTransformer:
    transformer = modules.Transformer(
        model_config["id"],
        model_kwargs={"revision": model_config["revision"]},
        max_seq_length=model_config["max_length"],
    )
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="mean",
    )
    return SentenceTransformer(modules=[transformer, pooling], device=str(device))


def embed(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    measure_latency: bool,
) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    latencies: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        started = perf_counter()
        vectors.append(
            model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        elapsed_ms = round((perf_counter() - started) * 1_000 / len(batch))
        if measure_latency:
            latencies.extend([elapsed_ms] * len(batch))
    return np.vstack(vectors), latencies


def build_prototypes(
    embeddings: np.ndarray, labels: list[str]
) -> tuple[list[str], np.ndarray]:
    vectors_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for embedding, label in zip(embeddings, labels, strict=True):
        vectors_by_label[label].append(embedding)
    prototype_labels = sorted(vectors_by_label)
    prototypes = np.vstack(
        [
            np.mean(vectors_by_label[label], axis=0)
            for label in prototype_labels
        ]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    return prototype_labels, prototypes


def softmax(scores: np.ndarray, scale: float) -> np.ndarray:
    scaled = scores * scale
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    model_config = load_json(args.model_config)
    training_config = model_config["training"]
    prototype_config = model_config["prototype_classifier"]
    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else training_config["batch_size"]
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else training_config["learning_rate"]
    )
    warmup_ratio = (
        args.warmup_ratio
        if args.warmup_ratio is not None
        else training_config["warmup_ratio"]
    )
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if train_batch_size < 2:
        raise ValueError(f"train batch size must be at least two, got {train_batch_size}")
    if learning_rate <= 0:
        raise ValueError(f"learning rate must be positive, got {learning_rate}")
    if not 0 <= warmup_ratio < 1:
        raise ValueError(f"warmup ratio must be in [0, 1), got {warmup_ratio}")
    if args.resume and args.load_model:
        raise ValueError("--resume cannot be combined with --load-model")

    train = load_stratified_records(args.dataset, "train", args.train_limit)
    evaluation = load_records(args.dataset, args.split, args.limit)
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir or args.model_output.with_suffix(".checkpoints")
    plan = {
        "classifier": "modernbert-sentence-transformer-prototypes",
        "backbone": f"{model_config['id']}@{model_config['revision']}",
        "split": args.split,
        "train_records": train.height,
        "evaluation_records": evaluation.height,
        "device": str(device),
        "max_length": model_config["max_length"],
        "loss": training_config["loss"],
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "prototype_embedding_batch_size": prototype_config["embedding_batch_size"],
        "inference_batch_size": prototype_config["inference_batch_size"],
        "prototype_softmax_scale": prototype_config["softmax_scale"],
        "model_output": str(args.model_output),
        "load_model": str(args.load_model) if args.load_model else None,
        "checkpoint_dir": str(checkpoint_dir),
        "resume": args.resume,
        "output": str(args.output),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    set_seed(training_config["seed"])
    started = perf_counter()
    if args.load_model:
        model = SentenceTransformer(str(args.load_model), device=str(device))
        trained = False
    else:
        if args.model_output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing trained model: {args.model_output}"
            )
        model = create_model(model_config, device)
        trained = True
    model_load_milliseconds = round((perf_counter() - started) * 1_000)

    if trained:
        pairs, dropped_pairs = make_label_distinct_pairs(
            train, train_batch_size, training_config["seed"]
        )
        data_loader = DataLoader(
            pairs,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=model.smart_batching_collate,
        )
        warmup_steps = math.ceil(len(data_loader) * epochs * warmup_ratio)
        started = perf_counter()
        model.fit(
            train_objectives=[(data_loader, losses.MultipleNegativesRankingLoss(model))],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            weight_decay=training_config["weight_decay"],
            checkpoint_path=str(checkpoint_dir),
            checkpoint_save_steps=training_config["checkpoint_steps"],
            checkpoint_save_total_limit=2,
            resume_from_checkpoint=args.resume,
            show_progress_bar=True,
        )
        training_milliseconds = round((perf_counter() - started) * 1_000)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_output))
    else:
        pairs = []
        dropped_pairs = 0
        warmup_steps = 0
        training_milliseconds = 0

    started = perf_counter()
    training_embeddings, _ = embed(
        model,
        train.get_column("text").to_list(),
        prototype_config["embedding_batch_size"],
        False,
    )
    prototype_labels, prototypes = build_prototypes(
        training_embeddings, train.get_column("gold_label").to_list()
    )
    prototype_milliseconds = round((perf_counter() - started) * 1_000)

    started = perf_counter()
    evaluation_embeddings, latencies = embed(
        model,
        evaluation.get_column("text").to_list(),
        prototype_config["inference_batch_size"],
        True,
    )
    inference_milliseconds = round((perf_counter() - started) * 1_000)
    probabilities = softmax(
        evaluation_embeddings @ prototypes.T,
        prototype_config["softmax_scale"],
    )
    rows: list[dict[str, Any]] = []
    for index, row_id in enumerate(evaluation.get_column("id").to_list()):
        label_probabilities = {
            label: float(score)
            for label, score in zip(prototype_labels, probabilities[index], strict=True)
        }
        predicted_index = int(probabilities[index].argmax())
        rows.append(
            {
                "id": row_id,
                "predicted_label": prototype_labels[predicted_index],
                "label_probabilities_json": serialize_probabilities(label_probabilities),
                "confidence": float(probabilities[index, predicted_index]),
                "has_confidence": True,
                "latency_ms": latencies[index],
                "model": f"{model_config['id']}@{model_config['revision']}+mnrl-prototypes",
            }
        )
    write_predictions(args.output, rows)
    plan.update(
        {
            "dry_run": False,
            "trained": trained,
            "training_pairs": len(pairs),
            "dropped_pairs": dropped_pairs,
            "warmup_steps": warmup_steps,
            "model_load_milliseconds": model_load_milliseconds,
            "training_milliseconds": training_milliseconds,
            "prototype_milliseconds": prototype_milliseconds,
            "inference_milliseconds": inference_milliseconds,
        }
    )
    plan["metadata"] = str(args.output.with_suffix(".metadata.json"))
    write_metadata(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
