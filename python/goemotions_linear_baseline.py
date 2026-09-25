"""Train independent frozen-ModernBERT emotion heads for GoEmotions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoTokenizer

from common import (
    apply_fallback_neutral,
    canonical_labels,
    encode_multilabel_predictions,
    encode_multilabel_targets,
    load_json,
    load_multilabel_records,
    multilabel_labels_from_records,
    resolve_device,
    serialize_probabilities,
    write_multilabel_predictions,
)
from linear_baseline import load_or_embed, write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/goemotions/records.parquet"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/modernbert-goemotions-linear.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/goemotions-linear/validation.predictions.parquet"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold-limit", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument(
        "--embedding-cache-dir", type=Path, default=Path("runs/goemotions-linear/embeddings")
    )
    parser.add_argument("--c", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Fixed global emotion threshold. If omitted, choose one on validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_threshold(
    emotion_probabilities: np.ndarray,
    gold_labels: list[list[float]],
    emotion_labels: list[str],
    all_labels: list[str],
    candidates: list[float],
) -> tuple[float, float]:
    best_threshold: float | None = None
    best_macro_f1 = -1.0
    for threshold in sorted(set(candidates)):
        if not 0 < threshold < 1:
            raise ValueError(f"threshold candidates must be between zero and one, got {threshold}")
        predictions = apply_fallback_neutral(
            emotion_probabilities.tolist(), emotion_labels, threshold
        )
        macro_f1 = f1_score(
            gold_labels,
            encode_multilabel_predictions(predictions, all_labels),
            average="macro",
            zero_division=0,
        )
        if macro_f1 > best_macro_f1 or (
            macro_f1 == best_macro_f1
            and (best_threshold is None or threshold > best_threshold)
        ):
            best_threshold = threshold
            best_macro_f1 = float(macro_f1)
    if best_threshold is None:
        raise ValueError("threshold candidate list is empty")
    return best_threshold, best_macro_f1


def main() -> None:
    args = parse_args()
    if args.train_batch_size < 1:
        raise ValueError(f"train batch size must be positive, got {args.train_batch_size}")
    if args.inference_batch_size < 1:
        raise ValueError(
            f"inference batch size must be positive, got {args.inference_batch_size}"
        )
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError(f"threshold must be between zero and one, got {args.threshold}")

    model_config = load_json(args.model_config)
    classifier_config = model_config["multi_label_logistic_regression"]
    regularization_c = args.c if args.c is not None else classifier_config["regularization_c"]
    max_iter = args.max_iter if args.max_iter is not None else classifier_config["max_iter"]
    seed = args.seed if args.seed is not None else classifier_config["seed"]
    if regularization_c <= 0:
        raise ValueError(f"regularization C must be positive, got {regularization_c}")
    if max_iter < 1:
        raise ValueError(f"max iterations must be positive, got {max_iter}")

    train = load_multilabel_records(args.dataset, "train", None)
    threshold_records = load_multilabel_records(args.dataset, "validation", args.threshold_limit)
    evaluation = load_multilabel_records(args.dataset, args.split, args.limit)
    all_labels = multilabel_labels_from_records(train)
    emotion_labels = [label for label in all_labels if label != "neutral"]
    if not emotion_labels:
        raise ValueError("dataset does not contain any emotion labels")
    device = resolve_device(args.device)
    plan = {
        "classifier": "frozen-modernbert-one-vs-rest-logistic-regression",
        "model": f"{model_config['id']}@{model_config['revision']}",
        "train_records": train.height,
        "threshold_records": threshold_records.height,
        "evaluation_records": evaluation.height,
        "split": args.split,
        "device": str(device),
        "train_batch_size": args.train_batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_length": model_config["max_length"],
        "regularization_c": regularization_c,
        "regularization_selection_candidates": classifier_config[
            "validation_regularization_candidates"
        ],
        "max_iter": max_iter,
        "seed": seed,
        "threshold_policy": "global validation-selected emotion threshold with fallback neutral",
        "threshold_candidates": classifier_config["threshold_candidates"],
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
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=regularization_c,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )
    ).fit(
        training_embeddings,
        np.asarray(encode_multilabel_targets(train, emotion_labels, ignore_neutral=True)),
    )
    fit_milliseconds = round((perf_counter() - started) * 1_000)

    threshold_embeddings, _, threshold_cache_hit, threshold_embedding_milliseconds = load_or_embed(
        threshold_records,
        tokenizer,
        model,
        device,
        model_config,
        args.train_batch_size,
        False,
        args.embedding_cache_dir,
    )
    threshold_probabilities = classifier.predict_proba(threshold_embeddings)
    threshold_gold = encode_multilabel_targets(
        threshold_records, all_labels, ignore_neutral=False
    )
    if args.threshold is None:
        threshold, validation_macro_f1 = select_threshold(
            threshold_probabilities,
            threshold_gold,
            emotion_labels,
            all_labels,
            classifier_config["threshold_candidates"],
        )
        threshold_selection = "validation"
    else:
        threshold = args.threshold
        threshold_selection = "fixed"
        validation_predictions = apply_fallback_neutral(
            threshold_probabilities, emotion_labels, threshold
        )
        validation_macro_f1 = float(
            f1_score(
                threshold_gold,
                encode_multilabel_predictions(validation_predictions, all_labels),
                average="macro",
                zero_division=0,
            )
        )

    evaluation_embeddings, latencies, evaluation_cache_hit, evaluation_embedding_milliseconds = (
        load_or_embed(
            evaluation,
            tokenizer,
            model,
            device,
            model_config,
            args.inference_batch_size,
            True,
            args.embedding_cache_dir,
        )
    )
    evaluation_probabilities = classifier.predict_proba(evaluation_embeddings)
    evaluation_predictions = apply_fallback_neutral(
        evaluation_probabilities.tolist(), emotion_labels, threshold
    )
    rows: list[dict[str, Any]] = []
    for row_index, row_id in enumerate(evaluation.get_column("id").to_list()):
        label_probabilities = {
            label: float(probability)
            for label, probability in zip(
                emotion_labels, evaluation_probabilities[row_index], strict=True
            )
        }
        rows.append(
            {
                "id": row_id,
                "predicted_labels_json": canonical_labels(evaluation_predictions[row_index]),
                "label_probabilities_json": serialize_probabilities(label_probabilities),
                "latency_ms": latencies[row_index],
                "model": f"{model_config['id']}@{model_config['revision']}+ovr-logreg",
            }
        )
    write_multilabel_predictions(args.output, rows)

    plan.update(
        {
            "dry_run": False,
            "model_load_milliseconds": model_load_milliseconds,
            "training_embedding_cache_hit": training_cache_hit,
            "training_embedding_milliseconds": training_embedding_milliseconds,
            "fit_milliseconds": fit_milliseconds,
            "threshold_embedding_cache_hit": threshold_cache_hit,
            "threshold_embedding_milliseconds": threshold_embedding_milliseconds,
            "threshold": threshold,
            "threshold_selection": threshold_selection,
            "threshold_validation_macro_f1": validation_macro_f1,
            "evaluation_embedding_cache_hit": evaluation_cache_hit,
            "evaluation_embedding_milliseconds": evaluation_embedding_milliseconds,
            "raw_probability_labels": emotion_labels,
        }
    )
    plan["metadata"] = str(args.output.with_suffix(".metadata.json"))
    write_metadata(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
