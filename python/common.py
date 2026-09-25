"""Shared artifact and runtime helpers for local classifier baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

DATASET_COLUMNS = {"id", "text", "split", "gold_label"}
MULTILABEL_DATASET_COLUMNS = {"id", "text", "split", "gold_labels_json"}
PREDICTION_SCHEMA = {
    "id": pl.String,
    "predicted_label": pl.String,
    "label_probabilities_json": pl.String,
    "confidence": pl.Float64,
    "has_confidence": pl.Boolean,
    "latency_ms": pl.Int64,
    "model": pl.String,
}
MULTILABEL_PREDICTION_SCHEMA = {
    "id": pl.String,
    "predicted_labels_json": pl.String,
    "label_probabilities_json": pl.String,
    "latency_ms": pl.Int64,
    "model": pl.String,
}
MULTILABEL_NEUTRAL_LABEL = "neutral"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path, split: str, limit: int | None) -> pl.DataFrame:
    records = pl.read_parquet(path)
    missing = DATASET_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    selected = records.filter(pl.col("split") == split).sort("id")
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")
        selected = selected.head(limit)
    if selected.is_empty():
        raise ValueError(f"no records found for split {split!r}")
    return selected


def load_stratified_records(path: Path, split: str, limit: int | None) -> pl.DataFrame:
    records = load_records(path, split, None)
    if limit is None:
        return records
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    labels = records.get_column("gold_label").unique().sort().to_list()
    if limit < len(labels):
        raise ValueError(
            f"stratified limit must include at least one example per label, got {limit}"
        )
    examples_per_label, extra_examples = divmod(limit, len(labels))
    selected: list[pl.DataFrame] = []
    for index, label in enumerate(labels):
        label_limit = examples_per_label + (1 if index < extra_examples else 0)
        label_records = records.filter(pl.col("gold_label") == label)
        if label_records.height < label_limit:
            raise ValueError(
                f"label {label!r} has {label_records.height} records, fewer than {label_limit}"
            )
        selected.append(label_records.head(label_limit))
    return pl.concat(selected).sort("id")


def load_multilabel_records(path: Path, split: str, limit: int | None) -> pl.DataFrame:
    records = pl.read_parquet(path)
    missing = MULTILABEL_DATASET_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    selected = records.filter(pl.col("split") == split).sort("id")
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")
        selected = selected.head(limit)
    if selected.is_empty():
        raise ValueError(f"no records found for split {split!r}")
    return selected


def multilabel_labels_from_records(
    records: pl.DataFrame, required_label: str | None = MULTILABEL_NEUTRAL_LABEL
) -> list[str]:
    labels: set[str] = set()
    for encoded_labels in records.get_column("gold_labels_json"):
        labels.update(json.loads(encoded_labels))
    if required_label is not None and required_label not in labels:
        raise ValueError(f"dataset does not contain the required label {required_label!r}")
    return sorted(labels)


def encode_multilabel_targets(
    records: pl.DataFrame, labels: list[str], ignore_neutral: bool
) -> list[list[float]]:
    label_index = {label: index for index, label in enumerate(labels)}
    encoded = [[0.0] * len(labels) for _ in range(records.height)]
    for row_index, encoded_labels in enumerate(records.get_column("gold_labels_json")):
        row_labels = json.loads(encoded_labels)
        if not isinstance(row_labels, list) or not row_labels:
            raise ValueError(f"record {records['id'][row_index]!r} has invalid gold labels")
        for label in row_labels:
            if ignore_neutral and label == MULTILABEL_NEUTRAL_LABEL:
                continue
            if label not in label_index:
                raise ValueError(f"record {records['id'][row_index]!r} contains unknown label {label!r}")
            encoded[row_index][label_index[label]] = 1.0
    return encoded


def apply_fallback_neutral(
    emotion_probabilities: list[list[float]], emotion_labels: list[str], threshold: float
) -> list[list[str]]:
    predictions: list[list[str]] = []
    for probabilities in emotion_probabilities:
        predicted = [
            label
            for label, probability in zip(emotion_labels, probabilities, strict=True)
            if probability >= threshold
        ]
        predictions.append(predicted if predicted else [MULTILABEL_NEUTRAL_LABEL])
    return predictions


def apply_multilabel_threshold(
    probabilities: list[list[float]], labels: list[str], threshold: float
) -> list[list[str]]:
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between zero and one, got {threshold}")
    predictions: list[list[str]] = []
    for row in probabilities:
        if len(row) != len(labels):
            raise ValueError("probability and label counts differ")
        selected = [
            label for label, probability in zip(labels, row, strict=True) if probability >= threshold
        ]
        if not selected:
            selected = [labels[max(range(len(row)), key=row.__getitem__)]]
        predictions.append(selected)
    return predictions


def encode_multilabel_predictions(predictions: list[list[str]], all_labels: list[str]) -> list[list[int]]:
    label_index = {label: index for index, label in enumerate(all_labels)}
    encoded = [[0] * len(all_labels) for _ in predictions]
    for row_index, labels in enumerate(predictions):
        for label in labels:
            encoded[row_index][label_index[label]] = 1
    return encoded


def canonical_labels(labels: list[str]) -> str:
    return json.dumps(sorted(labels), separators=(",", ":"))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch installation")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return torch.device(requested)


def humanize_label(label: str) -> str:
    return label.replace("_", " ").replace("?", "").strip()


def serialize_probabilities(probabilities: dict[str, float]) -> str:
    return json.dumps(probabilities, sort_keys=True, separators=(",", ":"))


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty prediction artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=PREDICTION_SCHEMA).write_parquet(path)


def write_multilabel_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty prediction artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=MULTILABEL_PREDICTION_SCHEMA).write_parquet(path)
