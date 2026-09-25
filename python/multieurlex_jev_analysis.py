"""Select a global MultiEURLEX Jev Noul threshold from cached validation probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score

from common import (
    apply_multilabel_threshold,
    encode_multilabel_predictions,
    encode_multilabel_targets,
    load_multilabel_records,
    multilabel_labels_from_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/multieurlex/records.parquet"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/jev-multieurlex/validation-head-tail-100k.checkpoint.jsonl"),
    )
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def read_probabilities(checkpoint: Path, expected_ids: set[str], labels: list[str]) -> dict[str, list[float]]:
    probabilities: dict[str, list[float]] = {}
    with checkpoint.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            entry = json.loads(line)
            prediction = entry["prediction"]
            record_id = prediction["id"]
            if record_id in probabilities:
                raise ValueError(f"checkpoint line {line_number} repeats record {record_id!r}")
            if record_id not in expected_ids:
                raise ValueError(f"checkpoint line {line_number} has unexpected record {record_id!r}")
            raw = prediction["label_probabilities"]
            if set(raw) != set(labels):
                raise ValueError(f"checkpoint line {line_number} has a different label set")
            probabilities[record_id] = [float(raw[label]) for label in labels]
    missing = expected_ids.difference(probabilities)
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} selected records")
    return probabilities


def threshold_candidates(start: float, stop: float, step: float) -> list[float]:
    if not 0 < start < 1 or not 0 < stop < 1 or step <= 0 or start > stop:
        raise ValueError("threshold range must be within (0, 1) with a positive step")
    candidates: list[float] = []
    threshold = start
    while threshold <= stop + step / 10:
        candidates.append(round(threshold, 10))
        threshold += step
    return candidates


def select_threshold(
    probabilities: list[list[float]],
    gold_targets: list[list[float]],
    labels: list[str],
    candidates: list[float],
) -> tuple[float, float]:
    best_threshold: float | None = None
    best_macro_f1 = -1.0
    for threshold in candidates:
        predictions = apply_multilabel_threshold(probabilities, labels, threshold)
        macro_f1 = float(
            f1_score(
                gold_targets,
                encode_multilabel_predictions(predictions, labels),
                average="macro",
                zero_division=0,
            )
        )
        if macro_f1 > best_macro_f1 or (
            macro_f1 == best_macro_f1
            and (best_threshold is None or threshold > best_threshold)
        ):
            best_threshold = threshold
            best_macro_f1 = macro_f1
    if best_threshold is None:
        raise ValueError("threshold candidate list is empty")
    return best_threshold, best_macro_f1


def main() -> None:
    args = parse_args()
    records = load_multilabel_records(args.dataset, "validation", None)
    all_records = load_multilabel_records(args.dataset, "train", None)
    labels = multilabel_labels_from_records(all_records, required_label=None)
    record_ids = records.get_column("id").to_list()
    probabilities_by_id = read_probabilities(args.checkpoint, set(record_ids), labels)
    probabilities = [probabilities_by_id[record_id] for record_id in record_ids]
    targets = encode_multilabel_targets(records, labels, ignore_neutral=False)
    candidates = threshold_candidates(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    threshold, macro_f1 = select_threshold(probabilities, targets, labels, candidates)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": "validation",
                "examples": records.height,
                "labels": labels,
                "threshold_candidates": candidates,
                "selected_threshold": threshold,
                "selected_macro_f1": macro_f1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
