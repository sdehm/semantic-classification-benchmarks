"""Compare exact-match error overlap across frozen GoEmotions prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/goemotions/records.parquet"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named multi-label prediction artifact; provide at least two",
    )
    return parser.parse_args()


def parse_prediction_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"prediction must be NAME=PATH, got {value!r}")
    return name, Path(raw_path)


def read_predictions(name: str, path: Path, split_ids: set[str]) -> dict[str, str]:
    predictions = pl.read_parquet(path)
    required = {"id", "predicted_labels_json"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    values = dict(predictions.select("id", "predicted_labels_json").iter_rows())
    unexpected = set(values).difference(split_ids)
    if unexpected:
        raise ValueError(f"{name} contains predictions outside the requested split")
    missing_ids = split_ids.difference(values)
    if missing_ids:
        raise ValueError(f"{name} is missing {len(missing_ids)} predictions")
    return values


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> None:
    args = parse_args()
    parsed_predictions = [parse_prediction_argument(value) for value in args.prediction]
    names = [name for name, _ in parsed_predictions]
    if len(names) < 2:
        raise ValueError("provide at least two named prediction artifacts")
    if len(set(names)) != len(names):
        raise ValueError("prediction names must be unique")

    records = (
        pl.read_parquet(args.dataset)
        .filter(pl.col("split") == args.split)
        .select("id", "gold_labels_json")
        .sort("id")
    )
    if records.is_empty():
        raise ValueError(f"no records found for split {args.split!r}")
    gold_by_id = dict(records.iter_rows())
    split_ids = set(gold_by_id)
    exact_correct: dict[str, set[str]] = {}
    for name, path in parsed_predictions:
        predictions = read_predictions(name, path, split_ids)
        exact_correct[name] = {
            record_id
            for record_id, gold_labels in gold_by_id.items()
            if predictions[record_id] == gold_labels
        }

    all_correct = set.intersection(*exact_correct.values())
    all_incorrect = split_ids.difference(set.union(*exact_correct.values()))
    exclusive = {
        name: len(correct.difference(*(other for other_name, other in exact_correct.items() if other_name != name)))
        for name, correct in exact_correct.items()
    }
    pairwise_error_jaccard = {
        left_name: {
            right_name: jaccard(
                split_ids.difference(left_correct), split_ids.difference(right_correct)
            )
            for right_name, right_correct in exact_correct.items()
        }
        for left_name, left_correct in exact_correct.items()
    }
    report = {
        "split": args.split,
        "examples": len(split_ids),
        "exact_matches": {name: len(correct) for name, correct in exact_correct.items()},
        "all_systems_exact_match": len(all_correct),
        "all_systems_incorrect": len(all_incorrect),
        "exclusive_exact_matches": exclusive,
        "pairwise_exact_error_jaccard": pairwise_error_jaccard,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
