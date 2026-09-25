"""Run the pinned ModernBERT NLI checkpoint as a local zero-shot classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import pipeline

from common import (
    humanize_label,
    load_json,
    load_records,
    resolve_device,
    serialize_probabilities,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/banking77/records.parquet"))
    parser.add_argument("--criteria", type=Path, default=Path("configs/datasets/banking77-label-criteria.json"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/modernbert-base-nli.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/nli/validation.predictions.parquet"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid checkpoint JSON at {path}:{line_number}") from error
            row_id = row.get("id")
            if not isinstance(row_id, str):
                raise ValueError(f"checkpoint row has no string id at {path}:{line_number}")
            if row_id in rows:
                raise ValueError(f"checkpoint contains duplicate id {row_id!r}")
            rows[row_id] = row
    return rows


def append_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    records = load_records(args.dataset, args.split, args.limit)
    criteria = load_json(args.criteria)
    model_config = load_json(args.model_config)
    benchmark_config = model_config["benchmark"]
    batch_size = args.batch_size if args.batch_size is not None else benchmark_config["batch_size"]
    max_length = args.max_length if args.max_length is not None else benchmark_config["max_length"]
    if batch_size < 1:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    if max_length < 1:
        raise ValueError(f"max length must be positive, got {max_length}")
    labels = sorted(criteria)
    candidate_labels = [humanize_label(label) for label in labels]
    candidate_to_label = dict(zip(candidate_labels, labels, strict=True))
    if len(candidate_to_label) != len(labels):
        raise ValueError("humanized labels are not unique")
    checkpoint_path = args.checkpoint or args.output.with_suffix(".checkpoint.jsonl")
    checkpoint_rows = load_checkpoint(checkpoint_path)
    record_ids = set(records.get_column("id").to_list())
    unexpected_ids = checkpoint_rows.keys() - record_ids
    if unexpected_ids:
        raise ValueError(
            f"checkpoint contains ids outside the requested split: {sorted(unexpected_ids)[:3]}"
        )

    plan = {
        "classifier": "zero-shot-nli",
        "model": f"{model_config['id']}@{model_config['revision']}",
        "split": args.split,
        "records": records.height,
        "candidate_labels": len(labels),
        "device": str(resolve_device(args.device)),
        "batch_size": batch_size,
        "max_length": max_length,
        "output": str(args.output),
        "checkpoint": str(checkpoint_path),
        "resumed_records": len(checkpoint_rows),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    started = perf_counter()
    classifier = pipeline(
        "zero-shot-classification",
        model=model_config["id"],
        revision=model_config["revision"],
        device=resolve_device(args.device),
    )
    model_load_milliseconds = round((perf_counter() - started) * 1_000)
    texts = records.get_column("text").to_list()
    ids = records.get_column("id").to_list()
    hypothesis_template = model_config["hypothesis_template"]
    pending = [(row_id, text) for row_id, text in zip(ids, texts, strict=True) if row_id not in checkpoint_rows]
    batches_completed = 0
    inference_started = perf_counter()

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_ids = [row_id for row_id, _ in batch]
        batch_texts = [text for _, text in batch]
        started = perf_counter()
        outputs = classifier(
            batch_texts,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            truncation=True,
            max_length=max_length,
        )
        elapsed_ms = round((perf_counter() - started) * 1_000 / len(batch_texts))
        if isinstance(outputs, dict):
            outputs = [outputs]
        batch_rows: list[dict[str, Any]] = []
        for index, output in enumerate(outputs):
            probabilities = {
                candidate_to_label[candidate]: float(score)
                for candidate, score in zip(output["labels"], output["scores"], strict=True)
            }
            predicted_label = candidate_to_label[output["labels"][0]]
            batch_rows.append(
                {
                    "id": batch_ids[index],
                    "predicted_label": predicted_label,
                    "label_probabilities_json": serialize_probabilities(probabilities),
                    "confidence": float(output["scores"][0]),
                    "has_confidence": True,
                    "latency_ms": elapsed_ms,
                    "model": f"{model_config['id']}@{model_config['revision']}",
                }
            )
        append_checkpoint(checkpoint_path, batch_rows)
        checkpoint_rows.update({row["id"]: row for row in batch_rows})
        batches_completed += 1
        if batches_completed % 10 == 0 or len(checkpoint_rows) == records.height:
            print(
                json.dumps(
                    {
                        "progress": {
                            "completed_records": len(checkpoint_rows),
                            "total_records": records.height,
                        }
                    }
                ),
                flush=True,
            )

    rows = [checkpoint_rows[row_id] for row_id in ids]
    if len(rows) != records.height:
        raise RuntimeError(
            f"checkpoint did not produce all records: {len(rows)} of {records.height}"
        )
    write_predictions(args.output, rows)
    plan.update(
        {
            "dry_run": False,
            "records": len(rows),
            "model_load_milliseconds": model_load_milliseconds,
            "inference_milliseconds": round((perf_counter() - inference_started) * 1_000),
        }
    )
    plan["metadata"] = str(args.output.with_suffix(".metadata.json"))
    write_metadata(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
