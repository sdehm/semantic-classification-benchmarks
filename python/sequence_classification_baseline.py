"""Fine-tune a standard ModernBERT sequence-classification head for three datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

from common import (
    apply_fallback_neutral,
    apply_multilabel_threshold,
    canonical_labels,
    encode_multilabel_predictions,
    encode_multilabel_targets,
    load_json,
    load_multilabel_records,
    load_records,
    load_stratified_records,
    multilabel_labels_from_records,
    resolve_device,
    serialize_probabilities,
    write_multilabel_predictions,
    write_predictions,
)
from goemotions_bce_baseline import select_threshold as select_goemotions_threshold
from linear_baseline import write_metadata
from multieurlex_bce_baseline import (
    class_positive_weights,
    select_threshold as select_multieurlex_threshold,
    tokenize_documents,
)


TASKS = {
    "banking77": ("data/banking77/records.parquet", "configs/models/modernbert-banking77-sequence.json"),
    "goemotions": ("data/goemotions/records.parquet", "configs/models/modernbert-goemotions-sequence.json"),
    "multieurlex": ("data/multieurlex/records.parquet", "configs/models/modernbert-multieurlex-sequence.json"),
}


class TextTargets(Dataset[tuple[str, int | list[float]]]):
    def __init__(self, records: pl.DataFrame, targets: list[int] | list[list[float]]) -> None:
        self.texts = records.get_column("text").to_list()
        self.targets = targets
        if len(self.texts) != len(targets):
            raise ValueError("text and target counts differ")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int | list[float]]:
        return self.texts[index], self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--positive-weighting", choices=("none", "inverse_sqrt"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_task_records(
    task: str, dataset: Path, split: str, limit: int | None, train_limit: int | None
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str], list[int] | list[list[float]], list[list[float]] | None]:
    if task == "banking77":
        train = load_stratified_records(dataset, "train", train_limit)
        validation = load_records(dataset, "validation", None)
        evaluation = load_records(dataset, split, limit)
        labels = sorted(train.get_column("gold_label").unique().to_list())
        index = {label: number for number, label in enumerate(labels)}
        for records in (validation, evaluation):
            unknown = set(records.get_column("gold_label").to_list()) - set(labels)
            if unknown:
                raise ValueError(f"evaluation contains unknown labels: {sorted(unknown)}")
        return train, validation, evaluation, labels, [
            index[label] for label in train.get_column("gold_label")
        ], None
    train = load_multilabel_records(dataset, "train", train_limit)
    validation = load_multilabel_records(dataset, "validation", None)
    evaluation = load_multilabel_records(dataset, split, limit)
    all_labels = multilabel_labels_from_records(
        train, required_label="neutral" if task == "goemotions" else None
    )
    labels = [label for label in all_labels if label != "neutral"] if task == "goemotions" else all_labels
    targets = encode_multilabel_targets(train, labels, ignore_neutral=task == "goemotions")
    validation_targets = encode_multilabel_targets(validation, all_labels, ignore_neutral=False)
    return train, validation, evaluation, labels, targets, validation_targets


def tokenize_texts(texts: list[str], tokenizer: Any, task: str, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if task != "multieurlex":
        return tokenizer(
            texts, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt"
        )
    if config["context"]["max_chunks"] != 1:
        raise ValueError("sequence classifier requires exactly one chunk per document")
    encoded, offsets, _ = tokenize_documents(texts, tokenizer, config["context"])
    if any(end - start != 1 for start, end in offsets):
        raise ValueError("sequence classifier requires exactly one chunk per document")
    return encoded


def collate(
    batch: list[tuple[str, int | list[float]]],
    tokenizer: Any,
    task: str,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    texts, targets = zip(*batch, strict=True)
    encoded = tokenize_texts(list(texts), tokenizer, task, config)
    encoded["labels"] = torch.tensor(
        targets, dtype=torch.long if task == "banking77" else torch.float32
    )
    return encoded


def save_model(
    path: Path, model: PreTrainedModel, task: str, labels: list[str], config: dict[str, Any]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing trained model: {path}")
    path.mkdir(parents=True)
    model.save_pretrained(path)
    with (path / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump({"task": task, "labels": labels, "model_config": config}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_model(
    path: Path, task: str, labels: list[str], config: dict[str, Any], device: torch.device
) -> PreTrainedModel:
    with (path / "benchmark.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata != {"task": task, "labels": labels, "model_config": config}:
        raise ValueError(f"trained model does not match the configured task or backbone: {path}")
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    if model.config.num_labels != len(labels) or model.config.classifier_pooling != config["classifier_pooling"]:
        raise ValueError(f"trained model has incompatible label count or pooling: {path}")
    return model.to(device)


def train_model(
    model: PreTrainedModel,
    tokenizer: Any,
    task: str,
    train: pl.DataFrame,
    targets: list[int] | list[list[float]],
    config: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    positive_weighting: str | None,
    device: torch.device,
) -> tuple[int, int]:
    training = config["training"]
    loader = DataLoader(
        TextTargets(train, targets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer, task, config),
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=training["weight_decay"])
    steps = len(loader) * epochs
    warmup_steps = math.ceil(steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
    loss_function = (
        nn.BCEWithLogitsLoss(pos_weight=class_positive_weights(targets, positive_weighting, device))
        if task != "banking77" else None
    )
    model.train()
    started = perf_counter()
    completed_steps = 0
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(device)
            encoded = {name: value.to(device) for name, value in batch.items()}
            if loss_function is None:
                loss = model(**encoded, labels=labels).loss
            else:
                loss = loss_function(model(**encoded).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            if completed_steps % 500 == 0 or completed_steps == steps:
                elapsed_seconds = round(perf_counter() - started)
                print(f"{task}: {completed_steps}/{steps} steps ({elapsed_seconds}s)", file=sys.stderr, flush=True)
    return round((perf_counter() - started) * 1_000), warmup_steps


def predict_probabilities(
    model: PreTrainedModel,
    tokenizer: Any,
    task: str,
    texts: list[str],
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[float]], list[int]]:
    model.eval()
    probabilities: list[list[float]] = []
    latencies: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenize_texts(batch, tokenizer, task, config)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        started = perf_counter()
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores = torch.softmax(logits, dim=-1) if task == "banking77" else torch.sigmoid(logits)
        elapsed_ms = round((perf_counter() - started) * 1_000 / len(batch))
        probabilities.extend(scores.cpu().float().tolist())
        latencies.extend([elapsed_ms] * len(batch))
    return probabilities, latencies


def apply_predictions(
    task: str, probabilities: list[list[float]], labels: list[str], threshold: float | None
) -> list[str] | list[list[str]]:
    if task == "banking77":
        return [labels[max(range(len(row)), key=row.__getitem__)] for row in probabilities]
    if threshold is None:
        raise ValueError("multi-label prediction requires a threshold")
    if task == "goemotions":
        return apply_fallback_neutral(probabilities, labels, threshold)
    return apply_multilabel_threshold(probabilities, labels, threshold)


def main() -> None:
    args = parse_args()
    default_dataset, default_config = TASKS[args.task]
    dataset = args.dataset or Path(default_dataset)
    config = load_json(args.model_config or Path(default_config))
    training = config["training"]
    epochs = args.epochs if args.epochs is not None else training["epochs"]
    batch_size = args.train_batch_size if args.train_batch_size is not None else training["batch_size"]
    inference_batch_size = (
        args.inference_batch_size
        if args.inference_batch_size is not None else config["inference"]["batch_size"]
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else training["learning_rate"]
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else training["warmup_ratio"]
    positive_weighting = (
        args.positive_weighting if args.positive_weighting is not None else training.get("positive_weighting")
    )
    output = args.output or Path(
        f"runs/sequence-classification/{args.task}.{args.split}.predictions.parquet"
    )
    model_output = args.model_output or Path(f"runs/sequence-classification/{args.task}.model")
    if epochs < 1 or batch_size < 1 or inference_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if learning_rate <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("learning rate must be positive and warmup ratio must be in [0, 1)")
    if args.task == "banking77" and (args.threshold is not None or args.positive_weighting is not None):
        raise ValueError("threshold and positive weighting apply only to multi-label tasks")
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError("threshold must be between zero and one")
    if config["classifier_pooling"] != "cls":
        raise ValueError("this baseline requires the default ModernBERT CLS classification head")
    expected_loss = "cross_entropy" if args.task == "banking77" else "binary_cross_entropy_with_logits"
    if training["loss"] != expected_loss:
        raise ValueError(f"{args.task} requires {expected_loss} loss")
    if args.task == "multieurlex" and (config["context_variant"] != "leading-256" or config["context"]["max_chunks"] != 1):
        raise ValueError("this baseline compares the single leading-256 context")
    train, validation, evaluation, labels, targets, validation_targets = load_task_records(
        args.task, dataset, args.split, args.limit, args.train_limit
    )
    device = resolve_device(args.device)
    plan = {
        "classifier": "AutoModelForSequenceClassification",
        "task": args.task,
        "backbone": f"{config['id']}@{config['revision']}",
        "classifier_pooling": config["classifier_pooling"],
        "context": (
            config["context_variant"] if args.task == "multieurlex"
            else f"leading-{config['max_length']}"
        ),
        "split": args.split,
        "train_records": train.height,
        "validation_records": validation.height,
        "evaluation_records": evaluation.height,
        "labels": labels,
        "loss": training["loss"],
        "positive_weighting": positive_weighting,
        "epochs": epochs,
        "train_batch_size": batch_size,
        "inference_batch_size": inference_batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "device": str(device),
        "model_output": str(model_output),
        "load_model": str(args.load_model) if args.load_model else None,
        "output": str(output),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    set_seed(training["seed"])
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config["id"], revision=config["revision"])
    if args.load_model:
        model = load_model(args.load_model, args.task, labels, config, device)
        training_milliseconds, warmup_steps = 0, 0
    else:
        if model_output.exists():
            raise FileExistsError(f"refusing to overwrite existing trained model: {model_output}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config["id"],
            revision=config["revision"],
            num_labels=len(labels),
            id2label={index: label for index, label in enumerate(labels)},
            label2id={label: index for index, label in enumerate(labels)},
            classifier_pooling=config["classifier_pooling"],
            problem_type=(
                "single_label_classification" if args.task == "banking77" else "multi_label_classification"
            ),
        ).to(device)
        training_milliseconds, warmup_steps = train_model(
            model, tokenizer, args.task, train, targets, config, epochs, batch_size,
            learning_rate, warmup_ratio, positive_weighting, device,
        )
        save_model(model_output, model, args.task, labels, config)
    model_load_milliseconds = round((perf_counter() - started) * 1_000) - training_milliseconds

    validation_probabilities, _ = predict_probabilities(
        model, tokenizer, args.task, validation.get_column("text").to_list(),
        config, inference_batch_size, device,
    )
    threshold: float | None = None
    if args.task == "banking77":
        validation_macro_f1 = float(
            f1_score(validation.get_column("gold_label").to_list(),
                     apply_predictions(args.task, validation_probabilities, labels, None),
                     average="macro", zero_division=0)
        )
    else:
        all_labels = labels + ["neutral"] if args.task == "goemotions" else labels
        if args.threshold is not None:
            threshold = args.threshold
        elif args.task == "goemotions":
            threshold, _ = select_goemotions_threshold(
                validation_probabilities, validation_targets, labels, all_labels,
                config["inference"]["threshold_candidates"],
            )
        else:
            threshold, _ = select_multieurlex_threshold(
                validation_probabilities, validation_targets, labels,
                config["inference"]["threshold_candidates"],
            )
        validation_macro_f1 = float(
            f1_score(
                validation_targets,
                encode_multilabel_predictions(
                    apply_predictions(args.task, validation_probabilities, labels, threshold), all_labels
                ),
                average="macro", zero_division=0,
            )
        )

    started = perf_counter()
    evaluation_probabilities, latencies = predict_probabilities(
        model, tokenizer, args.task, evaluation.get_column("text").to_list(),
        config, inference_batch_size, device,
    )
    inference_milliseconds = round((perf_counter() - started) * 1_000)
    predictions = apply_predictions(args.task, evaluation_probabilities, labels, threshold)
    model_name = f"{config['id']}@{config['revision']}+sequence-cls+{args.task}"
    if args.task == "banking77":
        rows = [
            {
                "id": row_id,
                "predicted_label": prediction,
                "label_probabilities_json": serialize_probabilities(
                    dict(zip(labels, scores, strict=True))
                ),
                "confidence": max(scores),
                "has_confidence": True,
                "latency_ms": latencies[index],
                "model": model_name,
            }
            for index, (row_id, prediction, scores) in enumerate(zip(
                evaluation.get_column("id"), predictions, evaluation_probabilities, strict=True
            ))
        ]
        write_predictions(output, rows)
    else:
        rows = [
            {
                "id": row_id,
                "predicted_labels_json": canonical_labels(prediction),
                "label_probabilities_json": serialize_probabilities(
                    dict(zip(labels, scores, strict=True))
                ),
                "latency_ms": latencies[index],
                "model": model_name,
            }
            for index, (row_id, prediction, scores) in enumerate(zip(
                evaluation.get_column("id"), predictions, evaluation_probabilities, strict=True
            ))
        ]
        write_multilabel_predictions(output, rows)
    plan.update({
        "dry_run": False,
        "trained": args.load_model is None,
        "model_load_milliseconds": model_load_milliseconds,
        "training_milliseconds": training_milliseconds,
        "warmup_steps": warmup_steps,
        "threshold": threshold,
        "threshold_validation_macro_f1": validation_macro_f1,
        "inference_milliseconds": inference_milliseconds,
    })
    plan["metadata"] = str(output.with_suffix(".metadata.json"))
    write_metadata(output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
