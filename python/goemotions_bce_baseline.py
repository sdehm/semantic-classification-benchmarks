"""Fine-tune ModernBERT with a sigmoid multi-label classification head for GoEmotions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
from linear_baseline import write_metadata


class TextTargets(Dataset[tuple[str, list[float]]]):
    def __init__(self, records: pl.DataFrame, targets: list[list[float]]) -> None:
        self.texts = records.get_column("text").to_list()
        self.targets = targets
        if len(self.texts) != len(self.targets):
            raise ValueError("text and target counts differ")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, list[float]]:
        return self.texts[index], self.targets[index]


class ModernBertSigmoidClassifier(nn.Module):
    def __init__(self, model_id: str, revision: str, label_count: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, revision=revision)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, label_count)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/goemotions/records.parquet"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/models/modernbert-goemotions-bce.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/goemotions-bce/validation.predictions.parquet"),
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("runs/goemotions-bce/validation.model")
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
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


def collate(
    batch: list[tuple[str, list[float]]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts, targets = zip(*batch, strict=True)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(targets, dtype=torch.float32)
    return encoded


def model_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "model.pt", directory / "model.json"


def save_model(
    directory: Path,
    model: ModernBertSigmoidClassifier,
    label_names: list[str],
    model_config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights_path, metadata_path = model_paths(directory)
    temporary_weights = weights_path.with_name(f"{weights_path.name}.tmp")
    temporary_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")
    torch.save(model.state_dict(), temporary_weights)
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": model_config["id"],
                "revision": model_config["revision"],
                "max_length": model_config["max_length"],
                "emotion_labels": label_names,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(temporary_weights, weights_path)
    os.replace(temporary_metadata, metadata_path)


def load_model(
    directory: Path, expected_labels: list[str], model_config: dict[str, Any], device: torch.device
) -> ModernBertSigmoidClassifier:
    weights_path, metadata_path = model_paths(directory)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["model_id"] != model_config["id"] or metadata["revision"] != model_config["revision"]:
        raise ValueError(f"trained model does not match configured backbone: {directory}")
    if metadata["emotion_labels"] != expected_labels:
        raise ValueError(f"trained model has different emotion labels: {directory}")
    model = ModernBertSigmoidClassifier(
        model_config["id"], model_config["revision"], len(expected_labels)
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model


def class_positive_weights(targets: list[list[float]], mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError(f"unsupported positive weighting mode {mode!r}")
    positives = np.asarray(targets, dtype=np.float32).sum(axis=0)
    negatives = len(targets) - positives
    if np.any(positives == 0):
        raise ValueError("cannot calculate weights for an emotion absent from training data")
    return torch.tensor(np.sqrt(negatives / positives), dtype=torch.float32, device=device)


def select_threshold(
    probabilities: list[list[float]],
    gold_targets: list[list[float]],
    emotion_labels: list[str],
    all_labels: list[str],
    candidates: list[float],
) -> tuple[float, float]:
    best_threshold: float | None = None
    best_macro_f1 = -1.0
    for threshold in sorted(set(candidates)):
        if not 0 < threshold < 1:
            raise ValueError(f"threshold candidates must be between zero and one, got {threshold}")
        predictions = apply_fallback_neutral(probabilities, emotion_labels, threshold)
        macro_f1 = float(
            f1_score(
                gold_targets,
                encode_multilabel_predictions(predictions, all_labels),
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


def predict_probabilities(
    model: ModernBertSigmoidClassifier,
    texts: list[str],
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    measure_latency: bool,
) -> tuple[list[list[float]], list[int]]:
    model.eval()
    probabilities: list[list[float]] = []
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
            scores = torch.sigmoid(model(**encoded))
        elapsed_ms = round((perf_counter() - started) * 1_000 / len(batch))
        probabilities.extend(scores.cpu().float().tolist())
        if measure_latency:
            latencies.extend([elapsed_ms] * len(batch))
    return probabilities, latencies


def train_model(
    model: ModernBertSigmoidClassifier,
    tokenizer: Any,
    train: pl.DataFrame,
    targets: list[list[float]],
    model_config: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    positive_weighting: str,
    checkpoint_dir: Path,
    emotion_labels: list[str],
    device: torch.device,
) -> tuple[int, int]:
    training_config = model_config["training"]
    loader = DataLoader(
        TextTargets(train, targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: collate(batch, tokenizer, model_config["max_length"]),
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=training_config["weight_decay"])
    total_steps = len(loader) * epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=class_positive_weights(targets, positive_weighting, device)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    started = perf_counter()
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(device)
            encoded = {name: value.to(device) for name, value in batch.items()}
            loss = loss_function(model(**encoded), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % training_config["checkpoint_steps"] == 0:
                save_model(checkpoint_dir / f"step-{step}", model, emotion_labels, model_config)
    return round((perf_counter() - started) * 1_000), warmup_steps


def main() -> None:
    args = parse_args()
    model_config = load_json(args.model_config)
    training_config = model_config["training"]
    inference_config = model_config["inference"]
    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    train_batch_size = (
        args.train_batch_size if args.train_batch_size is not None else training_config["batch_size"]
    )
    inference_batch_size = (
        args.inference_batch_size
        if args.inference_batch_size is not None
        else inference_config["batch_size"]
    )
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else training_config["learning_rate"]
    )
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else training_config["warmup_ratio"]
    positive_weighting = (
        args.positive_weighting
        if args.positive_weighting is not None
        else training_config["positive_weighting"]
    )
    if epochs < 1 or train_batch_size < 1 or inference_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if learning_rate <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("learning rate must be positive and warmup ratio must be in [0, 1)")
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError(f"threshold must be between zero and one, got {args.threshold}")

    train = load_multilabel_records(args.dataset, "train", None)
    validation = load_multilabel_records(args.dataset, "validation", None)
    evaluation = load_multilabel_records(args.dataset, args.split, args.limit)
    all_labels = multilabel_labels_from_records(train)
    emotion_labels = [label for label in all_labels if label != "neutral"]
    train_targets = encode_multilabel_targets(train, emotion_labels, ignore_neutral=True)
    validation_targets = encode_multilabel_targets(validation, all_labels, ignore_neutral=False)
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir or args.model_output.with_suffix(".checkpoints")
    plan = {
        "classifier": "modernbert-sigmoid-binary-cross-entropy",
        "backbone": f"{model_config['id']}@{model_config['revision']}",
        "split": args.split,
        "train_records": train.height,
        "validation_records": validation.height,
        "evaluation_records": evaluation.height,
        "emotion_labels": emotion_labels,
        "neutral_policy": "fallback when no emotion reaches the selected global threshold",
        "device": str(device),
        "max_length": model_config["max_length"],
        "loss": training_config["loss"],
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "inference_batch_size": inference_batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "positive_weighting": positive_weighting,
        "model_output": str(args.model_output),
        "load_model": str(args.load_model) if args.load_model else None,
        "checkpoint_dir": str(checkpoint_dir),
        "output": str(args.output),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    set_seed(training_config["seed"])
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_config["id"], revision=model_config["revision"])
    if args.load_model:
        model = load_model(args.load_model, emotion_labels, model_config, device)
        trained = False
        training_milliseconds = 0
        warmup_steps = 0
    else:
        if args.model_output.exists():
            raise FileExistsError(f"refusing to overwrite existing trained model: {args.model_output}")
        model = ModernBertSigmoidClassifier(
            model_config["id"], model_config["revision"], len(emotion_labels)
        ).to(device)
        trained = True
        training_milliseconds, warmup_steps = train_model(
            model,
            tokenizer,
            train,
            train_targets,
            model_config,
            epochs,
            train_batch_size,
            learning_rate,
            warmup_ratio,
            positive_weighting,
            checkpoint_dir,
            emotion_labels,
            device,
        )
        save_model(args.model_output, model, emotion_labels, model_config)
    model_load_milliseconds = round((perf_counter() - started) * 1_000) - training_milliseconds

    validation_probabilities, _ = predict_probabilities(
        model,
        validation.get_column("text").to_list(),
        tokenizer,
        device,
        model_config["max_length"],
        inference_batch_size,
        False,
    )
    if args.threshold is None:
        threshold, validation_macro_f1 = select_threshold(
            validation_probabilities,
            validation_targets,
            emotion_labels,
            all_labels,
            inference_config["threshold_candidates"],
        )
        threshold_selection = "validation"
    else:
        threshold = args.threshold
        threshold_selection = "fixed"
        validation_macro_f1 = float(
            f1_score(
                validation_targets,
                encode_multilabel_predictions(
                    apply_fallback_neutral(validation_probabilities, emotion_labels, threshold),
                    all_labels,
                ),
                average="macro",
                zero_division=0,
            )
        )

    started = perf_counter()
    evaluation_probabilities, latencies = predict_probabilities(
        model,
        evaluation.get_column("text").to_list(),
        tokenizer,
        device,
        model_config["max_length"],
        inference_batch_size,
        True,
    )
    inference_milliseconds = round((perf_counter() - started) * 1_000)
    predictions = apply_fallback_neutral(evaluation_probabilities, emotion_labels, threshold)
    rows: list[dict[str, Any]] = []
    for index, row_id in enumerate(evaluation.get_column("id").to_list()):
        rows.append(
            {
                "id": row_id,
                "predicted_labels_json": canonical_labels(predictions[index]),
                "label_probabilities_json": serialize_probabilities(
                    dict(zip(emotion_labels, evaluation_probabilities[index], strict=True))
                ),
                "latency_ms": latencies[index],
                "model": f"{model_config['id']}@{model_config['revision']}+sigmoid-bce",
            }
        )
    write_multilabel_predictions(args.output, rows)
    plan.update(
        {
            "dry_run": False,
            "trained": trained,
            "model_load_milliseconds": model_load_milliseconds,
            "training_milliseconds": training_milliseconds,
            "warmup_steps": warmup_steps,
            "threshold": threshold,
            "threshold_selection": threshold_selection,
            "threshold_validation_macro_f1": validation_macro_f1,
            "inference_milliseconds": inference_milliseconds,
        }
    )
    plan["metadata"] = str(args.output.with_suffix(".metadata.json"))
    write_metadata(args.output, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
