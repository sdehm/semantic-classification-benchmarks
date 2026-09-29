"""Focused checks for the sequence-classification baselines."""

import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, ModernBertConfig

sys.path.insert(0, str(Path(__file__).parent))

from common import apply_fallback_neutral
from sequence_classification_baseline import (
    TASKS,
    apply_predictions,
    collate,
    load_json,
    load_model,
    predict_probabilities,
    save_model,
    train_model,
)


class SequenceClassificationBaselineTest(unittest.TestCase):
    def test_configs_use_matching_backbone_and_training_policies(self) -> None:
        references = {
            "banking77": "configs/models/modernbert-sentence-transformer.json",
            "goemotions": "configs/models/modernbert-goemotions-bce.json",
            "multieurlex": "configs/models/modernbert-multieurlex-bce.json",
        }
        for task, (_, path) in TASKS.items():
            with self.subTest(task=task):
                config = load_json(Path(path))
                reference = load_json(Path(references[task]))
                self.assertEqual(config["classifier_pooling"], "cls")
                self.assertEqual((config["id"], config["revision"]), (reference["id"], reference["revision"]))
                for field in ("epochs", "batch_size", "learning_rate", "warmup_ratio", "weight_decay", "seed"):
                    self.assertEqual(config["training"][field], reference["training"][field])
                self.assertEqual(
                    config["training"]["loss"],
                    "cross_entropy" if task == "banking77" else "binary_cross_entropy_with_logits",
                )
                if task == "multieurlex":
                    self.assertEqual(config["context"], reference["context_candidates"]["leading-256"])
                else:
                    self.assertEqual(config["max_length"], reference["max_length"])
                if task != "banking77":
                    self.assertEqual(
                        config["training"]["positive_weighting"],
                        reference["training"]["positive_weighting"],
                    )
                    self.assertEqual(
                        config["inference"]["threshold_candidates"],
                        reference["inference"]["threshold_candidates"],
                    )

    def test_collate_single_label_uses_integer_targets(self) -> None:
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": torch.ones((len(texts), 4), dtype=torch.long)}

        encoded = collate(
            [("hello", 2), ("world", 1)],
            Tokenizer(),
            "banking77",
            {"max_length": 256},
        )
        self.assertEqual(encoded["labels"].dtype, torch.long)
        self.assertEqual(encoded["labels"].tolist(), [2, 1])

    def test_standard_head_supports_single_and_multilabel_logits(self) -> None:
        for task, targets in (
            ("banking77", torch.tensor([1, 2], dtype=torch.long)),
            ("goemotions", torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32)),
        ):
            with self.subTest(task=task):
                config = ModernBertConfig(
                    vocab_size=80, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                    intermediate_size=64, num_labels=3, max_position_embeddings=128,
                    pad_token_id=0, bos_token_id=1, cls_token_id=1,
                    eos_token_id=2, sep_token_id=2, classifier_pooling="cls",
                    problem_type=(
                        "single_label_classification" if task == "banking77"
                        else "multi_label_classification"
                    ),
                )
                model = AutoModelForSequenceClassification.from_config(config)
                result = model(
                    input_ids=torch.tensor([[1, 4, 2, 0], [1, 5, 6, 2]]),
                    attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
                    labels=targets,
                )
                self.assertEqual(tuple(result.logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(result.loss).item())

    def test_predictions_follow_existing_label_policies(self) -> None:
        self.assertEqual(apply_predictions(
            "banking77", [[0.2, 0.8]], ["a", "b"], None
        ), ["b"])
        self.assertEqual(
            apply_predictions("goemotions", [[0.2, 0.1]], ["joy", "sadness"], 0.5),
            apply_fallback_neutral([[0.2, 0.1]], ["joy", "sadness"], 0.5),
        )
        self.assertEqual(
            apply_predictions("multieurlex", [[0.2, 0.4]], ["law", "trade"], 0.5),
            [["trade"]],
        )

    def test_train_save_and_load_single_label_model(self) -> None:
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 4, 2] for _ in texts]),
                    "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
                }

        config = ModernBertConfig(
            vocab_size=80, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            intermediate_size=64, num_labels=2, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, cls_token_id=1,
            eos_token_id=2, sep_token_id=2, classifier_pooling="cls",
            problem_type="single_label_classification",
        )
        model = AutoModelForSequenceClassification.from_config(config)
        benchmark_config = {
            "max_length": 256,
            "classifier_pooling": "cls",
            "training": {"weight_decay": 0.01, "gradient_clip_norm": 1.0},
        }
        records = pl.DataFrame({"text": ["hello", "world"]})
        elapsed, warmup_steps = train_model(
            model, Tokenizer(), "banking77", records, [0, 1],
            benchmark_config, 1, 2, 2e-5, 0.1, None, torch.device("cpu"),
        )
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(warmup_steps, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            save_model(path, model, "banking77", ["a", "b"], benchmark_config)
            loaded = load_model(
                path, "banking77", ["a", "b"], benchmark_config, torch.device("cpu")
            )
            probabilities, latencies = predict_probabilities(
                loaded, Tokenizer(), "banking77", ["hello", "world"],
                benchmark_config, 2, torch.device("cpu"),
            )
            self.assertEqual(len(probabilities), 2)
            self.assertEqual(len(latencies), 2)
            for row in probabilities:
                self.assertAlmostEqual(sum(row), 1.0, places=6)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_model(path, "banking77", ["b", "a"], benchmark_config, torch.device("cpu"))

    def test_weighted_multilabel_training_uses_sequence_head(self) -> None:
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 4, 2] for _ in texts]),
                    "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
                }

        config = ModernBertConfig(
            vocab_size=80, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            intermediate_size=64, num_labels=3, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, cls_token_id=1,
            eos_token_id=2, sep_token_id=2, classifier_pooling="cls",
            problem_type="multi_label_classification",
        )
        model = AutoModelForSequenceClassification.from_config(config)
        records = pl.DataFrame({"text": ["first", "second"]})
        config_values = {
            "max_length": 256,
            "training": {"weight_decay": 0.01, "gradient_clip_norm": 1.0},
        }
        elapsed, steps = train_model(
            model, Tokenizer(), "goemotions", records,
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], config_values,
            1, 2, 2e-5, 0.1, "inverse_sqrt", torch.device("cpu"),
        )
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(steps, 1)
        probabilities, _ = predict_probabilities(
            model, Tokenizer(), "goemotions", ["first"], config_values, 1, torch.device("cpu")
        )
        self.assertTrue(all(0 < score < 1 for score in probabilities[0]))


if __name__ == "__main__":
    unittest.main()
