"""Focused tests for deterministic MultiEURLEX chunk selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import apply_multilabel_threshold
from multieurlex_bce_baseline import select_chunk_indexes


class MultiEURLEXBCEBaselineTest(unittest.TestCase):
    def test_uniform_chunk_selection_spans_document(self) -> None:
        self.assertEqual(select_chunk_indexes(3, 8, "uniform"), [0, 1, 2])
        self.assertEqual(select_chunk_indexes(10, 4, "uniform"), [0, 3, 6, 9])

    def test_threshold_falls_back_to_highest_probability_label(self) -> None:
        predictions = apply_multilabel_threshold([[0.1, 0.3, 0.2]], ["a", "b", "c"], 0.5)
        self.assertEqual(predictions, [["b"]])


if __name__ == "__main__":
    unittest.main()
