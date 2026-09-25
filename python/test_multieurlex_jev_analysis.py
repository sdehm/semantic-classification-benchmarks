"""Focused tests for MultiEURLEX Jev threshold selection."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from multieurlex_jev_analysis import parse_args, select_threshold, threshold_candidates


class MultiEURLEXJevAnalysisTest(unittest.TestCase):
    def test_select_threshold_prefers_higher_value_on_tie(self) -> None:
        threshold, macro_f1 = select_threshold(
            probabilities=[[0.9, 0.1], [0.1, 0.4]],
            gold_targets=[[1.0, 0.0], [0.0, 1.0]],
            labels=["first", "second"],
            candidates=[0.2, 0.3],
        )
        self.assertEqual(threshold, 0.3)
        self.assertEqual(macro_f1, 1.0)

    def test_threshold_candidates_include_bounds(self) -> None:
        self.assertEqual(threshold_candidates(0.1, 0.3, 0.1), [0.1, 0.2, 0.3])

    def test_cli_rejects_test_threshold_selection(self) -> None:
        with patch.object(sys, "argv", ["multieurlex_jev_analysis.py", "--split", "test"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
