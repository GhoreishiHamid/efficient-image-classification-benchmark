"""Lightweight tests for the results-summary utility (no model training)."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from results_summary import format_summary, load_results  # noqa: E402


class TestResultsSummary(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.csv_path = Path(self.temporary_directory.name) / "comparison.csv"

    def write_csv(self, rows, fields=None):
        fields = fields or [
            "model", "epochs", "test_accuracy_percent",
            "cpu_latency_ms", "checkpoint_size_mib",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def example_rows():
        return [
            {"model": "mobilenet_v3_small", "epochs": "1", "test_accuracy_percent": "76.75",
             "cpu_latency_ms": "12.81", "checkpoint_size_mib": "3.75"},
            {"model": "resnet18", "epochs": "1", "test_accuracy_percent": "83.86",
             "cpu_latency_ms": "65.21", "checkpoint_size_mib": "42.78"},
        ]

    def test_loads_saved_metrics(self):
        self.write_csv(self.example_rows())
        rows = load_results(self.csv_path)
        self.assertEqual(rows["mobilenet_v3_small"]["accuracy"], 76.75)
        self.assertEqual(rows["resnet18"]["epochs"], 1)

    def test_shows_accuracy_gap_and_latency_ratio(self):
        self.write_csv(self.example_rows())
        summary = format_summary(load_results(self.csv_path))
        self.assertIn("+7.11 percentage points", summary)
        self.assertIn("5.09x", summary)
        self.assertIn("preliminary", summary)

    def test_rejects_missing_metric_column(self):
        self.write_csv([], fields=["model", "epochs"])
        with self.assertRaisesRegex(ValueError, "Missing CSV columns"):
            load_results(self.csv_path)

    def test_rejects_zero_latency(self):
        rows = self.example_rows()
        rows[0]["cpu_latency_ms"] = "0"
        self.write_csv(rows)
        with self.assertRaisesRegex(ValueError, "Out-of-range metric"):
            load_results(self.csv_path)


if __name__ == "__main__":
    unittest.main()
