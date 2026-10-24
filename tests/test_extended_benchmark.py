"""Small integrity checks for the final multi-seed report."""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results" / "extended"


class TestExtendedBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (EXT / "experiment_metadata.json").exists():
            raise unittest.SkipTest("Extended experiment not run yet")
        cls.meta = json.loads((EXT / "experiment_metadata.json").read_text(encoding="utf-8"))

    def test_all_requested_combinations_reported(self):
        with (EXT / "per_run_metrics.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        wanted = {(model, str(seed)) for model in self.meta["models"]
                  for seed in self.meta["seeds"]}
        actual = {(r["model"], r["seed"]) for r in rows}
        self.assertEqual(wanted, actual)
        self.assertEqual(len(rows), len(wanted))
        for row in rows:
            self.assertGreaterEqual(float(row["test_accuracy_percent"]), 0)
            self.assertLessEqual(float(row["test_accuracy_percent"]), 100)
            self.assertGreater(float(row["cpu_latency_ms"]), 0)

    def test_per_class_results_complete(self):
        with (EXT / "per_class_f1.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.meta["models"]) * self.meta["classes"])
        self.assertTrue(all(0 <= float(r["f1_mean"]) <= 1 for r in rows))


if __name__ == "__main__":
    unittest.main()
