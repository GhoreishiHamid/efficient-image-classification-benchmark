"""Small, reproducible checks for this preliminary benchmark.

Run from the repository root:
    python -m unittest discover -s tests -v

The data-loader check is skipped when the Oxford-IIIT Pet dataset has
not yet been downloaded. No model training is performed.
"""

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"


class TestDataSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = json.loads((SPLITS / "train_indices.json").read_text(encoding="utf-8"))
        cls.val = json.loads((SPLITS / "val_indices.json").read_text(encoding="utf-8"))

    def test_expected_split_sizes(self):
        self.assertEqual(len(self.train), 2944)
        self.assertEqual(len(self.val), 736)

    def test_no_overlap_or_duplicates(self):
        self.assertEqual(len(self.train), len(set(self.train)))
        self.assertEqual(len(self.val), len(set(self.val)))
        self.assertTrue(set(self.train).isdisjoint(self.val))

    def test_every_trainval_image_is_accounted_for(self):
        self.assertEqual(set(self.train) | set(self.val), set(range(3680)))


class TestBenchmarkReport(unittest.TestCase):
    def test_comparison_csv_is_consistent(self):
        with (RESULTS / "comparison.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(
            {row["model"] for row in rows},
            {"mobilenet_v3_small", "resnet18"},
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(model=row["model"]):
                self.assertGreaterEqual(int(row["epochs"]), 1)
                self.assertEqual(int(row["train_samples"]), 2944)
                self.assertEqual(int(row["validation_samples"]), 736)
                self.assertEqual(int(row["test_samples"]), 3669)
                for key in (
                    "best_validation_accuracy_percent",
                    "test_accuracy_percent",
                ):
                    self.assertGreaterEqual(float(row[key]), 0)
                    self.assertLessEqual(float(row[key]), 100)
                self.assertGreater(float(row["cpu_latency_ms"]), 0)
                self.assertGreater(float(row["checkpoint_size_mib"]), 0)


class TestDataLoader(unittest.TestCase):
    def test_sample_batch_shape_and_class_labels(self):
        if not (ROOT / "data").is_dir():
            self.skipTest("Download the Oxford-IIIT Pet dataset first")

        # Import here so tests of saved splits/results work without PyTorch.
        from src.dataloaders import get_data_loaders

        train_loader, val_loader = get_data_loaders(batch_size=4)
        self.assertEqual(len(train_loader.dataset), 2944)
        self.assertEqual(len(val_loader.dataset), 736)

        images, labels = next(iter(val_loader))
        self.assertEqual(tuple(images.shape), (4, 3, 224, 224))
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertTrue(((labels >= 0) & (labels < 37)).all().item())


if __name__ == "__main__":
    unittest.main()
