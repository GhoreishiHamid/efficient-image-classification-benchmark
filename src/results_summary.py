"""Summarize existing benchmark results without retraining the models.

Run from the repository root:
    python src/results_summary.py

Reads results/comparison.csv and does not modify repository files.
"""

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "comparison.csv"
MOBILENET = "mobilenet_v3_small"
RESNET = "resnet18"
FIELDS = {
    "model",
    "epochs",
    "test_accuracy_percent",
    "cpu_latency_ms",
    "checkpoint_size_mib",
}


def load_results(path):
    """Read and validate saved results for the two benchmark models."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Results file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        result = {}
        for row in reader:
            name = row["model"].strip()
            if name not in (MOBILENET, RESNET):
                continue
            if name in result:
                raise ValueError(f"Duplicate model row: {name}")
            try:
                accuracy = float(row["test_accuracy_percent"])
                latency = float(row["cpu_latency_ms"])
                size = float(row["checkpoint_size_mib"])
                epochs = int(row["epochs"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value for {name}") from exc
            if (not all(math.isfinite(x) for x in (accuracy, latency, size))
                    or not 0 <= accuracy <= 100 or latency <= 0 or size <= 0
                    or epochs < 1):
                raise ValueError(f"Out-of-range metric for {name}")
            result[name] = {
                "accuracy": accuracy,
                "latency": latency,
                "size": size,
                "epochs": epochs,
            }

    if set(result) != {MOBILENET, RESNET}:
        raise ValueError("CSV must contain MobileNetV3-Small and ResNet18 results")
    return result


def format_summary(results):
    """Return a human-readable comparison based on measured CSV values."""
    mobile, resnet = results[MOBILENET], results[RESNET]
    accuracy_gap = resnet["accuracy"] - mobile["accuracy"]
    speed_ratio = resnet["latency"] / mobile["latency"]
    size_ratio = resnet["size"] / mobile["size"]
    return "\n".join([
        "Oxford-IIIT Pet | Recorded preliminary benchmark",
        "-" * 49,
        (f"MobileNetV3-Small: {mobile['accuracy']:.2f}% accuracy, "
         f"{mobile['latency']:.2f} ms/image, {mobile['size']:.2f} MiB"),
        (f"ResNet18:         {resnet['accuracy']:.2f}% accuracy, "
         f"{resnet['latency']:.2f} ms/image, {resnet['size']:.2f} MiB"),
        "",
        f"Accuracy difference (ResNet18 - MobileNet): {accuracy_gap:+.2f} percentage points",
        f"CPU latency ratio (ResNet18 / MobileNet): {speed_ratio:.2f}x",
        f"Checkpoint size ratio (ResNet18 / MobileNet): {size_ratio:.2f}x",
        "",
        f"Training epochs (MobileNet / ResNet18): {mobile['epochs']} / {resnet['epochs']}",
        "Note: hardware-specific, single-run preliminary results; not a model ranking.",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS,
                        help="Path to existing comparison.csv")
    args = parser.parse_args()
    try:
        print(format_summary(load_results(args.csv)))
    except ValueError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
