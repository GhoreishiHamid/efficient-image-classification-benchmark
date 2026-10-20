r"""Analyze saved checkpoints without retraining either model.

Run from the repository root on Windows:
    .\.venv\Scripts\python.exe src\analyze_results.py

Writes per-class metrics, confusion matrices, and a short analysis to results/.
Also inserts a concise, rerunnable results section into README.md.
The official test split is used only for *descriptive* analysis, not model tuning.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from dataloaders import MEAN, STD
from models import build_mobilenet
from train import build_resnet18


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"
README_PATH = ROOT / "README.md"
MODELS = (
    ("mobilenet_v3_small", "MobileNetV3-Small", build_mobilenet),
    ("resnet18", "ResNet18", build_resnet18),
)
START_MARKER = "<!-- classwise-analysis:start -->"
END_MARKER = "<!-- classwise-analysis:end -->"


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def get_predictions(model, loader):
    truth, predicted = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            truth.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
    return np.asarray(truth), np.asarray(predicted)


def top_confusions(cm, class_names, k=5):
    candidates = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                candidates.append((int(cm[true_idx, pred_idx]), class_names[true_idx], class_names[pred_idx]))
    return sorted(candidates, key=lambda value: (-value[0], value[1], value[2]))[:k]


def save_f1_chart(per_class_rows, class_names):
    score_map = {
        (row["model"], row["class_name"]): float(row["f1_score"])
        for row in per_class_rows
    }
    order = sorted(
        class_names,
        key=lambda name: sum(score_map[(slug, name)] for slug, _, _ in MODELS) / len(MODELS),
    )
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 13.5))
    bar_height = 0.38
    for i, (slug, display, _) in enumerate(MODELS):
        values = [score_map[(slug, cls)] for cls in order]
        ax.barh(y + (i - 0.5) * bar_height, values, height=bar_height, label=display)
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score (official test split)")
    ax.set_title("Per-class F1: same held-out test images")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classwise_f1.png", dpi=160)
    plt.close(fig)


def save_confusions(confusions):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), constrained_layout=True)
    for ax, (slug, name, _) in zip(axes, MODELS):
        cm = confusions[slug].astype(float)
        sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, sums, out=np.zeros_like(cm), where=sums != 0)
        im = ax.imshow(normalized, interpolation="nearest", vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        ax.set_xticks(np.arange(0, cm.shape[0], 5))
        ax.set_yticks(np.arange(0, cm.shape[0], 5))
    fig.colorbar(im, ax=list(axes), shrink=0.75, label="Fraction within true class")
    fig.suptitle("Row-normalized confusion matrices (37 classes)")
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def update_readme(summary_rows):
    if not README_PATH.exists():
        print("README.md not found; skipping README update.")
        return
    blocks = [
        START_MARKER,
        "## Class-wise error analysis",
        "",
        "The saved **single-epoch** checkpoints were evaluated again on the **same official test split** to examine class-level errors. "
        "This is descriptive analysis, **not an independent experiment or additional training**.",
        "",
        "| Model | Test accuracy | Macro F1 | Weighted F1 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        blocks.append(
            f"| {row['display_name']} | {row['test_accuracy_percent']:.2f}% | "
            f"{row['macro_f1_percent']:.2f}% | {row['weighted_f1_percent']:.2f}% |"
        )
    blocks.extend([
        "",
        "![Per-class F1 for the two models](results/classwise_f1.png)",
        "",
        "[Detailed analysis](results/class_analysis.md) · "
        "[Per-class metrics CSV](results/per_class_metrics.csv) · "
        "[Confusion matrices](results/confusion_matrices.png)",
        "",
        r"To reproduce the analysis after training, run `.\.venv\Scripts\python.exe src\analyze_results.py` "
        "from the repository root. It does not retrain the models.",
        "",
        "**Interpretation:** Macro F1 gives equal weight to every breed; weighted F1 accounts for class support. "
        "These test-set observations must not be used to choose new hyperparameters or model variants.",
        END_MARKER,
    ])
    section = "\n".join(blocks)
    text = README_PATH.read_text(encoding="utf-8")
    if START_MARKER in text and END_MARKER in text:
        before, tail = text.split(START_MARKER, 1)
        _, after = tail.split(END_MARKER, 1)
        updated = before.rstrip() + "\n\n" + section + after
    else:
        anchor = "## Dataset and evaluation protocol"
        if anchor in text:
            updated = text.replace(anchor, section + "\n\n" + anchor, 1)
        else:
            updated = text.rstrip() + "\n\n" + section + "\n"
    # Keep the displayed repository tree aligned with the added analysis files.
    updated = updated.replace(
        "│   └── train.py            # Training, test evaluation, CPU benchmark, reporting",
        "│   ├── train.py            # Training, test evaluation, CPU benchmark, reporting\n"
        "│   └── analyze_results.py # Class-level evaluation of saved checkpoints",
    )
    updated = updated.replace(
        "│   └── summary.md",
        "│   ├── summary.md\n"
        "│   ├── class_analysis.md\n"
        "│   ├── classification_metrics.csv\n"
        "│   ├── per_class_metrics.csv\n"
        "│   ├── classwise_f1.png\n"
        "│   └── confusion_matrices.png",
    )
    if updated != text:
        README_PATH.write_text(updated, encoding="utf-8")


def main():
    print("Analyzing saved models (no training or new checkpoints)...", flush=True)
    for slug, _, _ in MODELS:
        checkpoint = CHECKPOINTS_DIR / f"{slug}_best.pth"
        if not checkpoint.is_file():
            raise SystemExit(
                f"Missing {checkpoint}. Keep the checkpoints from your earlier train.py run. "
                "Do not rerun training just for this analysis."
            )
    RESULTS_DIR.mkdir(exist_ok=True)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = OxfordIIITPet(
        root=ROOT / "data", split="test", target_types="category",
        transform=test_transform, download=False,
    )
    class_names = list(dataset.classes)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    summary_rows, per_class_rows, confusions, error_pairs = [], [], {}, {}
    for slug, display, factory in MODELS:
        print(f"Evaluating {display}...", flush=True)
        model = factory()
        state = torch.load(CHECKPOINTS_DIR / f"{slug}_best.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        true, predicted = get_predictions(model, loader)
        if len(true) != len(dataset):
            raise RuntimeError("Test prediction count does not match dataset length.")
        p, r, f1, support = precision_recall_fscore_support(
            true, predicted, labels=list(range(len(class_names))), zero_division=0
        )
        accuracy = float(accuracy_score(true, predicted) * 100)
        macro_f1 = float(f1_score(true, predicted, average="macro", zero_division=0) * 100)
        weighted_f1 = float(f1_score(true, predicted, average="weighted", zero_division=0) * 100)
        summary_rows.append({
            "model": slug,
            "display_name": display,
            "test_samples": int(len(dataset)),
            "test_accuracy_percent": round(accuracy, 2),
            "macro_f1_percent": round(macro_f1, 2),
            "weighted_f1_percent": round(weighted_f1, 2),
        })
        for idx, name in enumerate(class_names):
            per_class_rows.append({
                "model": slug,
                "class_index": idx,
                "class_name": name,
                "test_support": int(support[idx]),
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1_score": round(float(f1[idx]), 4),
            })
        confusions[slug] = confusion_matrix(true, predicted, labels=list(range(len(class_names))))
        error_pairs[slug] = top_confusions(confusions[slug], class_names)
        print(f"{display}: test accuracy={accuracy:.2f}%, macro F1={macro_f1:.2f}%", flush=True)
        del model

    save_csv(
        RESULTS_DIR / "classification_metrics.csv", summary_rows,
        ["model", "display_name", "test_samples", "test_accuracy_percent", "macro_f1_percent", "weighted_f1_percent"],
    )
    save_csv(
        RESULTS_DIR / "per_class_metrics.csv", per_class_rows,
        ["model", "class_index", "class_name", "test_support", "precision", "recall", "f1_score"],
    )
    save_f1_chart(per_class_rows, class_names)
    save_confusions(confusions)

    report = [
        "# Class-wise Error Analysis",
        "",
        "This analysis uses the existing single-epoch checkpoints and the same official 3,669-image test split. "
        "No additional training was performed. This is not an independent replication.",
        "",
        "| Model | Test accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(f"| {row['display_name']} | {row['test_accuracy_percent']:.2f}% | "
                      f"{row['macro_f1_percent']:.2f}% | {row['weighted_f1_percent']:.2f}% |")
    report.extend([
        "",
        "![Per-class F1](classwise_f1.png)",
        "",
        "![Row-normalized confusion matrices](confusion_matrices.png)",
        "",
        "## Most frequent class confusions",
        "",
        "These are raw misclassification counts, not error rates normalized by class size.",
    ])
    for slug, display, _ in MODELS:
        report.extend(["", f"### {display}", "", "| True breed | Predicted breed | Count |", "|---|---|---:|"])
        for count, source, target in error_pairs[slug]:
            report.append(f"| {source} | {target} | {count} |")
    report.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- Macro F1 averages class F1 scores equally; weighted F1 weights classes by support.",
        "- A confusion matrix shows which labels the trained models mix up, not why errors occur.",
        "- The test set was already used for the overall accuracy report. Class-wise analysis here is descriptive only.",
        "- Do not tune architectures or hyperparameters against these test-set results.",
        "- Results remain limited to one training epoch, one data split/seed, and CPU evaluation.",
        "",
        "Raw numbers are in [classification_metrics.csv](classification_metrics.csv) and "
        "[per_class_metrics.csv](per_class_metrics.csv).",
    ])
    (RESULTS_DIR / "class_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    update_readme(summary_rows)
    print("\nDONE: results/classification_metrics.csv")
    print("DONE: results/per_class_metrics.csv")
    print("DONE: results/classwise_f1.png")
    print("DONE: results/confusion_matrices.png")
    print("DONE: results/class_analysis.md")
    print("DONE: README.md updated with class-wise analysis")


if __name__ == "__main__":
    main()
