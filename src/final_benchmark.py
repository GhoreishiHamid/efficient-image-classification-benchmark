r"""Extended CPU transfer-learning benchmark for Oxford-IIIT Pet.

Run from the repository root (Windows PowerShell):
    .\.venv\Scripts\python.exe src\final_benchmark.py --smoke-test
    .\.venv\Scripts\python.exe src\final_benchmark.py --epochs 3 --seeds 42 123 2026

The initial one-epoch results remain untouched in results/.
New outputs are saved separately in results/extended/.
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

import torchvision
from tqdm import tqdm
from dataloaders import MEAN, STD, get_data_loaders
from models import build_mobilenet
from train import build_resnet18

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "extended"
RECORDS = RESULTS / "run_records"
WEIGHTS = ROOT / "checkpoints" / "extended"
README = ROOT / "README.md"
DOCS = ROOT / "docs" / "extended_benchmark_protocol.md"
MODELS = (
    ("mobilenet_v3_small", "MobileNetV3-Small", build_mobilenet),
    ("resnet18", "ResNet18", build_resnet18),
)
README_START = "<!-- extended-benchmark:start -->"
README_END = "<!-- extended-benchmark:end -->"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def evaluate(model, loader, with_predictions=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    if len(true) == 0:
        raise RuntimeError("Evaluation data loader is empty.")
    acc = float(np.mean(true == predicted) * 100.0)
    if with_predictions:
        return acc, true, predicted
    return acc


def cpu_latency_ms(model, samples, warmups=5, repeats=20):
    """Median single-image forward latency across five fixed preprocessed test images."""
    model.eval()
    timings = []
    with torch.inference_mode():
        for sample in samples:
            for _ in range(warmups):
                model(sample)
            for _ in range(repeats):
                start = time.perf_counter()
                model(sample)
                timings.append((time.perf_counter() - start) * 1000.0)
    return float(median(timings))


def training_run(slug, display, factory, seed, epochs, batch_size, test_loader, test_images,
                 test_classes, smoke=False):
    seed_everything(seed)
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)
    model = factory().to("cpu")
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val = -1.0
    best_epoch = 0
    checkpoint = WEIGHTS / f"{slug}_seed{seed}_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        # BatchNorm statistics must remain frozen in frozen feature extractors.
        for layer in model.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()

        running_loss, seen = 0.0, 0
        for batch_id, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"{display} seed {seed} epoch {epoch}/{epochs}", leave=False)
        ):
            if smoke and batch_id >= 1:
                break
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            running_loss += float(loss.item()) * n
            seen += n

        if seen == 0:
            raise RuntimeError("Training data loader is empty.")
        if smoke:
            images, labels = next(iter(val_loader))
            model.eval()
            with torch.inference_mode():
                val_acc = float((model(images).argmax(1) == labels).float().mean() * 100.0)
        else:
            val_acc = evaluate(model, val_loader)

        avg_loss = running_loss / seen
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "validation_accuracy_percent": round(val_acc, 4),
        })
        print(f"{display} | seed={seed} | epoch {epoch}/{epochs} | "
              f"loss={avg_loss:.4f} | val accuracy={val_acc:.2f}%", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            if not smoke:
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)

    if smoke:
        print(f"SMOKE TEST OK: {display}. No results or checkpoints saved.", flush=True)
        return None

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    accuracy, true, predicted = evaluate(model, test_loader, with_predictions=True)
    labels = list(range(len(test_classes)))
    per_class_f1 = f1_score(true, predicted, labels=labels, average=None, zero_division=0)
    macro_f1 = float(f1_score(true, predicted, labels=labels,
                              average="macro", zero_division=0) * 100.0)
    cm = confusion_matrix(true, predicted, labels=labels)
    latency = cpu_latency_ms(model, test_images)

    record = {
        "model": slug,
        "display_name": display,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "best_validation_accuracy_percent": round(best_val, 4),
        "test_accuracy_percent": round(accuracy, 4),
        "macro_f1_percent": round(macro_f1, 4),
        "cpu_latency_ms": round(latency, 4),
        "checkpoint_size_mib": round(checkpoint.stat().st_size / (1024 ** 2), 4),
        "history": history,
        "per_class_f1": [round(float(v), 6) for v in per_class_f1],
        "confusion_matrix": cm.tolist(),
    }
    RECORDS.mkdir(parents=True, exist_ok=True)
    record_path = RECORDS / f"{slug}_seed{seed}.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SAVED: {record_path.relative_to(ROOT)}", flush=True)
    return record


def avg_sd(numbers):
    arr = np.asarray(numbers, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def plot_comparison(summary):
    names = [r["display_name"] for r in summary]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    series = [
        ("test_accuracy_mean", "test_accuracy_sd", "Test accuracy (%)"),
        ("cpu_latency_median_ms", None, "CPU latency (ms/image)"),
        ("checkpoint_size_mean_mib", None, "Checkpoint size (MiB)"),
    ]
    for ax, (key, sd_key, title) in zip(axes, series):
        means = [row[key] for row in summary]
        errors = [row[sd_key] for row in summary] if sd_key else None
        ax.bar(names, means, yerr=errors, capsize=4 if errors else 0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Oxford-IIIT Pet: Extended CPU Benchmark")
    fig.savefig(RESULTS / "benchmark_comparison.png", dpi=160)
    plt.close(fig)


def plot_learning_curves(records):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, display, _ in MODELS:
        subset = [r for r in records if r["model"] == name]
        epochs = [h["epoch"] for h in subset[0]["history"]]
        losses = np.asarray([[h["train_loss"] for h in r["history"]] for r in subset])
        vals = np.asarray([[h["validation_accuracy_percent"] for h in r["history"]] for r in subset])
        for ax, array in [(axes[0], losses), (axes[1], vals)]:
            mean = array.mean(axis=0)
            sd = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
            ax.plot(epochs, mean, marker="o", label=display)
            ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.14)
    axes[0].set_title("Training loss (mean ± SD across seeds)")
    axes[1].set_title("Validation accuracy (mean ± SD across seeds)")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(RESULTS / "learning_curves.png", dpi=160)
    plt.close(fig)


def plot_classwise(per_class, class_names):
    score = {(r["model"], r["class_name"]): r["f1_mean"] for r in per_class}
    order = sorted(class_names, key=lambda c: sum(score[(m, c)] for m, _, _ in MODELS))
    fig, ax = plt.subplots(figsize=(11, 13.2), constrained_layout=True)
    y = np.arange(len(order))
    for index, (slug, display, _) in enumerate(MODELS):
        ax.barh(y + (index - 0.5) * 0.38,
                [score[(slug, c)] for c in order], height=0.38, label=display)
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean per-class F1 across seeds (test split)")
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="lower right")
    fig.savefig(RESULTS / "classwise_f1.png", dpi=160)
    plt.close(fig)


def plot_confusions(records, seeds):
    # Pre-specified seed (first listed), NOT the highest test-scoring run.
    selected = [next(r for r in records if r["model"] == slug and r["seed"] == seeds[0])
                for slug, _, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, record in zip(axes, selected):
        cm = np.asarray(record["confusion_matrix"], dtype=float)
        den = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, den, out=np.zeros_like(cm), where=den != 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{record['display_name']} (seed {record['seed']})")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        ax.set_xticks(np.arange(0, len(cm), 5))
        ax.set_yticks(np.arange(0, len(cm), 5))
    fig.colorbar(im, ax=list(axes), shrink=0.72, label="Fraction of true class")
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def markdown_table(summary, decimals=2):
    lines = [
        "| Metric | MobileNetV3-Small | ResNet18 |",
        "|:--|--:|--:|",
    ]
    by_model = {r["model"]: r for r in summary}
    small, large = by_model["mobilenet_v3_small"], by_model["resnet18"]
    rows = [
        ("Test accuracy, mean ± SD", "test_accuracy_mean", "test_accuracy_sd", "%"),
        ("Macro F1, mean ± SD", "macro_f1_mean", "macro_f1_sd", "%"),
        ("Validation accuracy, mean ± SD", "validation_accuracy_mean", "validation_accuracy_sd", "%"),
    ]
    for label, value, error, unit in rows:
        lines.append(f"| {label} | {small[value]:.2f} ± {small[error]:.2f}{unit} | "
                     f"{large[value]:.2f} ± {large[error]:.2f}{unit} |")
    lines.append(f"| CPU latency (median of run medians) | {small['cpu_latency_median_ms']:.2f} ms | "
                 f"{large['cpu_latency_median_ms']:.2f} ms |")
    lines.append(f"| Checkpoint size (mean) | {small['checkpoint_size_mean_mib']:.2f} MiB | "
                 f"{large['checkpoint_size_mean_mib']:.2f} MiB |")
    return "\n".join(lines)


def update_readme(summary, args):
    block = "\n".join([
        README_START,
        "## Extended multi-seed benchmark (main follow-up)",
        "",
        f"Both models were trained for **{args.epochs} epochs per run** with the same "
        f"train/validation split and **{len(args.seeds)} pre-specified random seeds** "
        f"(`{', '.join(str(s) for s in args.seeds)}`). Checkpoints were selected "
        "using validation accuracy only; held-out test metrics were computed afterwards. "
        "This follow-up is more informative than the original single-epoch pilot, "
        "but remains a small CPU-only benchmark.",
        "",
        markdown_table(summary),
        "",
        "![Extended benchmark comparison](results/extended/benchmark_comparison.png)",
        "",
        "![Training and validation learning curves](results/extended/learning_curves.png)",
        "",
        "[Extended analysis](results/extended/report.md) · "
        "[Per-run CSV](results/extended/per_run_metrics.csv) · "
        "[Per-class F1](results/extended/classwise_f1.png) · "
        "[Confusion matrices](results/extended/confusion_matrices.png) · "
        "[Detailed protocol](docs/extended_benchmark_protocol.md)",
        "",
        "**Interpretation:** Test-set differences describe these runs and this hardware. "
        "The original single-epoch experiment remains below for comparison; "
        "its class-wise figures refer to the earlier checkpoints, not to the new runs. "
        "The official test set was already viewed in that pilot, so it is not a fresh, "
        "previously unseen holdout for the project as a whole.",
        README_END,
    ])
    if not README.exists():
        print("README missing, skipping automatic README update.")
        return
    contents = README.read_text(encoding="utf-8")
    # Replace obsolete pilot-only scope statement without damaging other README sections.
    contents = contents.replace(
        "> **Study scope:** This is an **initial, single-epoch experiment**, not a comprehensive model ranking. The reported measurements are specific to the experimental setup and CPU used.",
        "> **Study scope:** This repository contains an original single-epoch pilot and a follow-up multi-seed CPU benchmark. Results are descriptive, hardware-dependent, and not a general architecture ranking."
    )
    if README_START in contents and README_END in contents:
        prefix, tail = contents.split(README_START, 1)
        _, suffix = tail.split(README_END, 1)
        new_text = prefix.rstrip() + "\n\n" + block + suffix
    else:
        # Rename only the pilot header so readers do not confuse two result sets.
        contents = contents.replace("## Results\n", "## Original single-epoch pilot results\n", 1)
        anchor = "## Original single-epoch pilot results"
        if anchor in contents:
            new_text = contents.replace(anchor, block + "\n\n" + anchor, 1)
        else:
            new_text = contents.rstrip() + "\n\n" + block + "\n"
    README.write_text(new_text, encoding="utf-8")


def write_outputs(records, class_names, args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_run_cols = [
        "model", "seed", "epochs", "batch_size", "best_epoch", "train_samples",
        "validation_samples", "test_samples", "best_validation_accuracy_percent",
        "test_accuracy_percent", "macro_f1_percent", "cpu_latency_ms", "checkpoint_size_mib",
    ]
    save_csv(RESULTS / "per_run_metrics.csv", records, per_run_cols)
    history_rows = [{"model": r["model"], "seed": r["seed"], **h}
                    for r in records for h in r["history"]]
    save_csv(RESULTS / "training_history.csv", history_rows,
             ["model", "seed", "epoch", "train_loss", "validation_accuracy_percent"])

    summary = []
    per_class = []
    for slug, display, _ in MODELS:
        subset = [r for r in records if r["model"] == slug]
        acc_m, acc_s = avg_sd([r["test_accuracy_percent"] for r in subset])
        f1_m, f1_s = avg_sd([r["macro_f1_percent"] for r in subset])
        val_m, val_s = avg_sd([r["best_validation_accuracy_percent"] for r in subset])
        size_m, _ = avg_sd([r["checkpoint_size_mib"] for r in subset])
        summary.append({
            "model": slug,
            "display_name": display,
            "seeds": len(subset),
            "epochs_per_seed": args.epochs,
            "test_accuracy_mean": round(acc_m, 4),
            "test_accuracy_sd": round(acc_s, 4),
            "macro_f1_mean": round(f1_m, 4),
            "macro_f1_sd": round(f1_s, 4),
            "validation_accuracy_mean": round(val_m, 4),
            "validation_accuracy_sd": round(val_s, 4),
            "cpu_latency_median_ms": round(float(median([r["cpu_latency_ms"] for r in subset])), 4),
            "checkpoint_size_mean_mib": round(size_m, 4),
        })
        for index, name in enumerate(class_names):
            values = [r["per_class_f1"][index] for r in subset]
            mean, sd = avg_sd(values)
            per_class.append({
                "model": slug,
                "class_index": index,
                "class_name": name,
                "f1_mean": round(mean, 5),
                "f1_sd": round(sd, 5),
            })

    summary_cols = list(summary[0].keys())
    save_csv(RESULTS / "summary_metrics.csv", summary, summary_cols)
    save_csv(RESULTS / "per_class_f1.csv", per_class,
             ["model", "class_index", "class_name", "f1_mean", "f1_sd"])
    plot_comparison(summary)
    plot_learning_curves(records)
    plot_classwise(per_class, class_names)
    plot_confusions(records, args.seeds)

    info = {
        "dataset": "Oxford-IIIT Pet",
        "classes": len(class_names),
        "models": [slug for slug, _, _ in MODELS],
        "seeds": args.seeds,
        "epochs_per_seed": args.epochs,
        "batch_size": args.batch_size,
        "train_split": "Saved stratified split, 80:20 from official trainval",
        "checkpoint_selection": "Highest validation accuracy within each run",
        "test_usage": "Evaluated after within-run validation-based model selection; this test split was also seen in the earlier pilot",
        "optimizer": "Adam, lr=0.001",
        "loss": "CrossEntropyLoss",
        "weights": "ImageNet pretrained; frozen feature extractor; linear head trained",
        "latency": "CPU; median of 20 forward passes each on 5 fixed test images after 5 warmups per image; preprocessing and I/O excluded",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "operating_system": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "not reported",
        "cpu_threads": torch.get_num_threads(),
    }
    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Extended Multi-Seed CPU Benchmark", "",
        f"**Setup:** {len(args.seeds)} pre-specified seeds ({', '.join(map(str, args.seeds))}), "
        f"{args.epochs} epochs per seed and model, batch size {args.batch_size}. "
        "The train/validation split is fixed; augmentation and head initialization vary with the seed.",
        "", markdown_table(summary), "",
        "![Comparison](benchmark_comparison.png)", "",
        "![Learning curves](learning_curves.png)", "",
        "![Per-class F1](classwise_f1.png)", "",
        f"![Confusion matrices for pre-specified seed {args.seeds[0]}](confusion_matrices.png)", "",
        "## Interpretation and limitations", "",
        "- Both networks use frozen ImageNet feature extractors; only replacement linear heads are trained.",
        "- Both models use the same class labels, split indices, batch size, optimizer, and epoch budget.",
        "- Checkpoints are selected on validation accuracy. Test scores were recorded after selection and not used to tune hyperparameters.",
        "- The reported mean and sample standard deviation describe only the pre-specified runs, not a confidence interval.",
        "- CPU latency excludes preprocessing and disk I/O. It is hardware- and environment-dependent; results are not mobile-phone benchmarks.",
        "- No paired statistical significance test, fine-tuning study, energy benchmark or cross-hardware replication is claimed.",
        "- Confusion matrices show the FIRST pre-specified seed, not a run chosen using test performance.",
        "- Earlier `results/` figures are for the original one-epoch pilot and are not interchangeable with this follow-up.",
        "- The official test set was also assessed in the earlier pilot; it is therefore not a fresh, previously unseen holdout across the entire project.",
        "", "## Reproduce", "",
        "From the repository root on Windows PowerShell:", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe src\\final_benchmark.py "
        f"--epochs {args.epochs} --seeds {' '.join(map(str, args.seeds))}",
        "```", "",
        "The script can resume already completed model/seed runs (provided the saved "
        "record and local checkpoint still exist). See the [protocol](../../docs/extended_benchmark_protocol.md).", "",
    ]
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    protocol = [
        "# Extended Benchmark Protocol", "",
        "This is a follow-up to the original single-epoch Oxford-IIIT Pet experiment.", "",
        "## Design", "",
        "- Models: pretrained MobileNetV3-Small and ResNet18 (frozen backbones).",
        "- Fixed stratified training/validation indices: `splits/train_indices.json` and `splits/val_indices.json`.",
        "- Official test split held out for evaluation after selecting each run's best epoch by validation accuracy.",
        f"- Exactly {args.epochs} epochs each, with seeds {', '.join(map(str, args.seeds))}, "
        f"batch size {args.batch_size}, Adam LR=0.001, cross-entropy loss.",
        "- CPU only. Validation/test use deterministic transforms; training uses random crop and flip.",
        "- BatchNorm running statistics remain frozen; only learnable parameters in new linear heads are optimized.",
        "- Latency uses 5 fixed preprocessed test images, 5 warmups and 20 timed single-image forwards per image, "
        "summarized by median; excludes loading/preprocessing.",
        "- Results present mean ± sample SD across runs, not inferential confidence intervals.",
        "", "## Outputs", "",
        "- `results/extended/summary_metrics.csv`: per-model descriptive aggregates.",
        "- `results/extended/per_run_metrics.csv`: per-model and per-seed results.",
        "- `results/extended/training_history.csv`: training loss and validation accuracy for each epoch.",
        "- `results/extended/per_class_f1.csv`: mean/SD across runs per breed.",
        "- `results/extended/run_records/`: self-contained per-run JSON metrics, including confusion matrices.",
        "- `results/extended/*.png`: comparisons, learning curves, per-class F1, and confusion matrices.",
        "- `checkpoints/extended/`: local model checkpoints; excluded from Git.",
        "", "## Caveats", "",
        "This benchmark does not demonstrate mobile deployment or conclusively rank architectures. "
        "The official test images were already evaluated in the earlier pilot, and appear in each seed's evaluation; "
        "seed-level results are not independent datasets. "
        "The historical one-epoch scores remain in `results/` and use a different, shorter timing procedure.",
        "", "## Repeat", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe src\\final_benchmark.py "
        f"--epochs {args.epochs} --seeds {' '.join(map(str, args.seeds))}",
        "```", "",
        "If an interrupted run is resumed, completed combinations with matching settings "
        "and present local checkpoints are reused. `--force` retrains all combinations.",
    ]
    DOCS.parent.mkdir(exist_ok=True)
    DOCS.write_text("\n".join(protocol) + "\n", encoding="utf-8")
    update_readme(summary, args)
    print("\nFINAL OUTPUTS READY:", flush=True)
    for relative in ["results/extended/report.md", "results/extended/summary_metrics.csv",
                     "results/extended/benchmark_comparison.png",
                     "results/extended/learning_curves.png",
                     "results/extended/classwise_f1.png",
                     "results/extended/confusion_matrices.png",
                     "docs/extended_benchmark_protocol.md", "README.md"]:
        print("  " + relative, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true", help="Check training code without writing files")
    parser.add_argument("--force", action="store_true", help="Ignore completed run records and retrain")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(args.seeds) < 1:
        parser.error("epochs, batch-size and number of seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    torch.set_grad_enabled(True)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    print("Device: CPU | Extended experiment", flush=True)

    if args.smoke_test:
        for slug, display, factory in MODELS:
            training_run(slug, display, factory, 42, 1, args.batch_size,
                         test_loader=None, test_images=None, test_classes=None, smoke=True)
        print("SMOKE TEST PASSED: now run the full command.", flush=True)
        return

    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = OxfordIIITPet(
        root=ROOT / "data", split="test", target_types="category",
        transform=test_transform, download=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    fixed_positions = np.linspace(0, len(dataset) - 1, 5, dtype=int).tolist()
    test_images = [dataset[i][0].unsqueeze(0) for i in fixed_positions]
    test_classes = list(dataset.classes)
    records = []

    for slug, display, factory in MODELS:
        for seed in args.seeds:
            record_path = RECORDS / f"{slug}_seed{seed}.json"
            checkpoint_path = WEIGHTS / f"{slug}_seed{seed}_best.pth"
            if not args.force and record_path.exists() and checkpoint_path.exists():
                loaded = json.loads(record_path.read_text(encoding="utf-8"))
                if (loaded.get("model") == slug and loaded.get("seed") == seed and
                        loaded.get("epochs") == args.epochs and
                        loaded.get("batch_size") == args.batch_size and
                        len(loaded.get("per_class_f1", [])) == len(test_classes)):
                    print(f"RESUME: {display}, seed={seed} already complete.", flush=True)
                    records.append(loaded)
                    continue
            print(f"\nTRAINING: {display} | seed {seed} | {args.epochs} epochs", flush=True)
            record = training_run(slug, display, factory, seed, args.epochs,
                                  args.batch_size, test_loader, test_images, test_classes)
            records.append(record)

    if len(records) != len(MODELS) * len(args.seeds):
        raise RuntimeError("Missing completed runs; refusing to produce a partial summary.")
    write_outputs(records, test_classes, args)
    print("DONE: Extended benchmark complete. Initial single-epoch results were preserved.")


if __name__ == "__main__":
    main()
