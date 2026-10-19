r"""Train and compare MobileNetV3-Small and ResNet18 on Oxford-IIIT Pet.

Run from the repository root:
    .\.venv\Scripts\python.exe src\train.py --epochs 1

This is a lightweight, CPU-oriented transfer-learning benchmark.
"""

import argparse
import csv
import platform
import random
import sys
import time
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from dataloaders import MEAN, STD, get_data_loaders
from models import build_mobilenet


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
SEED = 42


def build_resnet18(num_classes=37):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_model(name, make_model, train_loader, val_loader, epochs):
    # Reinitialize random state to make repeated runs more reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    checkpoint = CHECKPOINT_DIR / f"{name}_best.pth"
    best_validation_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        # Frozen pretrained backbones must not update BatchNorm running statistics.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        total_loss = 0.0
        seen = 0
        for images, labels in tqdm(
            train_loader, desc=f"{name}: epoch {epoch}/{epochs}", leave=True
        ):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n
            seen += n

        validation_accuracy = accuracy(model, val_loader)
        print(
            f"{name} | epoch {epoch}/{epochs} | "
            f"train loss: {total_loss / seen:.4f} | "
            f"validation accuracy: {validation_accuracy:.2f}%",
            flush=True,
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model, checkpoint, best_validation_accuracy


def measure_cpu_latency_ms(model, sample, warmups=5, repetitions=30):
    # Input preprocessing/disk reading are deliberately excluded.
    model.eval()
    with torch.inference_mode():
        for _ in range(warmups):
            model(sample)
        measurements = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(sample)
            measurements.append((time.perf_counter() - start) * 1000.0)
    return median(measurements)


def save_comparison_chart(rows):
    labels = [row["model"] for row in rows]
    metrics = [
        ("test_accuracy_percent", "Test accuracy (%)"),
        ("cpu_latency_ms", "CPU latency (ms/image)"),
        ("checkpoint_size_mib", "Checkpoint size (MiB)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (key, title) in zip(axes, metrics):
        axis.bar(labels, [float(row[key]) for row in rows])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Oxford-IIIT Pet: Transfer Learning Comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Device: CPU (no GPU required)", flush=True)
    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    test_dataset = OxfordIIITPet(
        root=ROOT / "data",
        split="test",
        target_types="category",
        transform=test_transform,
        download=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    single_image = test_dataset[0][0].unsqueeze(0)
    rows = []

    for name, factory in [
        ("mobilenet_v3_small", build_mobilenet),
        ("resnet18", build_resnet18),
    ]:
        print(f"\n=== {name} ===", flush=True)
        model, checkpoint, val_acc = train_model(
            name, factory, train_loader, val_loader, args.epochs
        )
        test_acc = accuracy(model, test_loader)
        latency_ms = measure_cpu_latency_ms(model, single_image)
        size_mib = checkpoint.stat().st_size / (1024 * 1024)
        row = {
            "model": name,
            "epochs": args.epochs,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_dataset),
            "best_validation_accuracy_percent": round(val_acc, 2),
            "test_accuracy_percent": round(test_acc, 2),
            "cpu_latency_ms": round(latency_ms, 2),
            "checkpoint_size_mib": round(size_mib, 2),
        }
        rows.append(row)
        print(f"RESULT: {row}", flush=True)
        del model

    csv_path = RESULTS_DIR / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_comparison_chart(rows)

    report_lines = [
        "# Initial Image Classification Benchmark",
        "",
        "**Preliminary comparison only.** The experiment uses a limited training "
        "budget; results should not be interpreted as conclusive model rankings.",
        "",
        "| Model | Best validation accuracy | Test accuracy | CPU latency | Checkpoint size |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['model']} | {row['best_validation_accuracy_percent']:.2f}% | "
            f"{row['test_accuracy_percent']:.2f}% | "
            f"{row['cpu_latency_ms']:.2f} ms/image | "
            f"{row['checkpoint_size_mib']:.2f} MiB |"
        )
    report_lines.extend([
        "",
        "![Model comparison](model_comparison.png)",
        "",
        "## Experimental setup",
        "",
        "- Dataset: Oxford-IIIT Pet (37 classes).",
        "- Train/validation: 2,944 / 736 images (saved stratified split).",
        "- Test: the separate official 3,669-image test split, assessed after model selection.",
        "- Pre-trained ImageNet weights, frozen backbones, newly trained linear heads.",
        f"- Epochs: {args.epochs}; batch size: {args.batch_size}; Adam learning rate: 0.001.",
        "- Model selection: highest validation accuracy across epochs.",
        "- Latency: median of 30 single-image CPU forwards after 5 warmups, "
        "excluding preprocessing and disk I/O.",
        "- Checkpoint size: serialized full PyTorch model state_dict, measured in MiB.",
        f"- Python: {sys.version.split()[0]}; PyTorch: {torch.__version__}; "
        f"Torchvision: {torchvision.__version__}.",
        f"- CPU: {platform.processor() or 'not reported'}; "
        f"OS: {platform.system()} {platform.release()}; "
        f"PyTorch CPU threads: {torch.get_num_threads()}.",
        "",
        "## Limitations",
        "",
        "The models were only trained with frozen feature extractors. "
        "The number of epochs is limited, and CPU timings are hardware-dependent. "
        "No mobile-device benchmark or model optimization was performed. "
        "Replicate across multiple seeds and training budgets for stronger conclusions.",
        "",
    ])
    (RESULTS_DIR / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nDONE: {csv_path}")
    print(f"DONE: {RESULTS_DIR / 'model_comparison.png'}")
    print(f"DONE: {RESULTS_DIR / 'summary.md'}")
    print("Note: 1 epoch gives preliminary results, not a conclusive study.")


if __name__ == "__main__":
    main()
