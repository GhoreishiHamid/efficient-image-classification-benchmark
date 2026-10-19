import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"

# ImageNet normalization values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_loaders(batch_size=16):
    # Transformations for training images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # Transformations for validation images
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # Load the same dataset with different transforms
    train_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=False,
    )

    val_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=val_transform,
        download=False,
    )

    # Load saved split indices
    with open(SPLITS_DIR / "train_indices.json") as f:
        train_indices = json.load(f)

    with open(SPLITS_DIR / "val_indices.json") as f:
        val_indices = json.load(f)

    # Verify that the splits do not overlap
    assert set(train_indices).isdisjoint(val_indices)

    # Create training and validation subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def main():
    train_loader, val_loader = get_data_loaders()

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Inspect one batch
    images, labels = next(iter(train_loader))

    print("Image batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(labels.shape))
    print("DataLoader test completed successfully!")


if __name__ == "__main__":
    main()

    