import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = PROJECT_ROOT / "splits"


def main():
    # Load the training and validation dataset
    dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        download=False,
    )

    print("Reading dataset labels...")

    # Collect class labels
    labels = [dataset[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    # Create a reproducible stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # Sort indices before saving
    train_indices.sort()
    val_indices.sort()

    # Verify there is no overlap
    assert set(train_indices).isdisjoint(val_indices)

    # Save the split indices
    SPLITS_DIR.mkdir(exist_ok=True)

    with open(SPLITS_DIR / "train_indices.json", "w") as f:
        json.dump(train_indices, f)

    with open(SPLITS_DIR / "val_indices.json", "w") as f:
        json.dump(val_indices, f)

    # Display split information
    print("Total images:", len(dataset))
    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Number of classes:", len(dataset.classes))
    print("Split indices saved successfully!")


if __name__ == "__main__":
    main()

    