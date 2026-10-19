from pathlib import Path

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms


# Define the dataset directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    # Prepare images for classification models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load training and validation data
    trainval_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=transform,
        download=True,
    )

    # Load the official test dataset
    test_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="category",
        transform=transform,
        download=True,
    )

    # Display dataset information
    print("Train/Validation images:", len(trainval_dataset))
    print("Test images:", len(test_dataset))
    print("Number of classes:", len(trainval_dataset.classes))

    # Inspect one sample
    image, label = trainval_dataset[0]
    print("Sample image shape:", tuple(image.shape))
    print("Sample class:", trainval_dataset.classes[label])


if __name__ == "__main__":
    main()

    