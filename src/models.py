import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


NUM_CLASSES = 37


def build_mobilenet(num_classes=NUM_CLASSES):
    # Load a model pre-trained on ImageNet
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Freeze the pre-trained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier for our 37 classes
    in_features = model.classifier[0].in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def main():
    model = build_mobilenet()

    # Set the model to evaluation mode
    model.eval()

    # Create a sample input image
    sample_input = torch.randn(1, 3, 224, 224)

    # Test the model without training
    with torch.inference_mode():
        output = model(sample_input)

    print("Model: MobileNetV3-Small")
    print("Number of classes:", NUM_CLASSES)
    print("Input shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(output.shape))

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", trainable_params)
    print("Model test completed successfully!")


if __name__ == "__main__":
    main()

    