Efficient Image Classification: Accuracy vs. Inference Speed

Overview

This project investigates the trade-off between classification accuracy and computational efficiency using PyTorch.

The goal is to compare two convolutional neural network architectures, MobileNetV3-Small and ResNet18, on an image classification task, focusing on prediction accuracy, CPU inference speed, and model size.

Research Question

How does a lightweight image classification model compare with a larger model in terms of accuracy, inference speed, and model size?

Dataset

The project will use the "Oxford-IIIT Pet Dataset" (https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains images of 37 cat and dog breeds.

Models

- MobileNetV3-Small: A lightweight convolutional neural network designed for computational efficiency.
- ResNet18: A widely used convolutional neural network that will serve as a comparison model.

Pre-trained weights and transfer learning will be used to adapt both models to the classification task.

Evaluation Metrics

The models will be evaluated using:

- Classification Accuracy: Performance on the test dataset.
- CPU Inference Time: Time required to process input images on the same CPU.
- Model Size: Storage size of the trained model weights.

Tools and Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

Project Status

In Progress

The project is currently under development. Dataset preparation, model training, performance benchmarking, and result analysis will be added incrementally.
