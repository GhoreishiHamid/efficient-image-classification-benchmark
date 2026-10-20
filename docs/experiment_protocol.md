# Experiment Protocol and Reproducibility Notes

## Goal

Compare classification accuracy, CPU inference latency, and checkpoint size for a
lightweight CNN (MobileNetV3-Small) and a larger CNN (ResNet18). This is a
**preliminary, single-epoch study**, not a conclusive architecture ranking.

## Data and splits

- Dataset: Oxford-IIIT Pet, 37 cat and dog breeds.
- Official `trainval` partition: 3,680 images, split into 2,944 for training
  and 736 for validation using stratified sampling (`random_state=42`).
- Split indices are saved under [`splits/`](../splits/) so both models use
  the same examples.
- Separate official test partition: 3,669 images.
- Training transformations: random resized crop to 224 × 224, horizontal
  flip, tensor conversion, and ImageNet normalization.
- Validation/test transformations: resize to 256, center crop to 224 × 224,
  tensor conversion, and ImageNet normalization.

## Model training and evaluation

- Initialize both networks using ImageNet-pretrained weights.
- Freeze pretrained feature extractors and train a new 37-class linear head.
- Adam optimizer, learning rate 0.001, cross-entropy loss, batch size 16.
- The recorded benchmark trained each model for **one epoch** on CPU.
- Model selection is based on validation accuracy, not test accuracy.
- The saved test-set results were later used for **descriptive class-wise
  error analysis**; they should not be used for further tuning choices.

## CPU benchmark protocol

- Single preprocessed image, model forward pass only.
- Five warm-up passes, followed by 30 timed forward passes.
- Reported latency: median per-image milliseconds; preprocessing and disk
  I/O are excluded.
- Model size: on-disk size of the saved full PyTorch `state_dict` in MiB.
- CPU timing is hardware-dependent and does **not** represent measured
  mobile-device performance.

## Quality checks

From the repository root, after installing dependencies and downloading the dataset:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests verify saved split integrity, basic consistency of benchmark
results, and DataLoader output shapes. They do not verify model accuracy
or replace repeated experiments.

## Limitations and meaningful extensions

One epoch and one fixed seed are insufficient for strong performance
conclusions. For a more rigorous study, run several seeds, compare longer
training budgets using validation-based selection, report uncertainty,
and benchmark on additional hardware. These extensions have **not** been
performed in the current results.
