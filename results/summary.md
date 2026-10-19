# Initial Image Classification Benchmark

**Preliminary comparison only.** The experiment uses a limited training budget; results should not be interpreted as conclusive model rankings.

| Model | Best validation accuracy | Test accuracy | CPU latency | Checkpoint size |
|---|---:|---:|---:|---:|
| mobilenet_v3_small | 79.35% | 76.75% | 12.81 ms/image | 3.75 MiB |
| resnet18 | 85.60% | 83.86% | 65.21 ms/image | 42.78 MiB |

![Model comparison](model_comparison.png)

## Experimental setup

- Dataset: Oxford-IIIT Pet (37 classes).
- Train/validation: 2,944 / 736 images (saved stratified split).
- Test: the separate official 3,669-image test split, assessed after model selection.
- Pre-trained ImageNet weights, frozen backbones, newly trained linear heads.
- Epochs: 1; batch size: 16; Adam learning rate: 0.001.
- Model selection: highest validation accuracy across epochs.
- Latency: median of 30 single-image CPU forwards after 5 warmups, excluding preprocessing and disk I/O.
- Checkpoint size: serialized full PyTorch model state_dict, measured in MiB.
- Python: 3.13.14; PyTorch: 2.14.1+cpu; Torchvision: 0.29.1+cpu.
- CPU: Intel64 Family 6 Model 142 Stepping 9, GenuineIntel; OS: Windows 11; PyTorch CPU threads: 2.

## Limitations

The models were only trained with frozen feature extractors. The number of epochs is limited, and CPU timings are hardware-dependent. No mobile-device benchmark or model optimization was performed. Replicate across multiple seeds and training budgets for stronger conclusions.
