# Reproducibility Checklist

This checklist helps readers rerun the current **preliminary** image-classification benchmark locally. It does not claim bit-for-bit reproducibility across different hardware or library versions.

## Before running

- [ ] Clone the repository and work from its root directory.
- [ ] Use a compatible 64-bit Python environment (the recorded Windows run used Python 3.13).
- [ ] Create a virtual environment and install `requirements.txt`.
- [ ] Make sure there is enough free space to download the Oxford-IIIT Pet images and pretrained model weights.
- [ ] Keep `data/`, `.venv/`, and `checkpoints/` out of version control.

## Commands (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\data.py
.\.venv\Scripts\python.exe src\dataloaders.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To rerun the **original single-epoch benchmark** (both models):

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 1
```

**Caution:** The benchmark command trains the models again and may overwrite some files inside `results/`. If the purpose is only to inspect existing results, open `results/comparison.csv`, `results/summary.md`, and the plots instead of rerunning the experiment.

## What to check in the output

- [ ] The training and validation subsets use 2,944 and 736 images respectively.
- [ ] The official test set contains 3,669 images.
- [ ] The models predict one of 37 classes.
- [ ] Reported scores were computed using the saved experiment code and stated preprocessing steps.
- [ ] CPU timing is interpreted as **model-only latency**, not mobile-device or end-to-end latency.
- [ ] The README clearly labels the one-epoch comparison as preliminary.

## Notes for readers

The original observed values are documented in the [README](../README.md) and [benchmark results](../results/comparison.csv). Re-running may produce different accuracy or timing values. Do not replace an earlier result without recording the new settings and hardware context.
