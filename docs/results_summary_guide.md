# Quick Results Summary

The repository includes a lightweight command-line tool to review the **existing**
benchmark results without loading the image dataset, downloading pretrained weights,
or retraining either model.

## Run

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe src\results_summary.py
```

It reads `results/comparison.csv` and prints the stored test accuracy, CPU
inference latency, checkpoint size, and relative differences between the two
architectures. You can also specify another CSV with `--csv PATH`.

## Validate the utility

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_results_summary.py" -v
```

The tests use small synthetic CSV data and do not train models. They verify
metric parsing, ratio calculations, missing-column handling, and invalid timing
values.

## Interpretation

- The accuracy difference is measured in **percentage points**.
- CPU latency ratios refer only to the recorded hardware and timing protocol;
  they are not a general claim about mobile-device performance.
- Checkpoint size refers to the saved model `state_dict`, not runtime memory.
- The original comparison was based on **one training epoch and one run**.
  This command summarizes those recorded values; it does not add experimental
  evidence or change their scientific limitations.
