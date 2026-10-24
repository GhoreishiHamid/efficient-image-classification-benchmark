# Extended Benchmark Protocol

This is a follow-up to the original single-epoch Oxford-IIIT Pet experiment.

## Design

- Models: pretrained MobileNetV3-Small and ResNet18 (frozen backbones).
- Fixed stratified training/validation indices: `splits/train_indices.json` and `splits/val_indices.json`.
- Official test split held out for evaluation after selecting each run's best epoch by validation accuracy.
- Exactly 3 epochs each, with seeds 42, 123, 2026, batch size 16, Adam LR=0.001, cross-entropy loss.
- CPU only. Validation/test use deterministic transforms; training uses random crop and flip.
- BatchNorm running statistics remain frozen; only learnable parameters in new linear heads are optimized.
- Latency uses 5 fixed preprocessed test images, 5 warmups and 20 timed single-image forwards per image, summarized by median; excludes loading/preprocessing.
- Results present mean ± sample SD across runs, not inferential confidence intervals.

## Outputs

- `results/extended/summary_metrics.csv`: per-model descriptive aggregates.
- `results/extended/per_run_metrics.csv`: per-model and per-seed results.
- `results/extended/training_history.csv`: training loss and validation accuracy for each epoch.
- `results/extended/per_class_f1.csv`: mean/SD across runs per breed.
- `results/extended/run_records/`: self-contained per-run JSON metrics, including confusion matrices.
- `results/extended/*.png`: comparisons, learning curves, per-class F1, and confusion matrices.
- `checkpoints/extended/`: local model checkpoints; excluded from Git.

## Caveats

This benchmark does not demonstrate mobile deployment or conclusively rank architectures. The official test images were already evaluated in the earlier pilot, and appear in each seed's evaluation; seed-level results are not independent datasets. The historical one-epoch scores remain in `results/` and use a different, shorter timing procedure.

## Repeat

```powershell
.\.venv\Scripts\python.exe src\final_benchmark.py --epochs 3 --seeds 42 123 2026
```

If an interrupted run is resumed, completed combinations with matching settings and present local checkpoints are reused. `--force` retrains all combinations.
