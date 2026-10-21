# Interpreting the Accuracy–Efficiency Trade-off

## Purpose

This note explains what the initial Oxford-IIIT Pet benchmark does—and does not—show. The comparison uses ImageNet-pretrained MobileNetV3-Small and ResNet18 with frozen feature extractors and a newly trained 37-class classification head.

## Observed results

| Measurement | MobileNetV3-Small | ResNet18 |
| --- | ---: | ---: |
| Test accuracy | 76.75% | 83.86% |
| CPU inference latency (ms/image) | 12.81 | 65.21 |
| Saved checkpoint size (MiB) | 3.75 | 42.78 |
| Training epochs | 1 | 1 |

The values come from the **single-epoch, single-run** experiment saved in `results/comparison.csv`. They are descriptive measurements, not a general ranking of the two architectures.

## Key observations

1. **Accuracy:** ResNet18 achieved 7.11 percentage points higher test accuracy in this run.
2. **Latency:** MobileNetV3-Small had approximately 5.1 times lower measured CPU latency per image (`65.21 / 12.81`).
3. **Storage:** The saved MobileNetV3-Small checkpoint was approximately 11.4 times smaller (`42.78 / 3.75`).

These results suggest different practical priorities: ResNet18 performed better on test accuracy, while MobileNetV3-Small was more attractive when CPU response time and checkpoint storage were emphasized.

## Why the comparison is limited

- Both models were trained for **only one epoch** with a frozen backbone and a new classification head.
- The measurements use one CPU environment; hardware and software differences can change latency.
- Inference timing excludes image loading and preprocessing, so it is **not** end-to-end application latency.
- The experiment uses a single split and random seed. Repeated-seed experiments would be required for stronger statistical conclusions.
- The official test partition should not be used to choose training settings or optimize future model variants.

## Takeaway

A fair interpretation is that the initial experiment **illustrates an accuracy–efficiency trade-off under the recorded settings**. It does not establish that either architecture is always superior.

For the experimental setup and test data, see [Experiment Protocol](experiment_protocol.md), [CSV results](../results/comparison.csv), and the [main README](../README.md).
