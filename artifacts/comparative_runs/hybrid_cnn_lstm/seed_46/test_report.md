# Hybrid CNN-LSTM test report

## Status

This is an actual seed-46 run on the untouched fixed BUSI test set. Checkpoint epoch 11 was selected only by validation macro-F1.

Implementation scope: Four-scale CNN feature extractor followed by a two-layer bidirectional LSTM and learned attention pooling over the spatial feature sequence; three-class output.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, 0.0, 0.0]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8559 |
| Cohen's kappa | 0.7547 |
| Macro-F1 | 0.8696 |
| Macro sensitivity | 0.8712 |
| Macro specificity | 0.9138 |
| Macro AUROC (OvR) | 0.9529 |
| Macro AUPRC | 0.9297 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8769 | 0.8636 | 0.8462 | 0.8702 | 0.9336 | 0.9389 |
| malignant | 32 | 0.7273 | 0.7500 | 0.8953 | 0.7385 | 0.9251 | 0.8502 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[57, 9, 0]
[8, 24, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 664,484
- Training wall time (seconds): 301.1
- Test prediction file: `hybrid_cnn_lstm_seed46_test_predictions.csv`
- Machine-readable report: `hybrid_cnn_lstm_seed46_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
