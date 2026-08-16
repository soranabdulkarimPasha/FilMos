# Hybrid CNN-LSTM test report

## Status

This is an actual seed-42 run on the untouched fixed BUSI test set. Checkpoint epoch 10 was selected only by validation macro-F1.

Implementation scope: Four-scale CNN feature extractor followed by a two-layer bidirectional LSTM and learned attention pooling over the spatial feature sequence; three-class output.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, 0.09999997168779373, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8814 |
| Cohen's kappa | 0.7988 |
| Macro-F1 | 0.8934 |
| Macro sensitivity | 0.8971 |
| Macro specificity | 0.9305 |
| Macro AUROC (OvR) | 0.9483 |
| Macro AUPRC | 0.9150 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.9062 | 0.8788 | 0.8846 | 0.8923 | 0.9242 | 0.9262 |
| malignant | 32 | 0.7647 | 0.8125 | 0.9070 | 0.7879 | 0.9208 | 0.8188 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[58, 8, 0]
[6, 26, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 664,484
- Training wall time (seconds): 277.3
- Test prediction file: `hybrid_cnn_lstm_test_predictions.csv`
- Machine-readable report: `hybrid_cnn_lstm_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
