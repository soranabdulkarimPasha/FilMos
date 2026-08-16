# Hybrid CNN-LSTM test report

## Status

This is an actual seed-44 run on the untouched fixed BUSI test set. Checkpoint epoch 11 was selected only by validation macro-F1.

Implementation scope: Four-scale CNN feature extractor followed by a two-layer bidirectional LSTM and learned attention pooling over the spatial feature sequence; three-class output.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, 1.0, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8898 |
| Cohen's kappa | 0.8093 |
| Macro-F1 | 0.8969 |
| Macro sensitivity | 0.8914 |
| Macro specificity | 0.9293 |
| Macro AUROC (OvR) | 0.9514 |
| Macro AUPRC | 0.9302 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8841 | 0.9242 | 0.8462 | 0.9037 | 0.9333 | 0.9319 |
| malignant | 32 | 0.8276 | 0.7500 | 0.9419 | 0.7869 | 0.9208 | 0.8585 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[61, 5, 0]
[8, 24, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 664,484
- Training wall time (seconds): 303.7
- Test prediction file: `hybrid_cnn_lstm_seed44_test_predictions.csv`
- Machine-readable report: `hybrid_cnn_lstm_seed44_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
