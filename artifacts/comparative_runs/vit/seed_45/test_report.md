# ViT test report

## Status

This is an actual seed-45 run on the untouched fixed BUSI test set. Checkpoint epoch 11 was selected only by validation macro-F1.

Implementation scope: Standalone ViT-B/16-style baseline with 16x16 patches, six encoder blocks, 128-dimensional tokens and four attention heads. This implements the requested ViT model, not a multi-checkpoint ensemble.

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
| Accuracy | 0.7966 |
| Cohen's kappa | 0.6371 |
| Macro-F1 | 0.7955 |
| Macro sensitivity | 0.7876 |
| Macro specificity | 0.8639 |
| Macro AUROC (OvR) | 0.9165 |
| Macro AUPRC | 0.8494 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.7763 | 0.8939 | 0.6731 | 0.8310 | 0.8887 | 0.9067 |
| malignant | 32 | 0.6818 | 0.4688 | 0.9186 | 0.5556 | 0.8608 | 0.6414 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[59, 7, 0]
[17, 15, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 1,297,155
- Training wall time (seconds): 403.9
- Test prediction file: `vit_seed45_test_predictions.csv`
- Machine-readable report: `vit_seed45_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
