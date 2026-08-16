# ViT test report

## Status

This is an actual seed-44 run on the untouched fixed BUSI test set. Checkpoint epoch 7 was selected only by validation macro-F1.

Implementation scope: Standalone ViT-B/16-style baseline with 16x16 patches, six encoder blocks, 128-dimensional tokens and four attention heads. This implements the requested ViT model, not a multi-checkpoint ensemble.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, 0.3999999761581421, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8390 |
| Cohen's kappa | 0.7188 |
| Macro-F1 | 0.8464 |
| Macro sensitivity | 0.8396 |
| Macro specificity | 0.8959 |
| Macro AUROC (OvR) | 0.9136 |
| Macro AUPRC | 0.8516 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8310 | 0.8939 | 0.7692 | 0.8613 | 0.8843 | 0.8954 |
| malignant | 32 | 0.7407 | 0.6250 | 0.9186 | 0.6780 | 0.8565 | 0.6594 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[59, 7, 0]
[12, 20, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 1,297,155
- Training wall time (seconds): 313.6
- Test prediction file: `vit_seed44_test_predictions.csv`
- Machine-readable report: `vit_seed44_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
