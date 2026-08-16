# ViT test report

## Status

This is an actual seed-43 run on the untouched fixed BUSI test set. Checkpoint epoch 21 was selected only by validation macro-F1.

Implementation scope: Standalone ViT-B/16-style baseline with 16x16 patches, six encoder blocks, 128-dimensional tokens and four attention heads. This implements the requested ViT model, not a multi-checkpoint ensemble.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, -0.699999988079071, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8475 |
| Cohen's kappa | 0.7325 |
| Macro-F1 | 0.8531 |
| Macro sensitivity | 0.8447 |
| Macro specificity | 0.8998 |
| Macro AUROC (OvR) | 0.9259 |
| Macro AUPRC | 0.8761 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8333 | 0.9091 | 0.7692 | 0.8696 | 0.9006 | 0.9130 |
| malignant | 32 | 0.7692 | 0.6250 | 0.9302 | 0.6897 | 0.8772 | 0.7153 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[60, 6, 0]
[12, 20, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 1,297,155
- Training wall time (seconds): 577.9
- Test prediction file: `vit_seed43_test_predictions.csv`
- Machine-readable report: `vit_seed43_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
