# ViT test report

## Status

This is an actual seed-46 run on the untouched fixed BUSI test set. Checkpoint epoch 19 was selected only by validation macro-F1.

Implementation scope: Standalone ViT-B/16-style baseline with 16x16 patches, six encoder blocks, 128-dimensional tokens and four attention heads. This implements the requested ViT model, not a multi-checkpoint ensemble.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, -0.19999997317790985, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8220 |
| Cohen's kappa | 0.6816 |
| Macro-F1 | 0.8147 |
| Macro sensitivity | 0.8081 |
| Macro specificity | 0.8785 |
| Macro AUROC (OvR) | 0.9190 |
| Macro AUPRC | 0.8641 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.7922 | 0.9242 | 0.6923 | 0.8531 | 0.8846 | 0.9004 |
| malignant | 32 | 0.8000 | 0.5000 | 0.9535 | 0.6154 | 0.8725 | 0.6920 |
| normal | 20 | 0.9524 | 1.0000 | 0.9898 | 0.9756 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[61, 4, 1]
[16, 16, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 1,297,155
- Training wall time (seconds): 571.5
- Test prediction file: `vit_seed46_test_predictions.csv`
- Machine-readable report: `vit_seed46_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
