# CTMF-Net test report

## Status

This is an actual seed-43 run on the untouched fixed BUSI test set. Checkpoint epoch 17 was selected only by validation macro-F1.

Implementation scope: Three-class 128x128 adaptation of Wang et al.: four VGG-like CNN stages with CBAM, four three-block ViT stages, and two-pass cross-attention FIMs. The output head and input channels are adapted to the shared BUSI protocol.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, -0.4999999701976776, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8898 |
| Cohen's kappa | 0.8060 |
| Macro-F1 | 0.8928 |
| Macro sensitivity | 0.8807 |
| Macro specificity | 0.9243 |
| Macro AUROC (OvR) | 0.9519 |
| Macro AUPRC | 0.9176 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8630 | 0.9545 | 0.8077 | 0.9065 | 0.9374 | 0.9504 |
| malignant | 32 | 0.8800 | 0.6875 | 0.9651 | 0.7719 | 0.9182 | 0.8023 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[63, 3, 0]
[10, 22, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 2,612,143
- Training wall time (seconds): 1085.2
- Test prediction file: `ctmf_net_seed43_test_predictions.csv`
- Machine-readable report: `ctmf_net_seed43_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
