# CTMF-Net test report

## Status

This is an actual seed-44 run on the untouched fixed BUSI test set. Checkpoint epoch 23 was selected only by validation macro-F1.

Implementation scope: Three-class 128x128 adaptation of Wang et al.: four VGG-like CNN stages with CBAM, four three-block ViT stages, and two-pass cross-attention FIMs. The output head and input channels are adapted to the shared BUSI protocol.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, 0.4999999701976776, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8559 |
| Cohen's kappa | 0.7506 |
| Macro-F1 | 0.8651 |
| Macro sensitivity | 0.8605 |
| Macro specificity | 0.9088 |
| Macro AUROC (OvR) | 0.9387 |
| Macro AUPRC | 0.9117 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8551 | 0.8939 | 0.8077 | 0.8741 | 0.9172 | 0.9240 |
| malignant | 32 | 0.7586 | 0.6875 | 0.9186 | 0.7213 | 0.8990 | 0.8112 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[59, 7, 0]
[10, 22, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 2,612,143
- Training wall time (seconds): 1154.9
- Test prediction file: `ctmf_net_seed44_test_predictions.csv`
- Machine-readable report: `ctmf_net_seed44_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
