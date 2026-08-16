# DSDNet (same-mask adaptation) test report

## Status

This is an actual seed-45 run on the untouched fixed BUSI test set. Checkpoint epoch 14 was selected only by validation macro-F1.

Implementation scope: Three-class adaptation preserving four DSMA, SPGF and MILT stages. For information parity with mask-assisted FiLMoS-Net, the supplied BUSI union mask is the frozen spatial prior; this replaces the paper's externally pretrained BUSBRA segmentation network. The four guided scales begin at 16x16 for the shared 128x128 input. This must be reported as a same-protocol adaptation.

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
| Cohen's kappa | 0.7441 |
| Macro-F1 | 0.8568 |
| Macro sensitivity | 0.8444 |
| Macro specificity | 0.9012 |
| Macro AUROC (OvR) | 0.9598 |
| Macro AUPRC | 0.9333 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.8267 | 0.9394 | 0.7500 | 0.8794 | 0.9464 | 0.9579 |
| malignant | 32 | 0.8261 | 0.5938 | 0.9535 | 0.6909 | 0.9331 | 0.8420 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[62, 4, 0]
[13, 19, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 2,521,205
- Training wall time (seconds): 578.3
- Test prediction file: `dsdnet_seed45_test_predictions.csv`
- Machine-readable report: `dsdnet_seed45_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
