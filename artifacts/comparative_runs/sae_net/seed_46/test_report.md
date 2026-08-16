# SAE-Net (image-spectrum adaptation) test report

## Status

This is an actual seed-46 run on the untouched fixed BUSI test set. Checkpoint epoch 23 was selected only by validation macro-F1.

Implementation scope: Non-equivalent image-only adaptation of Xie et al. The grayscale DenseNet-like branch and 128-bin cumulative tissue-probability pooling are retained, but radial B-mode FFT statistics replace the unavailable raw-RF bump-wavelet spectra. Results must not be described as an exact SAE-Net reproduction.

## Common protocol

- Independent images: 780 (437 benign, 210 malignant, 133 normal)
- Fixed split: 544 train / 118 validation / 118 test, seed 42
- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128
- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale
- AdamW; maximum epochs 24; batch size 32; validation macro-F1 selection
- TTA enabled: True; class-logit bias selected on validation only: [0.0, -1.7999999523162842, -2.9802322387695312e-08]

## Test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9322 |
| Cohen's kappa | 0.8831 |
| Macro-F1 | 0.9371 |
| Macro sensitivity | 0.9328 |
| Macro specificity | 0.9563 |
| Macro AUROC (OvR) | 0.9758 |
| Macro AUPRC | 0.9573 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.9265 | 0.9545 | 0.9038 | 0.9403 | 0.9662 | 0.9735 |
| malignant | 32 | 0.9000 | 0.8438 | 0.9651 | 0.8710 | 0.9611 | 0.8983 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[63, 3, 0]
[5, 27, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 317,302
- Training wall time (seconds): 378.7
- Test prediction file: `sae_net_seed46_test_predictions.csv`
- Machine-readable report: `sae_net_seed46_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
