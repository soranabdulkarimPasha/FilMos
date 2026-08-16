# SAE-Net (image-spectrum adaptation) test report

## Status

This is an actual seed-45 run on the untouched fixed BUSI test set. Checkpoint epoch 6 was selected only by validation macro-F1.

Implementation scope: Non-equivalent image-only adaptation of Xie et al. The grayscale DenseNet-like branch and 128-bin cumulative tissue-probability pooling are retained, but radial B-mode FFT statistics replace the unavailable raw-RF bump-wavelet spectra. Results must not be described as an exact SAE-Net reproduction.

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
| Accuracy | 0.8898 |
| Cohen's kappa | 0.8124 |
| Macro-F1 | 0.9003 |
| Macro sensitivity | 0.9021 |
| Macro specificity | 0.9344 |
| Macro AUROC (OvR) | 0.9536 |
| Macro AUPRC | 0.9312 |

## Class-wise test metrics

| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| benign | 66 | 0.9077 | 0.8939 | 0.8846 | 0.9008 | 0.9336 | 0.9259 |
| malignant | 32 | 0.7879 | 0.8125 | 0.9186 | 0.8000 | 0.9273 | 0.8676 |
| normal | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion matrix

Rows are true classes and columns are predicted classes in benign, malignant, normal order.

```text
[59, 7, 0]
[6, 26, 0]
[0, 0, 20]
```

## Audit information

- Trainable parameters: 317,302
- Training wall time (seconds): 191.6
- Test prediction file: `sae_net_seed45_test_predictions.csv`
- Machine-readable report: `sae_net_seed45_test_report.json`

No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.
