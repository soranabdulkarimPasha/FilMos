# BUSI dataset preparation

The BUSI images are not redistributed in this repository. Obtain the dataset described by Al-Dhabyani et al. (Data in Brief, 2020; DOI: `10.1016/j.dib.2019.104863`) and place it at:

```text
data/Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

The public BUSI distribution contains diagnostic B-mode images and corresponding lesion-annotation images. FiLMoS-Net uses 780 independent diagnostic images: 437 benign, 210 malignant, and 133 normal. The annotation files are auxiliary inputs and are not counted as independent samples.

The corrected experiment constructs three aligned channels in this order:

1. original grayscale context;
2. binary lesion geometry (the union is used when more than one annotation exists);
3. lesion-only grayscale texture.

All three channels are resized together to 128 × 128 pixels using bilinear interpolation and normalized with ImageNet mean and standard deviation. Training-only augmentation applies horizontal flipping (`p=0.5`), rotation (±5°), translation (up to 2%), and scaling (`0.98–1.02`). Validation and test inputs are only resized, converted to tensors, and normalized.

Run the audit after placing BUSI:

```bash
python prepare_busi_data.py
```

The command verifies all 780 paths in `data/busi_split_seed42.csv`, the 544/118/118 split, class counts, and availability of a lesion annotation for every diagnostic image.
