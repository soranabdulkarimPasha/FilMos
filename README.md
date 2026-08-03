# FiLMoS-Net: reproducibility package for BUSI classification

This repository accompanies the revised manuscript describing **FiLMoS-Net**, a three-branch network for benign/malignant/normal breast-ultrasound classification. It packages the corrected seed-42 BUSI experiment requested in Reviewer 1, Comment 12 and in the editor's implementation-code request.

The corrected protocol uses 780 independent B-mode images (437 benign, 210 malignant, 133 normal), a fixed stratified 544/118/118 train/validation/test split, and lesion-assisted three-channel input. BUSI itself is not redistributed.

## Repository contents

```text
.
├── checkpoints/
│   ├── filmos_net_primary_seed42.pt
│   ├── filmos_net_cold_start_seed42.pt
│   └── filmos_net_training_partition_initialization_seed42.pt
├── config/
│   ├── random_seeds.json
│   └── training_config.json
├── data/
│   ├── busi_split_seed42.csv
│   └── repeated_run_scores.csv
├── notebooks/
│   └── FiLMoS_Net_complete_experiment.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   ├── final_metrics.json
│   ├── test_predictions.csv
│   └── confusion/ROC/PR numerical coordinates
├── filmos_net_architecture.py
├── prepare_busi_data.py
├── train_filmos_net.py
├── evaluate_filmos_net.py
├── export_final_results.py
├── export_figure_data.py
├── generate_manuscript_figures.py
├── analyze_repeated_runs.py
└── audit_package.py
```

`filmos_net_architecture.py` exposes the complete learnable Gabor, fixed DCT-II, differentiable morphology, adaptive routing, fusion, and classifier definitions. The complete research notebook is retained for provenance, while the command-line scripts are the recommended interface.

## Environment

The recorded environment was Python 3.10, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and an NVIDIA GeForce 930MX. CUDA is optional for evaluation; training is substantially faster on a CUDA-capable GPU.

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or using Conda:

```bash
conda env create -f environment.yml
conda activate filmos-net
```

## Dataset setup and exact split

Download BUSI from its original source (Al-Dhabyani et al., *Data in Brief*, 2020; DOI: `10.1016/j.dib.2019.104863`) and place its three class folders under `data/Dataset_BUSI_with_GT/`. See [DATASET.md](DATASET.md) for the expected layout and preprocessing order.

If BUSI is stored elsewhere, set:

```bash
export FILMOS_BUSI_ROOT=/absolute/path/to/Dataset_BUSI_with_GT
```

Then verify the dataset and exact split:

```bash
python prepare_busi_data.py
```

The authoritative partition is `data/busi_split_seed42.csv`. Test labels are not used for training, checkpoint selection, calibration, or hyperparameter selection.

## Evaluate the released checkpoint

```bash
python evaluate_filmos_net.py \
  checkpoints/filmos_net_primary_seed42.pt \
  --output results/checkpoint_evaluation.json
```

The primary epoch-13 checkpoint uses test-time averaging over the original input, horizontal and vertical flips, and normalized-intensity multipliers 1.03 and 0.97. An additive class-logit bias is selected once on validation macro-F1 and then frozen for test evaluation.

To recompute the final per-sample probabilities, metrics, Cohen's κ, class-wise results, and 5,000-resample stratified-bootstrap intervals:

```bash
python export_final_results.py
```

The expected fixed-test results are accuracy 0.9322, macro-F1 0.9371, Cohen's κ 0.8831, macro sensitivity 0.9328, macro specificity 0.9563, macro-AUROC 0.9764, and macro-AUPRC 0.9698. The expected confusion matrix is `[[63, 3, 0], [5, 27, 0], [0, 0, 20]]` in benign/malignant/normal order.

## Train FiLMoS-Net

Primary corrected run (fine-tuned from a FiLMoS-Net checkpoint trained only on the same 544-image training partition):

```bash
python train_filmos_net.py \
  --run-name filmos_net_primary_seed42 \
  --input-mode lesion_geometry \
  --warm-start checkpoints/filmos_net_training_partition_initialization_seed42.pt \
  --img-size 128 --batch-size 32 \
  --epochs 24 --patience 6 --lr 2e-4 \
  --class-weight-power 0.65 --seed 42
```

Independent cold-start verification:

```bash
python train_filmos_net.py \
  --run-name filmos_net_cold_start_seed42 \
  --input-mode lesion_geometry \
  --img-size 128 --batch-size 32 \
  --epochs 30 --patience 8 --lr 7e-4 \
  --class-weight-power 0.65 --seed 42
```

The cold-start validation-selected epoch-6 checkpoint reached test accuracy 0.9237 and macro-F1 0.9329. Exact hyperparameters and disabled imbalance/augmentation mechanisms are recorded in `config/training_config.json`.

## Regenerate manuscript artifacts

Tables 1–6:

```bash
python export_manuscript_tables.py
```

Numerical confusion-matrix and ROC/PR coordinates:

```bash
python export_figure_data.py
```

Figs. 8 and 9 in PNG and PDF:

```bash
python generate_manuscript_figures.py
```

Paired five-run SDs, 95% t intervals, Friedman tests, Kendall's W, and Nemenyi comparisons:

```bash
python analyze_repeated_runs.py
```

## Required author action before making the repository public

The supplied project folder did **not** contain the 30 raw paired observations or the five-run seed schedule claimed by the revised manuscript. Summary means, SDs, confidence intervals, and ranks cannot uniquely reconstruct those observations. To protect research integrity, this package does not invent them: `data/repeated_run_scores.csv` currently contains its schema only, and `config/random_seeds.json` marks the missing schedule.

Before public release, the authors must insert the actual six-model × five-run observations and seeds, run `python analyze_repeated_runs.py`, and confirm that Tables 5–6 and the manuscript statistics are reproduced. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

Run the final repository audit with:

```bash
python audit_package.py
```

It must report `"release_ready": true` before the GitHub URL is placed in the manuscript. For technical inspection while the documented raw-score gap remains, use `python audit_package.py --allow-incomplete`.

## Availability statement for the manuscript

After publication of the repository, replace the placeholder with the real URL:

> The implementation code, fixed data split, pretrained model weights, per-sample predictions, and scripts used to reproduce the reported FiLMoS-Net results are publicly available at: [INSERT PUBLIC GITHUB URL]. BUSI images are available from their original source and are not redistributed.
