# FiLMoS-Net reproducibility repository

This repository contains the complete, publication-facing code and artifacts for the three-class BUSI experiments reported in the FiLMoS-Net manuscript. File and directory names describe their scientific role and do not depend on a manuscript revision number.

## What is authoritative

- `artifacts/comparative_runs/` contains the 30 validation-selected cold-start runs used in Tables 2, 5, and 6: six architectures × seeds 42–46. Each run has one final checkpoint and its prediction/result artifacts.
- `artifacts/routing_interventions/` contains the five fixed-checkpoint routing modes used in Table 4. These are inference-time interventions on the shared seed-42 FiLMoS-Net checkpoint; they are not independently retrained reduced models.
- `artifacts/protocol_checks/` contains the warm-start and learning-rate sensitivity checks described separately in the manuscript and excluded from comparative Tables 2, 5, and 6.
- `results/manuscript/` contains the generated Tables 1–6, figures, bootstrap intervals, run-level metrics, statistics, and audits used in the manuscript.
- `data/busi_split_seed42.csv` is the authoritative 544/118/118 image partition. BUSI images are not redistributed.

Obsolete exploratory runs, duplicate `last.pt` files, earlier table exports, and revision-specific directories are deliberately excluded.

## Repository structure

```text
.
|-- artifacts/
|   |-- comparative_runs/             30 final runs, grouped by model and seed
|   |-- routing_interventions/         adaptive/fixed/forced-branch predictions
|   `-- protocol_checks/               two secondary FiLMoS-Net checks
|-- baselines/source/                  five comparison implementations
|-- config/                            seeds and exact training protocols
|-- data/                              fixed split and 30-row run-level scores
|-- docs/                              manuscript-to-artifact map
|-- notebooks/                         complete experiment notebook
|-- results/manuscript/                publication tables, figures, and audits
|-- filmos_net_architecture.py         FiLMoS-Net definition
|-- filmos_runtime.py                  shared runtime helpers
|-- train_filmos_net.py                FiLMoS-Net training entry point
|-- evaluate_filmos_net.py             fixed-checkpoint intervention evaluator
|-- recompute_manuscript_results.py    authoritative prediction-to-table pipeline
|-- generate_manuscript_figures.py     Figs. 8 and 9 from predictions
|-- verify_repeated_runs.py            independent paired-statistics check
|-- audit_repository.py                repository consistency audit
`-- build_manifest.py                  run-score synchronization and SHA-256 manifest
```

See [docs/ARTIFACT_MAP.md](docs/ARTIFACT_MAP.md) for the exact link between each manuscript table/figure and its source files.
See [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) for a clean first commit and upload workflow.

## Environment and dataset

The experiments used Python 3.10, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA mixed precision, and deterministic CuDNN execution.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download BUSI from Al-Dhabyani et al., *Data in Brief* (2020), DOI `10.1016/j.dib.2019.104863`, and follow [DATASET.md](DATASET.md). Place it at `data/Dataset_BUSI_with_GT/` or set `FILMOS_BUSI_ROOT` to an external dataset directory.

## Reproduce the comparative training

FiLMoS-Net, for each seed 42–46:

```bash
python train_filmos_net.py \
  --input-mode lesion_geometry --fusion-mode adaptive \
  --img-size 128 --batch-size 32 --epochs 24 --patience 6 \
  --lr 2e-4 --class-weight-power 0.65 --seed 42
```

Change only `--seed` for seeds 43–46. Outputs are written to `artifacts/comparative_runs/filmos_net/seed_<SEED>/`.

All five baselines for one seed:

```bash
python baselines/source/run_all.py --seed 42
```

Again repeat for seeds 43–46. Every model uses the fixed seed-42 image partition while the training seed controls initialization, shuffling, and augmentation.

## Reproduce Table 4 without retraining

```bash
for MODE in adaptive fixed orientation frequency morphology; do
  python evaluate_filmos_net.py \
    artifacts/comparative_runs/filmos_net/seed_42/best_checkpoint.pt \
    --fusion-mode ${MODE} \
    --output artifacts/routing_interventions/${MODE}/metrics.json \
    --predictions artifacts/routing_interventions/${MODE}/test_predictions.csv
done
```

The checkpoint and learned network parameters remain fixed. Only routing weights are changed at inference, and each mode's additive logit bias is selected on validation data before the held-out test evaluation.

## Regenerate and verify manuscript results

These commands do not retrain any model:

```bash
python recompute_manuscript_results.py
python generate_manuscript_figures.py \
  artifacts/comparative_runs/filmos_net/seed_42/test_predictions.csv \
  --output-dir results/manuscript/figures
python verify_repeated_runs.py
python audit_repository.py
python build_manifest.py
sha256sum -c MANIFEST.sha256
```

The numerical pipeline rejects missing runs, duplicate samples, inconsistent test identifiers, incomplete checkpoints, and disagreement between macro-F1 and the unweighted mean of the three class-wise F1 values. A publishable state requires:

- `results/manuscript/audit.json`: `"complete": true`
- `results/manuscript/repository_audit.json`: `"release_ready": true`
- successful verification of every entry in `MANIFEST.sha256`

## Availability

The manuscript points to <https://github.com/soranabdulkarimPasha/FilMos>. Before public release, the authors should add the institutionally approved license; no license is inferred by this package.
