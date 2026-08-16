#!/usr/bin/env python3
"""Synchronize run-level scores and create the repository SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "manuscript"
DISPLAY = {
    "filmos_cold": "FiLMoS-Net (cold start)",
    "hybrid_cnn_lstm": "Hybrid CNN-LSTM",
    "vit": "ViT",
    "dsdnet": "DSDNet (same-mask adaptation)",
    "ctmf_net": "CTMF-Net",
    "sae_net": "SAE-Net (image-spectrum adaptation)",
}


def main() -> None:
    source = RESULTS / "run_level_metrics.csv"
    if not source.exists():
        raise FileNotFoundError("Run recompute_manuscript_results.py first")
    runs = pd.read_csv(source)
    runs = runs[runs["model"].isin(DISPLAY) & runs["seed"].between(42, 46)].copy()
    if len(runs) != 30 or runs.groupby("model")["seed"].nunique().ne(5).any():
        raise ValueError("Expected 30 measured comparative rows with five distinct seeds per model")
    runs["run"] = runs["seed"].map({seed: index for index, seed in enumerate(range(42, 47), 1)})
    runs["model"] = runs["model"].map(DISPLAY)
    runs[["run", "seed", "model", "accuracy", "macro_f1", "macro_auroc", "macro_auprc"]].sort_values(
        ["run", "model"]
    ).to_csv(ROOT / "data" / "repeated_run_scores.csv", index=False)

    manifest_lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256" or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(runs)} run rows and {len(manifest_lines)} manifest entries")


if __name__ == "__main__":
    main()
