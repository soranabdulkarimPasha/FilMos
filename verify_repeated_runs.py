#!/usr/bin/env python3
"""Analyze the paired five-run ACC/MF1 scores reported in the manuscript."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t


ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ("FiLMoS-Net (cold start)", "Hybrid CNN-LSTM", "ViT", "DSDNet (same-mask adaptation)", "CTMF-Net", "SAE-Net (image-spectrum adaptation)")


def validate(frame: pd.DataFrame) -> None:
    required = {"run", "seed", "model", "accuracy", "macro_f1", "macro_auroc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected at least columns {sorted(required)}, found {list(frame.columns)}")
    if frame.empty:
        raise ValueError(
            "data/repeated_run_scores.csv contains no observations. Supply the 30 independently generated "
            "rows (six models x five paired training seeds)."
        )
    if len(frame) != 30 or set(frame["model"]) != set(MODEL_ORDER) or frame["run"].nunique() != 5:
        raise ValueError("Expected exactly six models on each of five paired runs (30 rows total).")
    counts = frame.groupby("run")["model"].nunique()
    if not (counts == 6).all():
        raise ValueError("Every run must contain one observation for each of the six models.")


def analyze_endpoint(frame: pd.DataFrame, endpoint: str) -> dict[str, object]:
    pivot = frame.pivot(index="run", columns="model", values=endpoint).loc[:, MODEL_ORDER]
    arrays = [pivot[name].to_numpy() for name in MODEL_ORDER]
    statistic, p_value = friedmanchisquare(*arrays)
    ranks = np.vstack([rankdata(-row, method="average") for row in pivot.to_numpy()])
    mean_ranks = ranks.mean(axis=0)
    n_runs, n_models = pivot.shape
    critical_difference = 2.850 * math.sqrt(n_models * (n_models + 1) / (6 * n_runs))
    standard_error = math.sqrt(n_models * (n_models + 1) / (6 * n_runs))
    summaries = []
    for index, name in enumerate(MODEL_ORDER):
        values = arrays[index]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(t.ppf(0.975, n_runs - 1) * sd / math.sqrt(n_runs))
        summaries.append(
            {"model": name, "mean": mean, "sample_sd": sd, "ci_low": mean - half_width,
             "ci_high": mean + half_width, "mean_rank": float(mean_ranks[index])}
        )
    comparisons = []
    for index, name in enumerate(MODEL_ORDER[1:], start=1):
        gap = abs(float(mean_ranks[index] - mean_ranks[0]))
        q_value = gap / standard_error * math.sqrt(2)
        nemenyi_p = float(studentized_range.sf(q_value, n_models, np.inf))
        comparisons.append(
            {"baseline": name, "rank_gap": gap, "p_value": nemenyi_p,
             "decision_at_0.05": "Significant" if gap > critical_difference else "Not significant"}
        )
    return {
        "friedman_chi_square": float(statistic),
        "friedman_p": float(p_value),
        "kendalls_w": float(statistic / (n_runs * (n_models - 1))),
        "nemenyi_critical_difference": critical_difference,
        "summary": summaries,
        "comparisons_with_filmos_net": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "repeated_run_scores.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "manuscript" / "paired_run_statistics_independent_check.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    try:
        validate(frame)
    except ValueError as error:
        parser.error(str(error))
    report = {endpoint: analyze_endpoint(frame, endpoint) for endpoint in ("accuracy", "macro_f1")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
