#!/usr/bin/env python3
"""Audit the publication repository's predictions, runs, interventions, and tables."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
CLASSES = ("benign", "malignant", "normal")
MODELS = ("filmos_cold", "hybrid_cnn_lstm", "vit", "dsdnet", "ctmf_net", "sae_net")
BASELINES = MODELS[1:]
SEEDS = tuple(range(42, 47))


def prediction_path(model: str, seed: int) -> Path:
    if model == "filmos_cold":
        model = "filmos_net"
    return ROOT / "artifacts" / "comparative_runs" / model / f"seed_{seed}" / "test_predictions.csv"


def normalized(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={
        "prob_benign": "probability_benign",
        "prob_malignant": "probability_malignant",
        "prob_normal": "probability_normal",
    })
    if "true_index" not in frame:
        frame["true_index"] = frame["true_class"].map({name: i for i, name in enumerate(CLASSES)})
    if "predicted_index" not in frame:
        frame["predicted_index"] = frame["predicted_class"].map({name: i for i, name in enumerate(CLASSES)})
    return frame.sort_values("relative_path").reset_index(drop=True)


def prediction_check(path: Path, reference_ids: tuple[str, ...] | None) -> tuple[dict, tuple[str, ...]]:
    if not path.exists():
        return {"ok": False, "path": str(path), "reason": "missing"}, reference_ids or ()
    frame = normalized(path)
    ids = tuple(frame["relative_path"].astype(str))
    y = frame["true_index"].to_numpy(int)
    pred = frame["predicted_index"].to_numpy(int)
    class_f1 = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    ok = (
        len(frame) == 118
        and len(set(ids)) == 118
        and (reference_ids is None or ids == reference_ids)
        and abs(float(macro) - float(np.mean(class_f1))) < 1e-12
    )
    return {
        "ok": ok, "path": str(path.relative_to(ROOT)), "rows": len(frame),
        "unique_test_ids": len(set(ids)), "macro_f1": float(macro),
        "classwise_f1_mean": float(np.mean(class_f1)),
    }, ids if reference_ids is None else reference_ids


def main() -> int:
    checks: dict[str, object] = {}
    split = pd.read_csv(ROOT / "data" / "busi_split_seed42.csv")
    counts = Counter(zip(split["subset"], split["class_name"]))
    expected = {("train", "benign"): 305, ("train", "malignant"): 146, ("train", "normal"): 93,
                ("validation", "benign"): 66, ("validation", "malignant"): 32, ("validation", "normal"): 20,
                ("test", "benign"): 66, ("test", "malignant"): 32, ("test", "normal"): 20}
    checks["fixed_split"] = {"ok": len(split) == 780 and dict(counts) == expected, "rows": len(split)}

    reference_ids = None
    prediction_results = {}
    for model in MODELS:
        for seed in SEEDS:
            key = f"{model}_seed{seed}"
            prediction_results[key], reference_ids = prediction_check(prediction_path(model, seed), reference_ids)
    checks["comparative_predictions"] = {
        "ok": len(prediction_results) == 30 and all(item["ok"] for item in prediction_results.values()),
        "expected_files": 30, "files": prediction_results,
    }

    ablations = {}
    for mode in ("fixed", "orientation", "frequency", "morphology"):
        path = ROOT / "artifacts" / "routing_interventions" / mode / "test_predictions.csv"
        ablations[mode], reference_ids = prediction_check(path, reference_ids)
    checks["ablation_predictions"] = {
        "ok": all(item["ok"] for item in ablations.values()),
        "design": "post-training routing intervention on one shared seed-42 checkpoint",
        "files": ablations,
    }

    scores_path = ROOT / "data" / "repeated_run_scores.csv"
    scores = pd.read_csv(scores_path)
    run_ok = (
        len(scores) == 30 and scores["seed"].nunique() == 5
        and set(scores["seed"]) == set(SEEDS)
        and scores.groupby("seed")["model"].nunique().eq(6).all()
    )
    checks["run_level_scores"] = {"ok": bool(run_ok), "rows": len(scores), "seeds": sorted(scores["seed"].unique().tolist())}

    numerical_audit_path = ROOT / "results" / "manuscript" / "audit.json"
    numerical_audit = json.loads(numerical_audit_path.read_text()) if numerical_audit_path.exists() else {}
    required_outputs = [
        "table_1_dataset_split.csv", "table_2_single_checkpoints.csv", "table_3_primary_classwise.csv", "table_4_ablation.csv",
        "table_5_five_run_summary.csv", "table_6_paired_statistics.json", "primary_seed42_bootstrap.json",
        "run_level_metrics.csv", "all_metrics.json", "protocol_verification_checks.csv",
    ]
    checks["generated_results"] = {
        "ok": bool(numerical_audit.get("complete")) and all((ROOT / "results" / "manuscript" / name).exists() for name in required_outputs),
        "pipeline_audit_complete": bool(numerical_audit.get("complete")), "required_outputs": required_outputs,
    }

    release_ready = all(bool(item["ok"]) for item in checks.values())
    report = {"technical_checks_ok": release_ready, "release_ready": release_ready, "checks": checks}
    output = ROOT / "results" / "manuscript" / "repository_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if release_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
