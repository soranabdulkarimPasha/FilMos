#!/usr/bin/env python3
"""Build every manuscript numerical result from archived predictions.

The script deliberately refuses incomplete or internally inconsistent inputs.
It is the single numerical source used for manuscript Tables 2--6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "results" / "manuscript"
CLASSES = ("benign", "malignant", "normal")
BASELINES = ("hybrid_cnn_lstm", "vit", "dsdnet", "ctmf_net", "sae_net")
DISPLAY = {
    "filmos_cold": "FiLMoS-Net (cold start)",
    "filmos_warm": "FiLMoS-Net (warm start; secondary)",
    "hybrid_cnn_lstm": "Hybrid CNN-LSTM",
    "vit": "ViT",
    "dsdnet": "DSDNet (same-mask adaptation)",
    "ctmf_net": "CTMF-Net",
    "sae_net": "SAE-Net (image-spectrum adaptation)",
}

ABLATIONS = ("fixed", "orientation", "frequency", "morphology")


def normalized_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "prob_benign": "probability_benign",
        "prob_malignant": "probability_malignant",
        "prob_normal": "probability_normal",
    }
    frame = frame.rename(columns=rename)
    if "true_index" not in frame:
        frame["true_index"] = frame["true_class"].map({name: i for i, name in enumerate(CLASSES)})
    if "predicted_index" not in frame:
        frame["predicted_index"] = frame["predicted_class"].map({name: i for i, name in enumerate(CLASSES)})
    required = {
        "relative_path", "true_index", "predicted_index",
        "probability_benign", "probability_malignant", "probability_normal",
    }
    if not required.issubset(frame.columns) or len(frame) != 118:
        raise ValueError(f"{path}: expected 118 rows and columns {sorted(required)}")
    if frame["relative_path"].duplicated().any():
        raise ValueError(f"{path}: duplicate test identifiers")
    return frame.sort_values("relative_path").reset_index(drop=True)


def metric_record(frame: pd.DataFrame) -> dict[str, object]:
    y = frame["true_index"].to_numpy(int)
    pred = frame["predicted_index"].to_numpy(int)
    prob = frame[[f"probability_{name}" for name in CLASSES]].to_numpy(float)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    specificity = []
    auc = []
    ap = []
    classwise = {}
    for index, name in enumerate(CLASSES):
        tn = cm.sum() - cm[index].sum() - cm[:, index].sum() + cm[index, index]
        fp = cm[:, index].sum() - cm[index, index]
        specificity.append(tn / (tn + fp))
        binary = (y == index).astype(int)
        auc.append(roc_auc_score(binary, prob[:, index]))
        ap.append(average_precision_score(binary, prob[:, index]))
        classwise[name] = {
            "support": int(support[index]), "precision": float(precision[index]),
            "sensitivity": float(recall[index]), "specificity": float(specificity[index]),
            "f1": float(f1[index]), "auroc": float(auc[index]),
            "average_precision": float(ap[index]),
        }
    macro_f1 = float(f1_score(y, pred, average="macro"))
    arithmetic = float(np.mean(f1))
    if not math.isclose(macro_f1, arithmetic, abs_tol=1e-12):
        raise AssertionError("macro-F1 differs from the unweighted class-wise mean")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "macro_f1": macro_f1,
        "macro_sensitivity": float(np.mean(recall)),
        "macro_specificity": float(np.mean(specificity)),
        "macro_auroc": float(np.mean(auc)),
        "macro_auprc": float(np.mean(ap)),
        "confusion_matrix": cm.tolist(),
        "classwise": classwise,
    }


def bootstrap_intervals(frame: pd.DataFrame, repetitions: int = 5000, seed: int = 2026) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    y = frame["true_index"].to_numpy(int)
    groups = [np.flatnonzero(y == index) for index in range(3)]
    samples = {name: [] for name in ("accuracy", "macro_f1", "macro_auroc", "macro_auprc")}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        item = metric_record(frame.iloc[indices].reset_index(drop=True))
        for name in samples:
            samples[name].append(item[name])
    return {name: {"lower_95": float(np.percentile(values, 2.5)), "upper_95": float(np.percentile(values, 97.5))} for name, values in samples.items()}


def run_paths() -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for seed in range(42, 47):
        paths[("filmos_cold", seed)] = ARTIFACTS / "comparative_runs" / "filmos_net" / f"seed_{seed}" / "test_predictions.csv"
    paths[("filmos_warm", 42)] = ARTIFACTS / "protocol_checks" / "warm_start_seed42" / "test_predictions.csv"
    for model in BASELINES:
        for seed in range(42, 47):
            paths[(model, seed)] = ARTIFACTS / "comparative_runs" / model / f"seed_{seed}" / "test_predictions.csv"
    return paths


def paired_analysis(frame: pd.DataFrame, endpoint: str) -> dict[str, object]:
    order = ("filmos_cold", *BASELINES)
    pivot = frame.pivot(index="seed", columns="model", values=endpoint).loc[:, order]
    if pivot.shape != (5, 6) or pivot.isna().any().any():
        raise ValueError(f"Incomplete five-run matrix for {endpoint}: {pivot.shape}")
    arrays = [pivot[name].to_numpy() for name in order]
    statistic, p_value = friedmanchisquare(*arrays)
    ranks = np.vstack([rankdata(-row, method="average") for row in pivot.to_numpy()])
    mean_ranks = ranks.mean(axis=0)
    n, k = pivot.shape
    se = math.sqrt(k * (k + 1) / (6 * n))
    cd = 2.850 * se
    comparisons = []
    for index, name in enumerate(order[1:], 1):
        gap = abs(float(mean_ranks[index] - mean_ranks[0]))
        q_value = gap / se * math.sqrt(2)
        comparisons.append({
            "baseline": DISPLAY[name], "rank_gap": gap,
            "p_value": float(studentized_range.sf(q_value, k, np.inf)),
            "significant_0_05": bool(gap > cd),
        })
    return {
        "friedman_chi_square": float(statistic), "friedman_p": float(p_value),
        "kendalls_w": float(statistic / (n * (k - 1))),
        "nemenyi_critical_difference": float(cd), "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(ROOT / "data" / "busi_split_seed42.csv")
    split_counts = split.groupby(["class_name", "subset"]).size().unstack(fill_value=0)
    table1 = []
    for name in CLASSES:
        row = {"class": name.title()}
        for subset in ("train", "validation", "test"):
            row[subset] = int(split_counts.loc[name, subset])
        row["total"] = row["train"] + row["validation"] + row["test"]
        table1.append(row)
    table1.append({
        "class": "Total",
        "train": sum(row["train"] for row in table1),
        "validation": sum(row["validation"] for row in table1),
        "test": sum(row["test"] for row in table1),
        "total": sum(row["total"] for row in table1),
    })
    pd.DataFrame(table1, columns=("class", "total", "train", "validation", "test")).to_csv(
        OUT / "table_1_dataset_split.csv", index=False
    )
    paths = run_paths()
    ablation_paths = {
        # Table 4 is a post-training routing intervention on the same selected
        # full-model checkpoint. This holds learned weights and epoch fixed.
        mode: ARTIFACTS / "routing_interventions" / mode / "test_predictions.csv"
        for mode in ABLATIONS
    }
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    missing = [str(path) for path in (*paths.values(), *ablation_paths.values()) if not path.exists()]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("Missing repeated-run predictions:\n" + "\n".join(missing))
    for key, path in paths.items():
        if path.exists():
            frames[key] = normalized_predictions(path)
    reference_ids = None
    records = []
    metrics: dict[str, dict] = {}
    for (model, seed), frame in frames.items():
        ids = tuple(frame["relative_path"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise AssertionError(f"Test identifiers differ for {model}, seed {seed}")
        result = metric_record(frame)
        metrics[f"{model}_seed{seed}"] = result
        records.append({"model": model, "display_name": DISPLAY[model], "seed": seed, **{
            key: result[key] for key in ("accuracy", "macro_f1", "macro_auroc", "macro_auprc")
        }})
    runs = pd.DataFrame(records).sort_values(["model", "seed"])
    runs.to_csv(OUT / "run_level_metrics.csv", index=False)

    # Single-checkpoint comparison contains only consistently initialized runs.
    single_order = (("filmos_cold", 42), *((name, 42) for name in BASELINES))
    table2 = []
    for model, seed in single_order:
        if (model, seed) not in frames:
            continue
        item = metrics[f"{model}_seed{seed}"]
        table2.append({
            "model": DISPLAY[model], "seed": seed,
            **{key: item[key] for key in ("accuracy", "cohen_kappa", "macro_f1", "macro_sensitivity", "macro_specificity", "macro_auroc", "macro_auprc")},
            **{f"f1_{name}": item["classwise"][name]["f1"] for name in CLASSES},
        })
    pd.DataFrame(table2).to_csv(OUT / "table_2_single_checkpoints.csv", index=False)

    if ("filmos_cold", 42) in frames:
        intervals = bootstrap_intervals(frames[("filmos_cold", 42)])
        (OUT / "primary_seed42_bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
        primary = metrics["filmos_cold_seed42"]
        rows = []
        for name in CLASSES:
            rows.append({"class": name.title(), **primary["classwise"][name]})
        pd.DataFrame(rows).to_csv(OUT / "table_3_primary_classwise.csv", index=False)

    # Protocol checks are deliberately separate from the comparative table.
    checks = []
    for label, model_key, source in (
        ("Warm-start verification", "filmos_warm_seed42", "same-partition FiLMoS initialization; lr=2e-4"),
    ):
        if model_key in metrics:
            item = metrics[model_key]
            checks.append({"analysis": label, "source": source, "accuracy": item["accuracy"], "macro_f1": item["macro_f1"], "macro_auroc": item["macro_auroc"]})
    verification = ARTIFACTS / "protocol_checks" / "cold_start_lr_7e4_seed42" / "test_predictions.csv"
    if verification.exists():
        item = metric_record(normalized_predictions(verification))
        checks.append({"analysis": "Independent cold-start verification", "source": "cold start; lr=7e-4", "accuracy": item["accuracy"], "macro_f1": item["macro_f1"], "macro_auroc": item["macro_auroc"]})
    pd.DataFrame(checks).to_csv(OUT / "protocol_verification_checks.csv", index=False)

    ablation_rows = []
    ablation_frames: dict[str, pd.DataFrame] = {}
    if ("filmos_cold", 42) in frames:
        full = metrics["filmos_cold_seed42"]
        ablation_rows.append({"configuration": "Full FiLMoS-Net (adaptive routing)", **{key: full[key] for key in ("accuracy", "macro_f1", "macro_auroc")}, **{f"f1_{name}": full["classwise"][name]["f1"] for name in CLASSES}})
    for mode in ABLATIONS:
        path = ablation_paths[mode]
        if path.exists():
            ablation_frames[mode] = normalized_predictions(path)
            if tuple(ablation_frames[mode]["relative_path"]) != reference_ids:
                raise AssertionError(f"Test identifiers differ for ablation {mode}")
            item = metric_record(ablation_frames[mode])
            ablation_rows.append({"configuration": mode.title(), **{key: item[key] for key in ("accuracy", "macro_f1", "macro_auroc")}, **{f"f1_{name}": item["classwise"][name]["f1"] for name in CLASSES}})
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(OUT / "table_4_ablation.csv", index=False)

    if all((model, seed) in frames for model in ("filmos_cold", *BASELINES) for seed in range(42, 47)):
        summary_rows = []
        for model in ("filmos_cold", *BASELINES):
            subset = runs[(runs.model == model) & runs.seed.between(42, 46)]
            if subset.seed.nunique() != 5:
                raise AssertionError(f"{model}: five distinct seeds required")
            row = {"model": DISPLAY[model]}
            for endpoint in ("accuracy", "macro_f1", "macro_auroc"):
                values = subset[endpoint].to_numpy(float)
                mean, sd = values.mean(), values.std(ddof=1)
                half = t.ppf(.975, 4) * sd / math.sqrt(5)
                row.update({f"{endpoint}_mean": mean, f"{endpoint}_sd": sd,
                            f"{endpoint}_ci_low": mean-half, f"{endpoint}_ci_high": mean+half})
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(OUT / "table_5_five_run_summary.csv", index=False)
        statistics = {endpoint: paired_analysis(runs, endpoint) for endpoint in ("accuracy", "macro_f1")}
        (OUT / "table_6_paired_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    (OUT / "all_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_hashes = {f"{model}_seed{seed}": hashlib.sha256(path.read_bytes()).hexdigest() for (model, seed), path in paths.items() if path.exists()}
    for mode in ABLATIONS:
        path = ARTIFACTS / "routing_interventions" / mode / "test_predictions.csv"
        if path.exists():
            prediction_hashes[f"routing_intervention_{mode}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_hashes = {}
    for model in ("filmos_net", *BASELINES):
        for seed in range(42, 47):
            checkpoint = ARTIFACTS / "comparative_runs" / model / f"seed_{seed}" / "best_checkpoint.pt"
            if checkpoint.exists():
                checkpoint_hashes[f"{model}_seed{seed}"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = {
        "prediction_files_found": len(frames) + len(ablation_frames),
        "prediction_files_expected": len(paths) + len(ABLATIONS),
        "comparative_and_protocol_files_found": len(frames),
        "ablation_files_found": len(ablation_frames),
        "ablation_design": "post-training routing intervention on one shared seed-42 checkpoint",
        "all_test_sets_identical": True, "test_images_per_file": 118,
        "macro_f1_identity_checked": True,
        "five_distinct_seed_requirement": [42, 43, 44, 45, 46],
        "checkpoint_files_found": len(checkpoint_hashes),
        "checkpoint_files_expected": 30,
        "checkpoint_hashes_unique": len(set(checkpoint_hashes.values())) == len(checkpoint_hashes),
        "prediction_hashes_unique": len(set(prediction_hashes.values())) == len(prediction_hashes),
        "complete": (
            len(frames) == len(paths)
            and len(ablation_frames) == len(ABLATIONS)
            and len(checkpoint_hashes) == 30
            and len(set(checkpoint_hashes.values())) == 30
            and len(set(prediction_hashes.values())) == len(prediction_hashes)
        ),
        "prediction_sha256": prediction_hashes,
        "checkpoint_sha256": checkpoint_hashes,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
