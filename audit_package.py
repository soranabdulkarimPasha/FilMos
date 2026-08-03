#!/usr/bin/env python3
"""Run pre-release integrity checks for the FiLMoS-Net repository."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from filmos_net_architecture import FiLMoSNet


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="report missing raw paired runs without failing")
    args = parser.parse_args()
    checks: dict[str, dict[str, object]] = {}

    forbidden = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if "mask" in path.name.lower())
    checks["public_names"] = {"ok": not forbidden, "offending_paths": forbidden}

    with (ROOT / "data" / "busi_split_seed42.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    split_counts = Counter((row["subset"], row["class_name"]) for row in split_rows)
    expected = {("train", "benign"): 305, ("train", "malignant"): 146, ("train", "normal"): 93,
                ("validation", "benign"): 66, ("validation", "malignant"): 32, ("validation", "normal"): 20,
                ("test", "benign"): 66, ("test", "malignant"): 32, ("test", "normal"): 20}
    checks["split_manifest"] = {"ok": len(split_rows) == 780 and dict(split_counts) == expected, "rows": len(split_rows)}

    with (ROOT / "results" / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    names = {"benign": 0, "malignant": 1, "normal": 2}
    labels = np.asarray([names[row["true_class"]] for row in prediction_rows])
    predictions = np.asarray([names[row["predicted_class"]] for row in prediction_rows])
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    expected_matrix = np.asarray([[63, 3, 0], [5, 27, 0], [0, 0, 20]])
    checks["test_predictions"] = {"ok": len(prediction_rows) == 118 and np.array_equal(matrix, expected_matrix), "rows": len(prediction_rows), "confusion_matrix": matrix.tolist()}

    model = FiLMoSNet(num_classes=3)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint_results = {}
    for name in (
        "filmos_net_primary_seed42.pt",
        "filmos_net_cold_start_seed42.pt",
        "filmos_net_training_partition_initialization_seed42.pt",
    ):
        path = ROOT / "checkpoints" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        checkpoint_results[name] = {"epoch": int(payload["epoch"]), "missing_keys": missing, "unexpected_keys": unexpected}
    checks["architecture_and_checkpoints"] = {"ok": parameter_count == 1288966 and all(not item["missing_keys"] and not item["unexpected_keys"] for item in checkpoint_results.values()), "trainable_parameters": parameter_count, "checkpoints": checkpoint_results}

    with (ROOT / "data" / "repeated_run_scores.csv").open(newline="", encoding="utf-8") as handle:
        repeated_rows = list(csv.DictReader(handle))
    paired_ok = len(repeated_rows) == 30
    checks["paired_run_raw_scores"] = {"ok": paired_ok, "rows": len(repeated_rows), "required_rows": 30,
        "note": "Actual observations were absent from the supplied source folder; do not synthesize them." if not paired_ok else "Complete"}

    technical_ok = all(item["ok"] for key, item in checks.items() if key != "paired_run_raw_scores")
    release_ready = technical_ok and paired_ok
    report = {"technical_checks_ok": technical_ok, "release_ready": release_ready, "checks": checks}
    print(json.dumps(report, indent=2))
    if release_ready or (technical_ok and args.allow_incomplete):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
