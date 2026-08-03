#!/usr/bin/env python3
"""Export the final, validation-calibrated FiLMoS test results."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from filmos_runtime import DATA_ROOT, ROOT, json_safe, load_notebook_namespace, make_config


CHECKPOINT = ROOT / "checkpoints" / "filmos_net_primary_seed42.pt"
OUTPUT = ROOT / "results"
CLASS_NAMES = ["benign", "malignant", "normal"]


def metric_set(labels, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "micro_f1": float(f1_score(labels, predictions, average="micro")),
        "macro_auroc_ovr": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        "macro_average_precision": float(
            np.mean(
                [average_precision_score((labels == index).astype(int), probabilities[:, index]) for index in range(3)]
            )
        ),
    }


def stratified_bootstrap(labels, predictions, probabilities, repetitions=5000, seed=2026):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == index) for index in range(3)]
    values = {name: [] for name in metric_set(labels, predictions, probabilities)}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_class])
        result = metric_set(labels[indices], predictions[indices], probabilities[indices])
        for name, value in result.items():
            values[name].append(value)
    return {
        name: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def main():
    ns = load_notebook_namespace()
    cfg = make_config(ns, img_size=128, batch_size=32, epochs=30, train_epoch_multiplier=1.0, patience=8)
    cfg = replace(
        cfg,
        model_name="filmos",
        use_preaugmented_train=False,
        use_online_train_augmentation=True,
        use_weighted_sampler=False,
        use_mask_roi_crop=False,
        use_mask_geometry_channels=True,
        class_weight_power=0.65,
        focal_gamma=0.0,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        sampler_class_boost=None,
        label_smoothing=0.02,
    )
    ns["seed_everything"](42)
    _, train_groups, validation_groups, test_groups, class_names = ns["make_group_splits"](cfg)
    _, validation_loader, test_loader = ns["make_loaders_from_group_splits"](
        cfg, train_groups, validation_groups, test_groups, class_names, display_name="final-export"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ns["build_model"](cfg, num_classes=3).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    validation_logits, validation_y, _ = ns["collect_logits"](model, validation_loader, device, use_tta=True)
    test_logits, test_y, test_loss = ns["collect_logits"](model, test_loader, device, use_tta=True)
    bias, validation_metrics = ns["tune_logit_bias"](validation_logits, validation_y, num_classes=3)
    test_metrics = ns["metrics_from_logits"](
        test_logits, test_y, num_classes=3, logit_bias=bias, loss=test_loss
    )
    calibrated_logits = test_logits + bias.view(1, -1)
    probabilities = torch.softmax(calibrated_logits, dim=1).numpy()
    labels = test_y.numpy().astype(np.int64)
    predictions = calibrated_logits.argmax(dim=1).numpy().astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    classwise = {}
    for index, class_name in enumerate(CLASS_NAMES):
        classwise[class_name] = {
            "support": int(support[index]),
            "precision": float(precision[index]),
            "sensitivity_recall": float(recall[index]),
            "specificity": float(test_metrics["per_class"][index]["specificity"]),
            "f1": float(f1[index]),
            "auroc_ovr": float(roc_auc_score((labels == index).astype(int), probabilities[:, index])),
            "average_precision": float(
                average_precision_score((labels == index).astype(int), probabilities[:, index])
            ),
        }

    split_counts = {
        subset: {
            CLASS_NAMES[index]: int(sum(int(group["label"]) == index for group in groups))
            for index in range(3)
        }
        for subset, groups in {
            "train": train_groups,
            "validation": validation_groups,
            "test": test_groups,
        }.items()
    }
    point_metrics = metric_set(labels, predictions, probabilities)
    report = {
        "protocol": {
            "model": "FiLMoS-Net",
            "architecture_changed": False,
            "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "seed": 42,
            "input_size": [128, 128],
            "batch_size": 32,
            "mask_assisted_channels": ["original grayscale context", "binary lesion mask", "lesion-only grayscale texture"],
            "weighted_sampler": False,
            "class_weight_power": 0.65,
            "focal_gamma": 0.0,
            "mixup": False,
            "tta": ["original", "horizontal flip", "vertical flip", "normalized-intensity x1.03", "normalized-intensity x0.97"],
            "calibration": "one additive class-logit bias selected on validation macro-F1 only",
            "validation_bias": bias,
        },
        "dataset": {
            "independent_images": 780,
            "input_class_counts": {"benign": 437, "malignant": 210, "normal": 133},
            "split_counts": split_counts,
        },
        "validation": validation_metrics,
        "test": {
            **point_metrics,
            "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
            "macro_sensitivity": float(np.mean(recall)),
            "macro_specificity": float(test_metrics["macro_specificity"]),
            "confusion_matrix": test_metrics["cm"],
            "classwise": classwise,
            "stratified_bootstrap_95_ci": stratified_bootstrap(labels, predictions, probabilities),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "final_metrics.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    samples = ns["groups_to_samples"](test_groups, include_augmented=False)
    with (OUTPUT / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "relative_path", "true_class", "predicted_class", "prob_benign", "prob_malignant", "prob_normal"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (path, _), truth, prediction, probability in zip(samples, labels, predictions, probabilities):
            writer.writerow(
                {
                    "relative_path": str(Path(path).resolve().relative_to(DATA_ROOT.resolve())),
                    "true_class": CLASS_NAMES[truth],
                    "predicted_class": CLASS_NAMES[prediction],
                    "prob_benign": f"{probability[0]:.10f}",
                    "prob_malignant": f"{probability[1]:.10f}",
                    "prob_normal": f"{probability[2]:.10f}",
                }
            )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
