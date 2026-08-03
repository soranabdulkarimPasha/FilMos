#!/usr/bin/env python3
"""Evaluate a fixed FiLMoS-Net checkpoint without further training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from filmos_runtime import ROOT, json_safe, load_notebook_namespace, make_config
from train_filmos_net import install_lesion_annotation_input, install_mild_augmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=ROOT / "checkpoints" / "filmos_net_primary_seed42.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "checkpoint_evaluation.json")
    args = parser.parse_args()

    ns = load_notebook_namespace()
    install_mild_augmentation(ns, preserve_full_frame=True)
    install_lesion_annotation_input(ns)
    ns["balanced_selection_score"] = lambda metric: float(metric["macro_f1"])
    cfg = make_config(ns, img_size=128, batch_size=32, epochs=30, train_epoch_multiplier=1.0, patience=8)
    cfg = replace(
        cfg,
        model_name="filmos",
        use_preaugmented_train=False,
        use_weighted_sampler=False,
        use_mask_roi_crop=False,
        use_mask_geometry_channels=True,
        use_online_train_augmentation=True,
        class_weight_power=0.65,
        focal_gamma=0.0,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        sampler_class_boost=None,
    )
    ns["seed_everything"](42)
    _, train_groups, validation_groups, test_groups, class_names = ns["make_group_splits"](cfg)
    _, validation_loader, test_loader = ns["make_loaders_from_group_splits"](
        cfg, train_groups, validation_groups, test_groups, class_names, display_name="fixed-checkpoint-evaluation"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ns["build_model"](cfg, num_classes=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    report = {"checkpoint": str(args.checkpoint), "epoch": int(checkpoint["epoch"]), "results": {}}
    for use_tta in (False, True):
        name = "original_only" if not use_tta else "all_current_tta"
        val_logits, val_y, _ = ns["collect_logits"](model, validation_loader, device, use_tta=use_tta)
        test_logits, test_y, test_loss = ns["collect_logits"](model, test_loader, device, use_tta=use_tta)
        bias, validation_metrics = ns["tune_logit_bias"](val_logits, val_y, num_classes=3)
        test_metrics = ns["metrics_from_logits"](
            test_logits, test_y, num_classes=3, logit_bias=bias, loss=test_loss
        )
        report["results"][name] = {
            "bias": bias,
            "validation": validation_metrics,
            "test": test_metrics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
