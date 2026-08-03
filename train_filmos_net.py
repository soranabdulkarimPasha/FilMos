#!/usr/bin/env python3
"""Train FiLMoS-Net under the corrected BUSI protocol."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from filmos_runtime import ROOT, json_safe, load_notebook_namespace, make_config


def install_mild_augmentation(ns, preserve_full_frame=False):
    transforms = ns["transforms"]

    def get_train_transforms(
        img_size: int,
        use_online_augmentation: bool,
        preserve_full_frame: bool = preserve_full_frame,
    ):
        if not use_online_augmentation:
            return ns["get_eval_transforms"](img_size)
        spatial_start = (
            [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)]
            if preserve_full_frame
            else [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.92, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]
        )
        photometric = [] if preserve_full_frame else [transforms.ColorJitter(brightness=0.08, contrast=0.10)]
        return transforms.Compose(
            spatial_start
            + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5 if preserve_full_frame else 7),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            ]
            + photometric
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    ns["get_train_transforms"] = get_train_transforms


def install_lesion_annotation_input(ns):
    """Preserve full-frame lesion geometry in three aligned input channels."""
    dataset_class = ns["BUSIClassificationDataset"]
    mask_paths_for_image = ns["mask_paths_for_image"]

    def getitem(self, index):
        path, label = self.samples[index]
        with Image.open(path) as opened:
            gray = np.asarray(opened.convert("L"), dtype=np.uint8).copy()
        union = np.zeros_like(gray, dtype=np.uint8)
        for mask_path in mask_paths_for_image(path):
            try:
                with Image.open(mask_path) as opened_mask:
                    candidate = np.asarray(opened_mask.convert("L"), dtype=np.uint8)
            except Exception:
                continue
            union = np.maximum(union, (candidate > 0).astype(np.uint8) * 255)
        masked = np.where(union > 0, gray, 0).astype(np.uint8)
        # R=original context, G=binary lesion geometry, B=lesion-only texture.
        image = Image.merge(
            "RGB",
            (Image.fromarray(gray), Image.fromarray(union), Image.fromarray(masked)),
        )
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }

    dataset_class.__getitem__ = getitem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="filmos_net_seed42")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--roi-margin", type=float, default=0.15)
    parser.add_argument("--class-weight-power", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-mode",
        choices=("lesion_geometry", "roi_crop"),
        default="lesion_geometry",
    )
    args = parser.parse_args()

    ns = load_notebook_namespace()
    install_mild_augmentation(ns, preserve_full_frame=args.input_mode == "lesion_geometry")
    if args.input_mode == "lesion_geometry":
        install_lesion_annotation_input(ns)
    # Both checkpoint selection and the validation-only bias search now target
    # the metric requested by the study: unweighted macro-F1.
    ns["balanced_selection_score"] = lambda metric: float(metric["macro_f1"])

    cfg = make_config(
        ns,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_epoch_multiplier=1.0,
        patience=args.patience,
    )
    cfg = replace(
        cfg,
        seed=args.seed,
        lr=args.lr,
        use_preaugmented_train=False,
        use_weighted_sampler=False,
        use_online_train_augmentation=True,
        use_mask_roi_crop=args.input_mode == "roi_crop",
        use_mask_geometry_channels=args.input_mode == "lesion_geometry",
        roi_margin=args.roi_margin,
        class_weight_power=args.class_weight_power,
        focal_gamma=0.0,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        sampler_class_boost=None,
        label_smoothing=0.02,
    )
    ns["seed_everything"](args.seed)
    _, train_groups, validation_groups, test_groups, class_names = ns["make_group_splits"](cfg)
    train_loader, validation_loader, test_loader = ns["make_loaders_from_group_splits"](
        cfg,
        train_groups,
        validation_groups,
        test_groups,
        class_names,
        display_name=args.run_name,
    )
    class_weights = ns["compute_class_weights_from_loader"](
        train_loader, num_classes=3, power=cfg.class_weight_power
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ns["build_model"](replace(cfg, model_name="filmos"), num_classes=3).to(device)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Warm-started weights from {args.warm_start}", flush=True)

    trial_root = ROOT / "results" / "training_runs" / args.run_name
    trial_root.mkdir(parents=True, exist_ok=True)
    os.chdir(trial_root)
    best_path = ns["fit"](
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        num_classes=3,
        class_weights=class_weights,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        use_amp=True,
        label_smoothing=cfg.label_smoothing,
        focal_gamma=cfg.focal_gamma,
        mixup_alpha=cfg.mixup_alpha,
        mixup_prob=cfg.mixup_prob,
        min_lr=cfg.min_lr,
        patience=cfg.patience,
        freeze_epochs=0,
        ckpt_dir="checkpoints",
        monitor="macro_f1",
    )
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    bias = checkpoint["logit_bias"].detach().cpu()

    results = {}
    for use_tta in (False, True):
        key = "all_current_tta" if use_tta else "original_only"
        val_logits, val_y, val_loss = ns["collect_logits"](
            model, validation_loader, device, use_tta=use_tta
        )
        test_logits, test_y, test_loss = ns["collect_logits"](
            model, test_loader, device, use_tta=use_tta
        )
        # Calibrate once on validation logits for the matching inference mode.
        inference_bias, val_metrics = ns["tune_logit_bias"](val_logits, val_y, num_classes=3)
        test_metrics = ns["metrics_from_logits"](
            test_logits, test_y, num_classes=3, logit_bias=inference_bias, loss=test_loss
        )
        results[key] = {
            "bias": inference_bias,
            "validation": val_metrics,
            "test": test_metrics,
        }

    report = {
        "trial": args.run_name,
        "architecture_changed": False,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "best_epoch": int(checkpoint["epoch"]),
        "training_config": {
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "roi_margin": cfg.roi_margin,
            "weighted_sampler": cfg.use_weighted_sampler,
            "class_weight_power": cfg.class_weight_power,
            "focal_gamma": cfg.focal_gamma,
            "mixup_prob": cfg.mixup_prob,
            "augmentation": "mild",
            "input_mode": args.input_mode,
            "selection_metric": "macro_f1",
        },
        "saved_checkpoint_bias": bias,
        "results": results,
    }
    (trial_root / "metrics.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )
    print("TRIAL_RESULT")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
