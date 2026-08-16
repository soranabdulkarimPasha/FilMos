#!/usr/bin/env python3
"""Train FiLMoS-Net under the corrected BUSI protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from filmos_runtime import ROOT, json_safe, load_notebook_namespace, make_config


def install_fusion_mode(model, mode: str) -> None:
    """Install the selected fusion rule on the notebook-defined model."""
    if mode == "adaptive":
        return

    def forward(self, x):
        f1 = self.branch_gabor(x)
        f2 = self.branch_spec(x)
        f3 = self.branch_morph(x)
        if mode == "fixed":
            w = torch.full((x.size(0), 3), 1.0 / 3.0, device=x.device, dtype=f1.dtype)
        elif mode == "orientation":
            w = torch.tensor((1.0, 0.0, 0.0), device=x.device, dtype=f1.dtype).expand(x.size(0), -1)
        elif mode == "frequency":
            w = torch.tensor((0.0, 1.0, 0.0), device=x.device, dtype=f1.dtype).expand(x.size(0), -1)
        elif mode == "morphology":
            w = torch.tensor((0.0, 0.0, 1.0), device=x.device, dtype=f1.dtype).expand(x.size(0), -1)
        fused = sum(w[:, index].view(-1, 1, 1, 1) * feat for index, feat in enumerate((f1, f2, f3)))
        logits = self.head(self.refine(fused))
        return {"logits": logits, "router_w": w}

    model.forward = types.MethodType(forward, model)


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
    parser.add_argument("--run-name", default=None, help="Optional display label; artifacts are keyed by seed.")
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
        "--fusion-mode",
        choices=("adaptive", "fixed", "orientation", "frequency", "morphology"),
        default="adaptive",
    )
    parser.add_argument(
        "--input-mode",
        choices=("lesion_geometry", "roi_crop"),
        default="lesion_geometry",
    )
    args = parser.parse_args()
    run_name = args.run_name or f"filmos_net_seed_{args.seed}"

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
    # The image identifiers stay fixed at the authoritative seed-42 split;
    # ``args.seed`` controls initialization, shuffling, and augmentation only.
    split_cfg = replace(cfg, seed=42)
    _, train_groups, validation_groups, test_groups, class_names = ns["make_group_splits"](split_cfg)
    ns["seed_everything"](args.seed)
    train_loader, validation_loader, test_loader = ns["make_loaders_from_group_splits"](
        cfg,
        train_groups,
        validation_groups,
        test_groups,
        class_names,
        display_name=run_name,
    )
    class_weights = ns["compute_class_weights_from_loader"](
        train_loader, num_classes=3, power=cfg.class_weight_power
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ns["build_model"](replace(cfg, model_name="filmos"), num_classes=3).to(device)
    model.fusion_mode = args.fusion_mode
    install_fusion_mode(model, args.fusion_mode)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Warm-started weights from {args.warm_start}", flush=True)

    trial_root = ROOT / "artifacts" / "comparative_runs" / "filmos_net" / f"seed_{args.seed}"
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
    shutil.copy2(best_path, trial_root / "best_checkpoint.pt")
    bias = checkpoint["logit_bias"].detach().cpu()

    results = {}
    prediction_payload = None
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
        if use_tta:
            calibrated = test_logits + inference_bias.view(1, -1)
            prediction_payload = (
                test_y.cpu().numpy(), calibrated.argmax(dim=1).cpu().numpy(),
                torch.softmax(calibrated, dim=1).cpu().numpy(),
            )

    report = {
        "trial": run_name,
        "architecture_changed": False,
        "fusion_mode": args.fusion_mode,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "best_epoch": int(checkpoint["epoch"]),
        "training_config": {
            "split_seed": 42,
            "training_seed": args.seed,
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
    if prediction_payload is not None:
        labels, predictions, probabilities = prediction_payload
        samples = ns["groups_to_samples"](test_groups, include_augmented=False)
        with (trial_root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["relative_path", "true_index", "true_class", "predicted_index", "predicted_class", "probability_benign", "probability_malignant", "probability_normal"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for (path, _), truth, prediction, probability in zip(samples, labels, predictions, probabilities):
                writer.writerow({
                    "relative_path": str(Path(path).resolve().relative_to(Path(cfg.data_root).resolve())),
                    "true_index": int(truth), "true_class": class_names[int(truth)],
                    "predicted_index": int(prediction), "predicted_class": class_names[int(prediction)],
                    "probability_benign": f"{probability[0]:.10f}",
                    "probability_malignant": f"{probability[1]:.10f}",
                    "probability_normal": f"{probability[2]:.10f}",
                })
    print("TRIAL_RESULT")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
