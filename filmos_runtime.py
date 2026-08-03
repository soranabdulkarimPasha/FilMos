#!/usr/bin/env python3
"""Runtime helpers for the corrected FiLMoS-Net BUSI experiment.

This wrapper executes the notebook's actual definitions (rather than a rewritten
model), changes only the Colab data path, and exports an auditable split,
prediction table, and metrics JSON.  It intentionally does not generate or
modify any manuscript figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "FiLMoS_Net_complete_experiment.ipynb"
DATA_ROOT = Path(
    os.environ.get("FILMOS_BUSI_ROOT", ROOT / "data" / "Dataset_BUSI_with_GT")
).expanduser().resolve()
OUTPUT_ROOT = ROOT / "results" / "training_runs"
CLASS_NAMES = ["benign", "malignant", "normal"]


def load_notebook_namespace():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module_name = "filmos_notebook_module"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOK)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for cell_index in (0, 3, 4, 6, 7, 8, 9):
        source = "".join(payload["cells"][cell_index].get("source", []))
        exec(compile(source, f"{NOTEBOOK.name}:cell-{cell_index}", "exec"), namespace)
    return namespace


def make_config(ns, img_size=128, batch_size=32, epochs=24, train_epoch_multiplier=1.0, patience=6):
    return ns["Config"](
        data_root=str(DATA_ROOT),
        img_size=img_size,
        batch_size=batch_size,
        num_workers=2,
        seed=42,
        val_ratio=0.15,
        test_ratio=0.15,
        use_preaugmented_train=False,
        use_online_train_augmentation=True,
        use_weighted_sampler=False,
        train_epoch_multiplier=train_epoch_multiplier,
        use_mask_roi_crop=False,
        use_mask_geometry_channels=True,
        roi_margin=0.15,
        apply_basic_transforms=True,
        pretrained=False,
        epochs=epochs,
        freeze_epochs=0,
        lr=2e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        label_smoothing=0.02,
        class_weight_power=0.65,
        focal_gamma=0.0,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        sampler_class_boost=None,
        patience=patience,
    )


def json_safe(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def split_counts(groups):
    counts = Counter(int(item["label"]) for item in groups)
    return {CLASS_NAMES[i]: counts[i] for i in range(3)}


def write_split_manifest(ns, split_groups):
    rows = []
    for subset, groups in split_groups.items():
        samples = ns["groups_to_samples"](groups, include_augmented=False)
        for path, label in samples:
            rows.append(
                {
                    "subset": subset,
                    "class_index": int(label),
                    "class_name": CLASS_NAMES[int(label)],
                    "relative_path": str(Path(path).resolve().relative_to(DATA_ROOT.resolve())),
                }
            )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "busi_split_seed42.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke_test(ns, cfg, train_groups, val_groups, test_groups, class_names):
    import torch
    import torch.nn.functional as functional
    from dataclasses import replace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = ns["make_loaders_from_group_splits"](
        cfg, train_groups, val_groups, test_groups, class_names, display_name="smoke-test"
    )
    model = ns["build_model"](replace(cfg, model_name="filmos"), num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batch = next(iter(train_loader))
    x = batch["image"].to(device)
    y = batch["label"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            logits = model(x)["logits"]
            loss = functional.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(x)["logits"]
        loss = functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    elapsed = time.monotonic() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "device": str(device),
                "batch_shape": list(x.shape),
                "loss": float(loss.detach().cpu()),
                "seconds": elapsed,
                "peak_allocated_GiB": peak_gib,
            },
            indent=2,
        ),
        flush=True,
    )


def run_full(ns, cfg, train_groups, val_groups, test_groups, class_names, seeds):
    import numpy as np
    import torch
    from sklearn.metrics import (
        average_precision_score,
        cohen_kappa_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_outputs = []
    for model_name, run_seed in (("filmos", seed) for seed in seeds):
        print(f"\nStarting final ensemble member: {model_name}, seed={run_seed}", flush=True)
        run_outputs.append(
            ns["train_eval_one_run"](
                cfg,
                model_name,
                run_seed,
                device,
                train_groups,
                val_groups,
                test_groups,
                class_names,
            )
        )

    val_y = run_outputs[0]["val_y"]
    test_y = run_outputs[0]["test_y"]
    val_logits = torch.stack(
        [run["val_logits"] + run["bias"].view(1, -1) for run in run_outputs], dim=0
    ).mean(dim=0)
    test_logits = torch.stack(
        [run["test_logits"] + run["bias"].view(1, -1) for run in run_outputs], dim=0
    ).mean(dim=0)
    ensemble_bias, validation_metrics = ns["tune_logit_bias"](val_logits, val_y, num_classes=3)
    test_metrics = ns["metrics_from_logits"](
        test_logits, test_y, num_classes=3, logit_bias=ensemble_bias
    )

    calibrated_logits = test_logits + ensemble_bias.view(1, -1)
    probabilities = torch.softmax(calibrated_logits, dim=1).numpy()
    labels = test_y.numpy().astype(np.int64)
    predictions = calibrated_logits.argmax(dim=1).numpy().astype(np.int64)
    per_class_auc = [
        float(roc_auc_score((labels == index).astype(np.int64), probabilities[:, index]))
        for index in range(3)
    ]
    per_class_ap = [
        float(average_precision_score((labels == index).astype(np.int64), probabilities[:, index]))
        for index in range(3)
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    kappa = float(cohen_kappa_score(labels, predictions))

    test_samples = ns["groups_to_samples"](test_groups, include_augmented=False)
    with (OUTPUT_ROOT / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "relative_path",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "prob_benign",
            "prob_malignant",
            "prob_normal",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (path, _), truth, prediction, probability in zip(
            test_samples, labels, predictions, probabilities
        ):
            writer.writerow(
                {
                    "relative_path": str(Path(path).resolve().relative_to(DATA_ROOT.resolve())),
                    "true_index": int(truth),
                    "true_class": CLASS_NAMES[int(truth)],
                    "predicted_index": int(prediction),
                    "predicted_class": CLASS_NAMES[int(prediction)],
                    "prob_benign": f"{probability[0]:.10f}",
                    "prob_malignant": f"{probability[1]:.10f}",
                    "prob_normal": f"{probability[2]:.10f}",
                }
            )

    summary = {
        "provenance": {
            "notebook": NOTEBOOK.name,
            "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "ensemble": [{"model": "filmos", "seed": seed} for seed in seeds],
            "tta": ["original", "horizontal_flip", "vertical_flip", "brightness_1.03", "brightness_0.97"],
            "calibration": "validation-only additive class-logit bias",
        },
        "dataset": {
            "independent_input_images": 780,
            "class_counts": {"benign": 437, "malignant": 210, "normal": 133},
            "split_counts": {
                "train": split_counts(train_groups),
                "validation": split_counts(val_groups),
                "test": split_counts(test_groups),
            },
        },
        "ensemble_bias": ensemble_bias,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "independent_verification": {
            "cohen_kappa": kappa,
            "micro_f1_equals_accuracy": float((predictions == labels).mean()),
            "macro_f1": float(np.mean(f1)),
            "macro_sensitivity": float(np.mean(recall)),
            "macro_specificity": float(test_metrics["macro_specificity"]),
            "classwise": {
                CLASS_NAMES[i]: {
                    "precision": float(precision[i]),
                    "sensitivity": float(recall[i]),
                    "specificity": float(test_metrics["per_class"][i]["specificity"]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                    "auroc_ovr": per_class_auc[i],
                    "average_precision": per_class_ap[i],
                }
                for i in range(3)
            },
            "macro_auroc_ovr": float(np.mean(per_class_auc)),
            "macro_average_precision": float(np.mean(per_class_ap)),
        },
    }
    with (OUTPUT_ROOT / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
    print("\nFINAL_METRICS", flush=True)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one forward/backward batch only")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    ns = load_notebook_namespace()
    cfg = make_config(
        ns,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_epoch_multiplier=args.epoch_multiplier,
        patience=args.patience,
    )
    ns["seed_everything"](cfg.seed)
    _, train_groups, val_groups, test_groups, class_names = ns["make_group_splits"](cfg)
    print(
        json.dumps(
            {
                "input_counts": {"benign": 437, "malignant": 210, "normal": 133},
                "train": split_counts(train_groups),
                "validation": split_counts(val_groups),
                "test": split_counts(test_groups),
            },
            indent=2,
        ),
        flush=True,
    )
    write_split_manifest(ns, {"train": train_groups, "validation": val_groups, "test": test_groups})
    if args.smoke:
        smoke_test(ns, cfg, train_groups, val_groups, test_groups, class_names)
    else:
        run_full(ns, cfg, train_groups, val_groups, test_groups, class_names, args.seeds)


if __name__ == "__main__":
    main()
