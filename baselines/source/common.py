#!/usr/bin/env python3
"""Shared, leakage-safe BUSI training and evaluation utilities.

All comparison architectures use the exact FiLMoS-Net seed-42 manifest,
mask-assisted channels, preprocessing, augmentation, optimization budget,
checkpoint rule, TTA views, and validation-only calibration.  Only the model
architecture changes between comparison files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("FILMOS_BUSI_ROOT", ROOT / "data" / "Dataset_BUSI_with_GT"))
MANIFEST = ROOT / "data" / "busi_split_seed42.csv"
CLASS_NAMES = ("benign", "malignant", "normal")
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 24
    patience: int = 6
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    class_weight_power: float = 0.65
    label_smoothing: float = 0.02
    seed: int = 42
    num_workers: int = 0
    use_tta: bool = True


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_paths_for_image(path: Path) -> list[Path]:
    """Return every BUSI lesion mask belonging to one independent image."""
    return sorted(path.parent.glob(f"{path.stem}_mask*.png"))


class BUSIComparisonDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int, training: bool):
        self.rows = rows
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def _aligned_augmentation(self, image: Image.Image) -> Image.Image:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if not self.training:
            return image
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
        angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        max_shift = int(round(self.image_size * 0.02))
        translate = [
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item()),
            int(torch.randint(-max_shift, max_shift + 1, (1,)).item()),
        ]
        scale = float(torch.empty(1).uniform_(0.98, 1.02).item())
        return TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0],
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = DATA_ROOT / row["relative_path"]
        with Image.open(path) as opened:
            gray = np.asarray(opened.convert("L"), dtype=np.uint8).copy()
        union = np.zeros_like(gray, dtype=np.uint8)
        for mask_path in mask_paths_for_image(path):
            with Image.open(mask_path) as opened_mask:
                candidate = np.asarray(opened_mask.convert("L"), dtype=np.uint8)
            if candidate.shape != gray.shape:
                candidate = np.asarray(
                    Image.fromarray(candidate).resize(
                        (gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST
                    )
                )
            union = np.maximum(union, (candidate > 0).astype(np.uint8) * 255)
        lesion = np.where(union > 0, gray, 0).astype(np.uint8)
        aligned = Image.merge(
            "RGB",
            (Image.fromarray(gray), Image.fromarray(union), Image.fromarray(lesion)),
        )
        aligned = self._aligned_augmentation(aligned)
        tensor = TF.pil_to_tensor(aligned).float().div_(255.0)
        tensor = (tensor - MEAN) / STD
        return {
            "image": tensor,
            "label": int(row["class_index"]),
            "relative_path": row["relative_path"],
        }


def load_manifest() -> dict[str, list[dict[str, str]]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing fixed split manifest: {MANIFEST}")
    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subset = row["subset"]
            if subset not in splits:
                raise ValueError(f"Unexpected subset {subset!r} in manifest")
            relative_path = row["relative_path"]
            if relative_path in seen:
                raise ValueError(f"Data leakage/duplicate manifest image: {relative_path}")
            if not (DATA_ROOT / relative_path).exists():
                raise FileNotFoundError(DATA_ROOT / relative_path)
            seen.add(relative_path)
            splits[subset].append(row)
    counts = {key: len(value) for key, value in splits.items()}
    if counts != {"train": 544, "validation": 118, "test": 118}:
        raise ValueError(f"Unexpected fixed split counts: {counts}")
    if len(seen) != 780:
        raise ValueError(f"Expected 780 independent images, found {len(seen)}")
    return splits


def make_loaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits = load_manifest()
    generator = torch.Generator().manual_seed(cfg.seed)

    def worker_init(worker_id: int) -> None:
        worker_seed = cfg.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train = DataLoader(
        BUSIComparisonDataset(splits["train"], cfg.image_size, training=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
        generator=generator,
    )
    validation = DataLoader(
        BUSIComparisonDataset(splits["validation"], cfg.image_size, training=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test = DataLoader(
        BUSIComparisonDataset(splits["test"], cfg.image_size, training=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train, validation, test


def decoded_channels(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization and return aligned channels in [0, 1]."""
    mean = MEAN.to(device=x.device, dtype=x.dtype)
    std = STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def class_weights_from_manifest(power: float, device: torch.device) -> torch.Tensor:
    splits = load_manifest()
    counts = np.bincount(
        [int(row["class_index"]) for row in splits["train"]], minlength=3
    ).astype(np.float64)
    weights = np.power(counts, -power)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _specificities(cm: np.ndarray) -> np.ndarray:
    values = []
    total = cm.sum()
    for index in range(len(CLASS_NAMES)):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        tn = total - tp - fp - fn
        values.append(float(tn / (tn + fp)) if tn + fp else 0.0)
    return np.asarray(values)


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, bias: torch.Tensor | None = None
) -> dict[str, object]:
    logits = logits.float().cpu()
    labels_np = labels.long().cpu().numpy()
    if bias is not None:
        logits = logits + bias.float().cpu().view(1, -1)
    probabilities = logits.softmax(dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels_np, predictions, labels=list(range(3)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, predictions, labels=list(range(3)), zero_division=0
    )
    specificity = _specificities(cm)
    aucs: list[float] = []
    aps: list[float] = []
    for index in range(3):
        target = (labels_np == index).astype(np.int64)
        aucs.append(float(roc_auc_score(target, probabilities[:, index])))
        aps.append(float(average_precision_score(target, probabilities[:, index])))
    return {
        "accuracy": float((predictions == labels_np).mean()),
        "cohen_kappa": float(cohen_kappa_score(labels_np, predictions)),
        "macro_f1": float(f1.mean()),
        "macro_sensitivity": float(recall.mean()),
        "macro_specificity": float(specificity.mean()),
        "macro_auroc_ovr": float(np.mean(aucs)),
        "macro_average_precision": float(np.mean(aps)),
        "confusion_matrix": cm.tolist(),
        "classwise": {
            name: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "sensitivity_recall": float(recall[index]),
                "specificity": float(specificity[index]),
                "f1": float(f1[index]),
                "auroc_ovr": aucs[index],
                "average_precision": aps[index],
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        views = [x]
        if use_tta:
            views.extend((torch.flip(x, dims=(-1,)), torch.flip(x, dims=(-2,)), x * 1.03, x * 0.97))
        logits = torch.stack([model(view) for view in views], dim=0).mean(dim=0)
        all_logits.append(logits.cpu())
        all_labels.append(torch.as_tensor(batch["label"]).cpu())
        all_paths.extend(list(batch["relative_path"]))
    return torch.cat(all_logits), torch.cat(all_labels), all_paths


def tune_validation_bias(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    """Tune two additive logits on validation macro-F1; class 0 is the anchor."""
    best_bias = torch.zeros(3)
    best_metrics = compute_metrics(logits, labels, best_bias)
    best_key = (float(best_metrics["macro_f1"]), float(best_metrics["accuracy"]), 0.0)
    grid = torch.linspace(-2.0, 2.0, 41)
    for malignant_bias in grid:
        for normal_bias in grid:
            bias = torch.tensor([0.0, float(malignant_bias), float(normal_bias)])
            metrics = compute_metrics(logits, labels, bias)
            key = (
                float(metrics["macro_f1"]),
                float(metrics["accuracy"]),
                -float(torch.linalg.vector_norm(bias)),
            )
            if key > best_key:
                best_key = key
                best_bias = bias
                best_metrics = metrics
    return best_bias, best_metrics


@torch.no_grad()
def evaluate_unbiased(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    logits, labels, _ = collect_logits(model, loader, device, use_tta=False)
    return compute_metrics(logits, labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[int, list[dict[str, float]]]:
    weights = class_weights_from_manifest(cfg.class_weight_power, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.minimum_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        correct = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = y.numel()
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            correct += int((logits.argmax(dim=1) == y).sum())
        scheduler.step()
        validation_metrics = evaluate_unbiased(model, validation_loader, device)
        score = float(validation_metrics["macro_f1"])
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / sample_count,
            "train_accuracy": correct / sample_count,
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_macro_f1": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_macro_f1": score,
                    "config": asdict(cfg),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return int(checkpoint["epoch"]), history


def write_predictions(
    path: Path,
    relative_paths: list[str],
    labels: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    probabilities = (logits + bias.view(1, -1)).softmax(dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["relative_path", "true_index", "true_class", "predicted_index", "predicted_class"]
            + [f"probability_{name}" for name in CLASS_NAMES]
        )
        for sample_path, label, prediction, probability in zip(
            relative_paths, labels.numpy(), predictions, probabilities
        ):
            writer.writerow(
                [sample_path, label, CLASS_NAMES[label], prediction, CLASS_NAMES[prediction]]
                + [f"{value:.10f}" for value in probability]
            )


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    metrics = report["test"]
    classwise = metrics["classwise"]
    config = report["protocol"]
    lines = [
        f"# {report['display_name']} test report",
        "",
        "## Status",
        "",
        f"This is an actual seed-{config['seed']} run on the untouched fixed BUSI test set. "
        f"Checkpoint epoch {config['best_epoch']} was selected only by validation macro-F1.",
        "",
        f"Implementation scope: {report['implementation_scope']}",
        "",
        "## Common protocol",
        "",
        "- Independent images: 780 (437 benign, 210 malignant, 133 normal)",
        "- Fixed split: 544 train / 118 validation / 118 test, seed 42",
        "- Input: grayscale context, union lesion mask, lesion-only texture; 128 x 128",
        "- Training-only augmentation: horizontal flip, +/-5 degree rotation, 2% translation, 0.98-1.02 scale",
        f"- AdamW; maximum epochs {config['epochs']}; batch size {config['batch_size']}; validation macro-F1 selection",
        f"- TTA enabled: {config['use_tta']}; class-logit bias selected on validation only: {config['validation_bias']}",
        "",
        "## Test metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Cohen's kappa | {metrics['cohen_kappa']:.4f} |",
        f"| Macro-F1 | {metrics['macro_f1']:.4f} |",
        f"| Macro sensitivity | {metrics['macro_sensitivity']:.4f} |",
        f"| Macro specificity | {metrics['macro_specificity']:.4f} |",
        f"| Macro AUROC (OvR) | {metrics['macro_auroc_ovr']:.4f} |",
        f"| Macro AUPRC | {metrics['macro_average_precision']:.4f} |",
        "",
        "## Class-wise test metrics",
        "",
        "| Class | Support | Precision | Recall | Specificity | F1 | AUROC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        item = classwise[name]
        lines.append(
            f"| {name} | {item['support']} | {item['precision']:.4f} | "
            f"{item['sensitivity_recall']:.4f} | {item['specificity']:.4f} | "
            f"{item['f1']:.4f} | {item['auroc_ovr']:.4f} | {item['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are true classes and columns are predicted classes in benign, malignant, normal order.",
            "",
            "```text",
            *[str(row) for row in metrics["confusion_matrix"]],
            "```",
            "",
            "## Audit information",
            "",
            f"- Trainable parameters: {config['trainable_parameters']:,}",
            f"- Training wall time (seconds): {config['training_seconds']:.1f}",
            f"- Test prediction file: `{report['prediction_file']}`",
            f"- Machine-readable report: `{report['json_report_file']}`",
            "",
            "No target performance constraint was used. Test labels were not used for training, checkpoint selection, or calibration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    model_key: str,
    display_name: str,
    implementation_scope: str,
    build_model: Callable[[int], nn.Module],
) -> None:
    parser = argparse.ArgumentParser(description=f"Train and test {display_name} on fixed BUSI split")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One forward/backward batch; no test report")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_tta=not args.no_tta,
    )
    seed_everything(cfg.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_model(3).to(device)
    print(
        json.dumps(
            {
                "model": display_name,
                "device": str(device),
                "trainable_parameters": parameter_count(model),
                "implementation_scope": implementation_scope,
            }
        ),
        flush=True,
    )
    train_loader, validation_loader, test_loader = make_loaders(cfg)
    if args.smoke:
        batch = next(iter(train_loader))
        x = batch["image"].to(device)
        y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        print(json.dumps({"smoke_logits": list(logits.shape), "smoke_loss": float(loss)}))
        return

    output_root = ROOT / "artifacts" / "comparative_runs" / model_key / f"seed_{cfg.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "best_checkpoint.pt"
    started = time.monotonic()
    best_epoch, history = train_model(
        model, train_loader, validation_loader, cfg, device, checkpoint_path
    )
    training_seconds = time.monotonic() - started
    validation_logits, validation_labels, _ = collect_logits(
        model, validation_loader, device, use_tta=cfg.use_tta
    )
    bias, validation_metrics = tune_validation_bias(validation_logits, validation_labels)
    test_logits, test_labels, test_paths = collect_logits(
        model, test_loader, device, use_tta=cfg.use_tta
    )
    test_metrics = compute_metrics(test_logits, test_labels, bias)
    predictions_path = output_root / "test_predictions.csv"
    json_path = output_root / "test_report.json"
    markdown_path = output_root / "test_report.md"
    history_path = output_root / "training_history.json"
    write_predictions(predictions_path, test_paths, test_labels, test_logits, bias)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    report: dict[str, object] = {
        "model_key": model_key,
        "display_name": display_name,
        "implementation_scope": implementation_scope,
        "protocol": {
            **asdict(cfg),
            "best_epoch": best_epoch,
            "device": str(device),
            "trainable_parameters": parameter_count(model),
            "training_seconds": training_seconds,
            "validation_bias": [float(value) for value in bias],
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "test_labels_used_for_selection": False,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "prediction_file": predictions_path.name,
        "json_report_file": json_path.name,
        "training_history_file": str(history_path.relative_to(ROOT)),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(markdown_path, report)
    print(json.dumps({"report": str(markdown_path), "test": test_metrics}, indent=2), flush=True)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
