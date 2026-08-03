#!/usr/bin/env python3
"""Regenerate manuscript Figs. 8 and 9 from per-sample probabilities."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CLASS_NAMES = ("benign", "malignant", "normal")
COLORS = ("#2664A8", "#C23B3B", "#29845A")


def load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    with (RESULTS / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([class_to_index[row["true_class"]] for row in rows])
    predictions = np.asarray([class_to_index[row["predicted_class"]] for row in rows])
    probabilities = np.asarray(
        [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows], dtype=float
    )
    return labels, predictions, probabilities


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def figure_8(labels: np.ndarray, probabilities: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for index, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        binary = (labels == index).astype(int)
        false_positive, true_positive, _ = roc_curve(binary, probabilities[:, index])
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, index])
        auroc = roc_auc_score(binary, probabilities[:, index])
        average_precision = average_precision_score(binary, probabilities[:, index])
        axes[0].plot(false_positive, true_positive, color=color, lw=2, label=f"{name.title()} (AUROC={auroc:.4f})")
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{name.title()} (AP={average_precision:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.55", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    save_figure(figure, "figure_8_roc_pr_curves")


def figure_9(labels: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(3))
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ConfusionMatrixDisplay(matrix, display_labels=[name.title() for name in CLASS_NAMES]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    ConfusionMatrixDisplay(normalized, display_labels=[name.title() for name in CLASS_NAMES]).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".3f"
    )
    axes[0].set_title("Counts")
    axes[1].set_title("Row-normalized")
    figure.tight_layout()
    save_figure(figure, "figure_9_confusion_matrices")


def main() -> None:
    labels, predictions, probabilities = load_predictions()
    if len(labels) != 118:
        raise ValueError(f"Expected 118 test predictions, found {len(labels)}")
    figure_8(labels, probabilities)
    figure_9(labels, predictions)
    print(f"Regenerated Figs. 8 and 9 in {FIGURES}")


if __name__ == "__main__":
    main()
