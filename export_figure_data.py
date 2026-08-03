#!/usr/bin/env python3
"""Export numerical confusion-matrix and ROC/PR coordinates without drawing figures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CLASS_NAMES = ["benign", "malignant", "normal"]


def write_confusion(labels: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(CLASS_NAMES)))
    with (RESULTS / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", *[f"pred_{name}" for name in CLASS_NAMES]])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])


def write_roc(labels: np.ndarray, probabilities: np.ndarray) -> None:
    with (RESULTS / "roc_curve_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "threshold", "false_positive_rate", "true_positive_rate"])
        for index, class_name in enumerate(CLASS_NAMES):
            binary_labels = (labels == index).astype(np.int64)
            fpr, tpr, thresholds = roc_curve(binary_labels, probabilities[:, index])
            for threshold, x, y in zip(thresholds, fpr, tpr):
                threshold_text = "inf" if np.isinf(threshold) else f"{threshold:.10f}"
                writer.writerow([class_name, threshold_text, f"{x:.10f}", f"{y:.10f}"])


def write_pr(labels: np.ndarray, probabilities: np.ndarray) -> None:
    with (RESULTS / "pr_curve_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "threshold", "recall", "precision"])
        for index, class_name in enumerate(CLASS_NAMES):
            binary_labels = (labels == index).astype(np.int64)
            precision, recall, thresholds = precision_recall_curve(binary_labels, probabilities[:, index])
            for point_index, (x, y) in enumerate(zip(recall, precision)):
                threshold = "" if point_index == len(thresholds) else f"{thresholds[point_index]:.10f}"
                writer.writerow([class_name, threshold, f"{x:.10f}", f"{y:.10f}"])


def main() -> None:
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    with (RESULTS / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([class_to_index[row["true_class"]] for row in rows], dtype=np.int64)
    predictions = np.asarray([class_to_index[row["predicted_class"]] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[f"prob_{class_name}"]) for class_name in CLASS_NAMES] for row in rows], dtype=np.float64
    )
    write_confusion(labels, predictions)
    write_roc(labels, probabilities)
    write_pr(labels, probabilities)
    print(f"Exported numerical plot data for {len(rows)} test predictions to {RESULTS}")


if __name__ == "__main__":
    main()
