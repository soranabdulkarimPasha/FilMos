#!/usr/bin/env python3
"""Generate manuscript ROC/PR and confusion-matrix figures from one CSV."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc

CLASSES = ("benign", "malignant", "normal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions).rename(columns={
        "prob_benign": "probability_benign", "prob_malignant": "probability_malignant", "prob_normal": "probability_normal"
    })
    y = frame["true_class"].map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    pred = frame["predicted_class"].map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    prob = frame[[f"probability_{name}" for name in CLASSES]].to_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 5.5), constrained_layout=True)
    colors = ("#1665AC", "#C23B22", "#2A8C55")
    for index, (name, color) in enumerate(zip(CLASSES, colors)):
        binary = (y == index).astype(int)
        fpr, tpr, _ = roc_curve(binary, prob[:, index])
        precision, recall, _ = precision_recall_curve(binary, prob[:, index])
        axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{name.title()} (AUC={auc(fpr,tpr):.4f})")
        axes[1].plot(recall, precision, lw=2, color=color, label=f"{name.title()} (AP={average_precision_score(binary,prob[:,index]):.4f})")
    axes[0].plot([0,1],[0,1],"--",color="0.5",lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves", xlim=(0,1), ylim=(0,1.01))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision–recall curves", xlim=(0,1), ylim=(0,1.01))
    for ax in axes: ax.legend(loc="lower left", fontsize=8)
    for suffix in ("png", "pdf"): fig.savefig(args.output_dir / f"figure_8_cold_start_roc_pr.{suffix}", dpi=300)
    plt.close(fig)

    cm = confusion_matrix(y, pred, labels=[0,1,2])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 5.2), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0], xticklabels=[x.title() for x in CLASSES], yticklabels=[x.title() for x in CLASSES])
    sns.heatmap(norm, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axes[1], xticklabels=[x.title() for x in CLASSES], yticklabels=[x.title() for x in CLASSES])
    axes[0].set_title("Counts"); axes[1].set_title("Row-normalized")
    for ax in axes: ax.set(xlabel="Predicted class", ylabel="True class")
    for suffix in ("png", "pdf"): fig.savefig(args.output_dir / f"figure_9_cold_start_confusion.{suffix}", dpi=300)
    plt.close(fig)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(args.output_dir / "figure_9_confusion_matrix.csv")


if __name__ == "__main__":
    main()
