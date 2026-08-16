#!/usr/bin/env python3
"""Run all five comparison experiments sequentially and build a summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "manuscript"
MODELS = ("ctmf_net", "sae_net", "dsdnet", "vit", "hybrid_cnn_lstm")


def build_summary(seed: int) -> None:
    rows = []
    for model in MODELS:
        report_path = ROOT / "artifacts" / "comparative_runs" / model / f"seed_{seed}" / "test_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report["test"]
        rows.append(
            {
                "model": report["display_name"],
                "accuracy": metrics["accuracy"],
                "kappa": metrics["cohen_kappa"],
                "macro_f1": metrics["macro_f1"],
                "sensitivity": metrics["macro_sensitivity"],
                "specificity": metrics["macro_specificity"],
                "macro_auroc": metrics["macro_auroc_ovr"],
                "parameters": report["protocol"]["trainable_parameters"],
            }
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"baseline_training_summary_seed_{seed}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Comparison summary",
        "",
        f"The five rows are independently trained baseline runs for seed {seed}. No target score was imposed.",
        "",
        "| Model | ACC | Kappa | Macro-F1 | Sens | Spec | AUROC | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['kappa']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['macro_auroc']:.4f} | {row['parameters']:,} |"
        )
    (RESULTS / f"baseline_training_summary_seed_{seed}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    for model in MODELS:
        command = [
            sys.executable,
            str(HERE / f"{model}.py"),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.no_tta:
            command.append("--no-tta")
        print(f"Running {model}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        build_summary(args.seed)


if __name__ == "__main__":
    main()
