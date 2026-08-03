#!/usr/bin/env python3
"""Audit BUSI placement and verify the exact seed-42 split manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_diagnostic_image(path: Path) -> bool:
    """Return True for an image used as a classification sample."""
    return path.suffix.lower() in IMAGE_EXTENSIONS and "_mask" not in path.stem.lower()


def lesion_annotations(path: Path) -> list[Path]:
    """Locate the BUSI lesion-annotation files paired with one image."""
    matches: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        matches.extend(path.parent.glob(f"{path.stem}_mask*{extension}"))
    return sorted(item for item in matches if item.is_file())


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    default_root = Path(os.environ.get("FILMOS_BUSI_ROOT", ROOT / "data" / "Dataset_BUSI_with_GT"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "busi_split_seed42.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    duplicate_paths = len(rows) - len({row["relative_path"] for row in rows})
    subset_counts = Counter((row["subset"], row["class_name"]) for row in rows)
    expected = {
        ("train", "benign"): 305, ("train", "malignant"): 146, ("train", "normal"): 93,
        ("validation", "benign"): 66, ("validation", "malignant"): 32, ("validation", "normal"): 20,
        ("test", "benign"): 66, ("test", "malignant"): 32, ("test", "normal"): 20,
    }

    missing_classes = [name for name in CLASS_NAMES if not (args.data_root / name).is_dir()]
    missing_images: list[str] = []
    missing_annotations: list[str] = []
    found_counts: Counter[str] = Counter()
    if not missing_classes:
        for class_name in CLASS_NAMES:
            found_counts[class_name] = sum(
                1 for path in (args.data_root / class_name).iterdir() if path.is_file() and is_diagnostic_image(path)
            )
        for row in rows:
            image_path = args.data_root / row["relative_path"]
            if not image_path.is_file():
                missing_images.append(row["relative_path"])
            elif not lesion_annotations(image_path):
                missing_annotations.append(row["relative_path"])

    manifest_ok = len(rows) == 780 and duplicate_paths == 0 and dict(subset_counts) == expected
    dataset_ok = (
        not missing_classes
        and dict(found_counts) == {"benign": 437, "malignant": 210, "normal": 133}
        and not missing_images
        and not missing_annotations
    )
    report = {
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_rows": len(rows),
        "duplicate_manifest_paths": duplicate_paths,
        "manifest_counts": {f"{subset}/{name}": count for (subset, name), count in sorted(subset_counts.items())},
        "diagnostic_image_counts": dict(found_counts),
        "missing_class_directories": missing_classes,
        "missing_images": missing_images,
        "images_without_lesion_annotation": missing_annotations,
        "manifest_ok": manifest_ok,
        "dataset_ok": dataset_ok,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if manifest_ok and dataset_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
