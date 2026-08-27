"""
Splits a raw, class-per-folder image dataset into train/val/test folders.

Expected input layout (data/raw/):
    data/raw/
    ├── cloudy/
    ├── rain/
    ├── shine/
    └── sunrise/

Produces:
    data/dataset/train/<class>/...
    data/dataset/val/<class>/...
    data/dataset/test/<class>/...

Usage:
    python utils/prepare_dataset.py
    python utils/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import os
import random
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402


def split_dataset(raw_dir, out_dir, train_ratio, val_ratio, test_ratio, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train + val + test ratios must sum to 1.0"
    )

    random.seed(seed)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Place your class-named folders of images inside data/raw/ first."
        )

    class_folders = [
        d for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))
    ]

    if not class_folders:
        raise RuntimeError(f"No class folders found inside {raw_dir}")

    print(f"Found classes: {class_folders}")

    for split in ["train", "val", "test"]:
        for cls in class_folders:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    summary = {}

    for cls in class_folders:
        cls_dir = os.path.join(raw_dir, cls)
        images = [
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = images[:n_train]
        val_files = images[n_train : n_train + n_val]
        test_files = images[n_train + n_val :]

        for fname in train_files:
            shutil.copy2(os.path.join(cls_dir, fname), os.path.join(out_dir, "train", cls, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(cls_dir, fname), os.path.join(out_dir, "val", cls, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(cls_dir, fname), os.path.join(out_dir, "test", cls, fname))

        summary[cls] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    print("\nDataset split summary:")
    for cls, counts in summary.items():
        print(
            f"  {cls:<10} total={counts['total']:<5} "
            f"train={counts['train']:<5} val={counts['val']:<5} test={counts['test']:<5}"
        )

    print(f"\nDone. Split dataset saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=config.RAW_DATA_DIR)
    parser.add_argument("--out", type=str, default=config.DATASET_DIR)
    parser.add_argument("--train", type=float, default=config.TRAIN_SPLIT)
    parser.add_argument("--val", type=float, default=config.VAL_SPLIT)
    parser.add_argument("--test", type=float, default=config.TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    split_dataset(args.raw, args.out, args.train, args.val, args.test, args.seed)
