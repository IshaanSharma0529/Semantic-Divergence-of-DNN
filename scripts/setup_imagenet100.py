"""
Script: setup_imagenet100.py
Purpose: Extract the Kaggle-format ImageNet-100 archive.zip and build the
         directory structure expected by VIT/src/swin_utils.py.

Input zip structure (Kaggle ImageNet-100):
    Labels.json              ← synset_id → human name (100 classes)
    train.X1/{synset}/*.JPEG ← 25 synsets (shard 1)
    train.X2/{synset}/*.JPEG ← 25 synsets (shard 2)
    train.X3/{synset}/*.JPEG ← 25 synsets (shard 3)
    train.X4/{synset}/*.JPEG ← 25 synsets (shard 4)
    val.X/{synset}/*.JPEG    ← 100 synsets

Output structure (ImageNet100_Training/data/):
    train/{synset}/*.JPEG    ← merged from all 4 train shards
    val/{synset}/*.JPEG      ← 80% of original val.X
    test/{synset}/*.JPEG     ← 20% of original val.X

Also writes:
    ImageNet100_Training/class_names.txt
    ImageNet100_Training/class_mapping.json
    ImageNet100_Training/dataset_info.json

Run from project root:
    python scripts/setup_imagenet100.py [--zip path/to/archive.zip]
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

SEED        = 42
VAL_FRAC    = 0.8   # fraction of val.X that goes to val/  (rest → test/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP  = PROJECT_ROOT / "archive.zip"
OUT_ROOT     = PROJECT_ROOT / "ImageNet100_Training"
DATA_DIR     = OUT_ROOT / "data"

TRAIN_SHARDS = ["train.X1", "train.X2", "train.X3", "train.X4"]
VAL_SHARD    = "val.X"


def load_labels(zf: zipfile.ZipFile) -> dict:
    """Read Labels.json from zip → {synset_id: human_name}."""
    with zf.open("Labels.json") as f:
        return json.load(f)


def collect_entries(zf: zipfile.ZipFile) -> tuple[dict, dict]:
    """
    Returns:
        train_entries: {synset: [zip_path, ...]}   (merged from all shards)
        val_entries:   {synset: [zip_path, ...]}
    """
    train_entries: dict[str, list[str]] = {}
    val_entries:   dict[str, list[str]] = {}

    for entry in zf.infolist():
        name = entry.filename
        if entry.is_dir():
            continue

        parts = name.split("/")
        if len(parts) < 3:
            continue   # Labels.json or top-level file

        shard, synset, filename = parts[0], parts[1], parts[2]

        if shard in TRAIN_SHARDS:
            train_entries.setdefault(synset, []).append(name)
        elif shard == VAL_SHARD:
            val_entries.setdefault(synset, []).append(name)

    return train_entries, val_entries


def extract_file(zf: zipfile.ZipFile, zip_path: str, out_path: Path):
    """Extract a single file from zip to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(zip_path) as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def run(zip_path: Path):
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found.")
        print("  Provide path with --zip or place archive.zip in the project root.")
        sys.exit(1)

    # ── check if already done ──
    if DATA_DIR.exists():
        splits = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        if {"train", "val", "test"}.issubset(set(splits)):
            synsets = [d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()]
            if len(synsets) >= 100:
                print(f"✓ Already set up: {len(synsets)} synsets in {DATA_DIR}")
                return
        print("  Partial setup found. Rebuilding...")

    print(f"Opening {zip_path.name} ({zip_path.stat().st_size / 1024**3:.1f} GB)...")
    rng = random.Random(SEED)

    with zipfile.ZipFile(zip_path, "r") as zf:

        # ── Labels ──
        print("  Loading Labels.json...")
        labels = load_labels(zf)
        print(f"  Classes: {len(labels)}")

        # ── Collect entries ──
        print("  Scanning zip entries (may take a moment)...")
        train_entries, val_entries = collect_entries(zf)
        print(f"  Train synsets: {len(train_entries)}, Val synsets: {len(val_entries)}")

        all_synsets = sorted(set(list(train_entries.keys()) + list(val_entries.keys())))

        # ── Extract train shards → data/train/ ──
        total_train = sum(len(v) for v in train_entries.values())
        print(f"\n[1/3] Extracting {total_train:,} training images...")
        done = 0
        for synset, entries in sorted(train_entries.items()):
            for zip_path_entry in entries:
                filename = zip_path_entry.split("/")[-1]
                out_path = DATA_DIR / "train" / synset / filename
                if not out_path.exists():
                    extract_file(zf, zip_path_entry, out_path)
                done += 1
                if done % 5000 == 0:
                    print(f"    {done:,}/{total_train:,} training images...", end="\r")
        print(f"    {total_train:,}/{total_train:,} training images done.    ")

        # ── Split val.X → data/val/ + data/test/ ──
        total_val = sum(len(v) for v in val_entries.values())
        print(f"\n[2/3] Splitting {total_val:,} val images → val/ + test/...")
        done = 0
        val_count = test_count = 0

        for synset, entries in sorted(val_entries.items()):
            shuffled = list(entries)
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * VAL_FRAC))

            for i, zip_path_entry in enumerate(shuffled):
                filename   = zip_path_entry.split("/")[-1]
                split_name = "val" if i < split_idx else "test"
                out_path   = DATA_DIR / split_name / synset / filename
                if not out_path.exists():
                    extract_file(zf, zip_path_entry, out_path)
                if split_name == "val":
                    val_count += 1
                else:
                    test_count += 1
                done += 1
                if done % 1000 == 0:
                    print(f"    {done:,}/{total_val:,} val images...", end="\r")

        print(f"    val: {val_count:,} | test: {test_count:,}           ")

    # ── Write metadata ──
    print("\n[3/3] Writing metadata...")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # class_names.txt (sorted by synset)
    sorted_synsets = sorted(all_synsets)
    class_names    = [labels.get(s, s) for s in sorted_synsets]

    with open(OUT_ROOT / "class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # class_mapping.json: synset → {index, human_name}
    class_mapping = {
        synset: {"index": idx, "human_name": labels.get(synset, synset)}
        for idx, synset in enumerate(sorted_synsets)
    }
    with open(OUT_ROOT / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    # dataset_info.json
    dataset_info = {
        "dataset":     "ImageNet-100 (Kaggle subset)",
        "num_classes": len(all_synsets),
        "seed":        SEED,
        "val_frac":    VAL_FRAC,
        "splits": {
            "train": total_train,
            "val":   val_count,
            "test":  test_count,
        },
        "synsets":     sorted_synsets,
    }
    with open(OUT_ROOT / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ImageNet-100 ready at: {DATA_DIR}")
    print(f"  Classes  : {len(all_synsets)}")
    print(f"  Train    : {total_train:,} images")
    print(f"  Val      : {val_count:,} images")
    print(f"  Test     : {test_count:,} images")
    print(f"{'='*60}")
    print("\nNext: run VIT/train_swin.py to fine-tune Swin-Base on ImageNet-100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Kaggle ImageNet-100 dataset")
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP,
        help=f"Path to archive.zip (default: {DEFAULT_ZIP})"
    )
    args = parser.parse_args()
    run(args.zip)
