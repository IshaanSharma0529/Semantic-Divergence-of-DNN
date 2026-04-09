"""
Script: setup_caltech101.py
Purpose: Extract caltech-101.zip to Model Training/caltech101_data/

The zip contains a nested structure:
  caltech-101/101_ObjectCategories.tar.gz  ← actual image data
  caltech-101/Annotations.tar              ← not needed

This script:
  1. Extracts 101_ObjectCategories.tar.gz from the zip in-memory
  2. Writes the class images to:  Model Training/caltech101_data/101_ObjectCategories/
  3. Verifies 101 class directories exist

Dataset: Caltech-101
Output:  Model Training/caltech101_data/101_ObjectCategories/{class_name}/*.jpg

Run from project root:
    python scripts/setup_caltech101.py
"""

import io
import os
import sys
import json
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH     = PROJECT_ROOT / "caltech-101.zip"
OUT_DIR      = PROJECT_ROOT / "Model Training" / "caltech101_data"
TAR_ENTRY    = "caltech-101/101_ObjectCategories.tar.gz"

EXCLUDE_DIRS = {"__MACOSX", ".DS_Store", "BACKGROUND_Google"}


def extract_caltech101():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found.")
        print("Expected the file at the project root.")
        sys.exit(1)

    if (OUT_DIR / "101_ObjectCategories").exists():
        classes = [d for d in (OUT_DIR / "101_ObjectCategories").iterdir()
                   if d.is_dir() and d.name not in EXCLUDE_DIRS]
        if len(classes) >= 101:
            print(f"✓ Already extracted: {len(classes)} classes found at {OUT_DIR}")
            return
        print(f"  Partial extraction found ({len(classes)} classes). Re-extracting...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / 1024**2:.1f} MB)...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # ── Step 1: read the inner tar.gz into memory ──
        if TAR_ENTRY not in zf.namelist():
            # Fallback: search for the tar.gz
            matches = [n for n in zf.namelist() if n.endswith("101_ObjectCategories.tar.gz")]
            if not matches:
                print(f"ERROR: Could not find 101_ObjectCategories.tar.gz inside {ZIP_PATH.name}")
                print(f"  Contents: {zf.namelist()[:10]}")
                sys.exit(1)
            tar_entry = matches[0]
        else:
            tar_entry = TAR_ENTRY

        print(f"  Found: {tar_entry}")
        tar_bytes_size = zf.getinfo(tar_entry).file_size
        print(f"  Extracting tar.gz ({tar_bytes_size / 1024**2:.1f} MB) to memory...")

        tar_bytes = zf.read(tar_entry)

    # ── Step 2: extract tar.gz to OUT_DIR ──
    print(f"  Unpacking to {OUT_DIR} ...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        members = tf.getmembers()
        total   = len(members)

        for i, member in enumerate(members, 1):
            if i % 1000 == 0 or i == total:
                print(f"    {i}/{total} files extracted...", end="\r")

            # Skip __MACOSX and hidden files
            parts = Path(member.name).parts
            if any(p in EXCLUDE_DIRS or p.startswith(".") for p in parts):
                continue

            tf.extract(member, path=OUT_DIR)

    print()  # newline after \r

    # ── Step 3: verify ──
    obj_cat = OUT_DIR / "101_ObjectCategories"
    if not obj_cat.exists():
        print(f"ERROR: Expected {obj_cat} after extraction but not found.")
        sys.exit(1)

    classes = sorted([
        d.name for d in obj_cat.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRS
    ])

    print(f"\n✓ Extraction complete")
    print(f"  Classes: {len(classes)}")
    print(f"  Path:    {obj_cat}")
    print(f"  Sample:  {classes[:5]}...")

    # Count total images
    total_images = sum(
        len(list(p.glob("*.jpg")) + list(p.glob("*.JPG")) + list(p.glob("*.jpeg")))
        for p in obj_cat.iterdir() if p.is_dir()
    )
    print(f"  Images:  {total_images:,}")

    if len(classes) < 101:
        print(f"WARNING: Expected 101 classes, found {len(classes)}")
    else:
        print(f"\nDataset ready. Next: run Model Training/Model_Training.ipynb")


if __name__ == "__main__":
    extract_caltech101()
