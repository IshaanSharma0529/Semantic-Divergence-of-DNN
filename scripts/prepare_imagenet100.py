"""
ImageNet-100 Dataset Preparation
=================================
Select 100 classes from ImageNet-1K and create the directory structure
needed for training.

Prerequisites:
  - Download ImageNet-1K (ILSVRC2012) to a local directory
  - Provide the path via --imagenet-root argument

This script:
  1. Randomly selects 100 classes (with fixed seed for reproducibility)
  2. Creates the directory structure under ImageNet100_Training/data/
  3. Symlinks or copies images (configurable)
  4. Writes class_names.txt and class_mapping.json
  5. Creates train/val/test splits per class

Run from project root:
    python scripts/prepare_imagenet100.py --imagenet-root /path/to/ILSVRC2012
"""

import os, sys, json, shutil, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
NUM_CLASSES = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'ImageNet100_Training'
DATA_DIR = OUT_DIR / 'data'


def load_imagenet_synset_mapping(imagenet_root):
    """Try to load synset-to-human mapping from ImageNet directory."""
    mapping_files = [
        imagenet_root / 'LOC_synset_mapping.txt',
        imagenet_root / 'synset_words.txt',
        imagenet_root / 'map_clsloc.txt',
    ]
    synset_to_name = {}
    for mf in mapping_files:
        if mf.exists():
            with open(mf, 'r') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        synset_to_name[parts[0]] = parts[1].split(',')[0].strip()
            break
    return synset_to_name


def select_classes(train_dir, n_classes=NUM_CLASSES, seed=SEED,
                   min_images=50):
    """Select N classes from ImageNet with at least min_images samples."""
    rng = np.random.RandomState(seed)
    all_dirs = sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and len(list(d.iterdir())) >= min_images
    ])
    print(f"  Found {len(all_dirs)} classes with >= {min_images} images")

    selected_idx = rng.choice(len(all_dirs), size=n_classes, replace=False)
    selected_idx.sort()
    selected = [all_dirs[i] for i in selected_idx]
    return selected


def create_splits(class_dir, train_frac=TRAIN_SPLIT, val_frac=VAL_SPLIT,
                  seed=SEED):
    """Create train/val/test splits for a single class."""
    rng = np.random.RandomState(seed)
    images = sorted([
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ['.jpeg', '.jpg', '.png', '.JPEG']
    ])
    rng.shuffle(images)

    n = len(images)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    return {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ImageNet-100 dataset subset')
    parser.add_argument('--imagenet-root', type=str, required=True,
                        help='Path to ILSVRC2012 root directory')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symlinks')
    parser.add_argument('--max-per-class', type=int, default=None,
                        help='Maximum images per class (for quick testing)')
    args = parser.parse_args()

    imagenet_root = Path(args.imagenet_root)
    train_dir = imagenet_root / 'train'

    if not train_dir.exists():
        # Check for ILSVRC alternative structure
        alt = imagenet_root / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'train'
        if alt.exists():
            train_dir = alt
        else:
            print(f"  ERROR: Could not find train directory at {train_dir}")
            print(f"  Please provide path containing 'train/' subdirectory")
            return

    print("=" * 60)
    print("ImageNet-100 Dataset Preparation")
    print("=" * 60)

    # ── 1. Select classes ──
    print(f"\n[1/4] Selecting {NUM_CLASSES} classes (seed={SEED})...")
    selected_synsets = select_classes(train_dir)
    print(f"  Selected: {selected_synsets[:5]} ... (total {len(selected_synsets)})")

    # ── 2. Load synset names ──
    synset_map = load_imagenet_synset_mapping(imagenet_root)
    class_mapping = {}
    for idx, synset in enumerate(selected_synsets):
        human_name = synset_map.get(synset, synset)
        class_mapping[synset] = {
            'index': idx,
            'human_name': human_name,
        }

    # ── 3. Create directory structure and copy/link ──
    print(f"\n[2/4] Creating directory structure at {DATA_DIR}...")
    stats = {'train': 0, 'val': 0, 'test': 0}

    for synset in selected_synsets:
        src_dir = train_dir / synset
        splits = create_splits(src_dir)

        for split_name, images in splits.items():
            if args.max_per_class:
                images = images[:args.max_per_class]

            dst_dir = DATA_DIR / split_name / synset
            dst_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dst_path = dst_dir / img_path.name
                if not dst_path.exists():
                    if args.copy:
                        shutil.copy2(img_path, dst_path)
                    else:
                        try:
                            os.symlink(img_path, dst_path)
                        except OSError:
                            shutil.copy2(img_path, dst_path)
                stats[split_name] += 1

    print(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")

    # ── 4. Write metadata ──
    print(f"\n[3/4] Writing metadata files...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # class_names.txt
    class_names = [synset_map.get(s, s) for s in selected_synsets]
    with open(OUT_DIR / 'class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    # class_mapping.json (synset → index + human name)
    with open(OUT_DIR / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)

    # selected_synsets.txt
    with open(OUT_DIR / 'selected_synsets.txt', 'w') as f:
        for s in selected_synsets:
            f.write(s + '\n')

    # dataset_info.json
    info = {
        'dataset': 'ImageNet-100',
        'source': 'ILSVRC2012 subset',
        'num_classes': NUM_CLASSES,
        'seed': SEED,
        'splits': stats,
        'img_size': [224, 224],
        'synsets': selected_synsets,
    }
    with open(OUT_DIR / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  class_names.txt: {len(class_names)} classes")
    print(f"  class_mapping.json: synset → index mapping")

    # ── 5. Print checklist ──
    print(f"\n[4/4] Next steps:")
    print(f"  1. Verify data at {DATA_DIR}")
    print(f"  2. Train models using configs/imagenet100_training_config.yaml")
    print(f"  3. Run attacks using the same experiment scripts")
    print(f"     (update shared_utils.py DATA_DIR to point to ImageNet-100)")

    print("\n" + "=" * 60)
    print(f"  ImageNet-100 prepared: {NUM_CLASSES} classes, {sum(stats.values())} images")
    print("=" * 60)


if __name__ == '__main__':
    main()
