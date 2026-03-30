"""
Task 5 — Multi-Seed Statistical Validation Runner
==================================================
Re-run the core FGSM and PGD evaluations under 3 different random seeds
to compute confidence intervals and statistical significance tests.

Seeds: [42, 123, 456]

For each seed:
  1. Re-shuffle the dataset using the seed (re-split train/val/test)
  2. Run FGSM at all epsilons for all 3 models
  3. Run PGD at all epsilons for all 3 models
  4. Record accuracy, fooling rate, confidence metrics

After all seeds:
  - Bootstrap 95% CIs on accuracy and fooling rate
  - Wilcoxon signed-rank test: VGG19 vs ResNet50 vs DenseNet121

Writes to:
    Model Training/fgsm_results/multi_seed/

Run from project root:
    python scripts/multi_seed_runner.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    find_image_root, build_raw_test_dataset, load_models,
    build_all_logits_models, preprocess_for_model,
    fgsm_attack, pgd_attack,
    MODEL_NAMES, EPSILONS, ADV_EPSILONS, PGD_STEPS, SEED, IMG_SIZE, BATCH_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, NUM_CLASSES,
    DATA_DIR, RESULTS_DIR,
)

OUT_DIR = RESULTS_DIR / 'multi_seed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]


# ────────────────────────────────────────────────────────────────
# Build test split for a given seed (re-shuffle)
# ────────────────────────────────────────────────────────────────
def build_test_split_for_seed(seed):
    """Build test split using a specific seed for shuffling."""
    image_root = find_image_root(DATA_DIR)
    exclude = {'__MACOSX', '.DS_Store', 'BACKGROUND_Google', '__pycache__'}
    class_names = sorted([
        d.name for d in image_root.iterdir()
        if d.is_dir() and d.name not in exclude and not d.name.startswith('.')
    ])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_paths, all_labels = [], []
    for class_name in class_names:
        class_dir = image_root / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[class_name])
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_paths))
    all_paths = all_paths[indices]
    all_labels = all_labels[indices]

    train_size = int(TRAIN_SPLIT * len(all_paths))
    val_size = int(VAL_SPLIT * len(all_paths))
    test_paths = all_paths[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    return test_paths, test_labels, class_names


# ────────────────────────────────────────────────────────────────
# Evaluate under attack (per-sample results for Wilcoxon)
# ────────────────────────────────────────────────────────────────
def evaluate_fgsm_per_sample(model, dataset, epsilon, model_name):
    """FGSM evaluation returning per-sample correctness."""
    per_sample = []
    for images, labels in dataset:
        labels_int = tf.cast(labels, tf.int32)
        bs = tf.shape(images)[0].numpy()

        if epsilon > 0:
            adv_images = fgsm_attack(model, images, labels, epsilon, model_name)
        else:
            adv_images = images

        adv_pre = preprocess_for_model(adv_images, model_name)
        preds = tf.cast(model(adv_pre, training=False), tf.float32)
        adv_classes = tf.argmax(preds, axis=1, output_type=tf.int32)
        correct = tf.cast(tf.equal(adv_classes, labels_int), tf.int32).numpy()
        per_sample.extend(correct.tolist())

    return np.array(per_sample)


def evaluate_pgd_per_sample(logits_model, model, dataset, epsilon,
                             steps, step_size, model_name):
    """PGD evaluation returning per-sample correctness."""
    per_sample = []
    for images, labels in dataset:
        labels_int = tf.cast(labels, tf.int32)
        bs = tf.shape(images)[0].numpy()

        adv_images = pgd_attack(logits_model, images, labels, epsilon,
                                steps, step_size, model_name)
        adv_pre = preprocess_for_model(adv_images, model_name)
        preds = tf.cast(model(adv_pre, training=False), tf.float32)
        adv_classes = tf.argmax(preds, axis=1, output_type=tf.int32)
        correct = tf.cast(tf.equal(adv_classes, labels_int), tf.int32).numpy()
        per_sample.extend(correct.tolist())

    return np.array(per_sample)


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TASK 5: Multi-Seed Statistical Validation")
    print("=" * 60)

    print("\n[1/3] Loading models (shared across seeds)...")
    models = load_models()
    logits_models = build_all_logits_models(models)

    all_seed_results = {}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*48}")
        print(f"  SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'─'*48}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        test_paths, test_labels, class_names = build_test_split_for_seed(seed)
        raw_test_ds = build_raw_test_dataset(test_paths, test_labels)
        print(f"  Test samples: {len(test_labels)}")

        seed_results = {}

        # ── FGSM ──
        print(f"\n  FGSM evaluation:")
        for name in MODEL_NAMES:
            for eps in EPSILONS:
                t0 = time.time()
                per_sample = evaluate_fgsm_per_sample(
                    models[name], raw_test_ds, eps, name)
                acc = float(np.mean(per_sample))
                key = f"{name}_fgsm_eps{eps}"
                seed_results[key] = {
                    'accuracy': round(acc, 6),
                    'per_sample': per_sample.tolist(),
                }
                print(f"    {name} ε={eps}: acc={acc:.4f} ({time.time()-t0:.1f}s)")

        # ── PGD ──
        print(f"\n  PGD evaluation:")
        for name in MODEL_NAMES:
            for eps in ADV_EPSILONS:
                step_size = eps / 4.0
                t0 = time.time()
                per_sample = evaluate_pgd_per_sample(
                    logits_models[name], models[name], raw_test_ds,
                    eps, PGD_STEPS, step_size, name)
                acc = float(np.mean(per_sample))
                key = f"{name}_pgd_eps{eps}"
                seed_results[key] = {
                    'accuracy': round(acc, 6),
                    'per_sample': per_sample.tolist(),
                }
                print(f"    {name} ε={eps}: acc={acc:.4f} ({time.time()-t0:.1f}s)")

        all_seed_results[str(seed)] = seed_results

    # ── Save raw per-seed results ──
    print("\n[2/3] Saving raw results...")
    # Save without per_sample (too large) in summary
    summary = {}
    for seed_str, seed_res in all_seed_results.items():
        summary[seed_str] = {
            k: {'accuracy': v['accuracy']}
            for k, v in seed_res.items()
        }
    with open(OUT_DIR / 'multi_seed_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Save full per-sample for statistical tests ──
    with open(OUT_DIR / 'multi_seed_per_sample.json', 'w') as f:
        json.dump(all_seed_results, f)

    print(f"  Saved to {OUT_DIR}")
    print(f"\n[3/3] Run 'python scripts/aggregate_results.py' to compute CIs and tests.")

    # ── Quick summary table ──
    rows = []
    for seed_str in [str(s) for s in SEEDS]:
        for name in MODEL_NAMES:
            for attack in ['fgsm', 'pgd']:
                eps_list = EPSILONS if attack == 'fgsm' else ADV_EPSILONS
                for eps in eps_list:
                    key = f"{name}_{attack}_eps{eps}"
                    if key in all_seed_results[seed_str]:
                        rows.append({
                            'Seed': int(seed_str),
                            'Model': name,
                            'Attack': attack,
                            'Epsilon': eps,
                            'Accuracy': all_seed_results[seed_str][key]['accuracy'],
                        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'multi_seed_accuracy_table.csv', index=False)
    print(f"\n  Summary: {len(rows)} evaluation runs across {len(SEEDS)} seeds")

    print("\n" + "=" * 60)
    print("  MULTI-SEED RUNNER COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
