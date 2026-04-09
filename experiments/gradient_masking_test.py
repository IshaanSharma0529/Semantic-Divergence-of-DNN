"""
Task 1 — Gradient Masking Diagnostic
=====================================
Purpose: determine whether the DenseNet121 FGSM non-monotonicity
(accuracy increases from 16.57 % at ε=0.02 to 20.02 % at ε=0.04)
is caused by obfuscated/masked gradients.

Three sub-tests, each produces a binary verdict:
  (a) Gradient-norm histogram:   compare ‖∇_x L‖ distributions across
      models; near-zero norms → masked gradients.
  (b) Restart PGD superiority:   if PGD with N random restarts beats
      single-restart PGD by > 5 pp at same ε, gradients are unreliable.
  (c) Loss-landscape monotonicity: accuracy MUST decrease as ε grows
      for a rigorous attack.  If it does not, report non-monotonicity.

Writes JSON verdict + CSV of per-model metrics to:
    Model Training/fgsm_results/gradient_masking/
Generates figures to:
    Model Training/fgsm_results/gradient_masking/figures/

Run from project root:
    python experiments/gradient_masking_test.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    load_test_split, build_raw_test_dataset, load_models,
    build_all_logits_models, preprocess_for_model,
    fgsm_attack, pgd_attack,
    MODEL_NAMES, ADV_EPSILONS, PGD_STEPS, SEED,
    seed_everything, RESULTS_DIR
)

OUT_DIR = RESULTS_DIR / 'gradient_masking'
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_RESTARTS = 5          # restart-PGD random starts
EPS_FOR_HIST = 0.01     # epsilon used for gradient-norm histogram
SUPERIORITY_THRESHOLD = 0.05   # 5 pp drop → diagnosis = masked


# ────────────────────────────────────────────────────────────────
# (a) Gradient-norm histogram
# ────────────────────────────────────────────────────────────────
def compute_gradient_norms(model, dataset, model_name, epsilon=EPS_FOR_HIST):
    """Compute per-sample ‖∇_x L‖ on raw [0,1] images."""
    norms = []
    for images, labels in dataset:
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(images)
            preprocessed = preprocess_for_model(images, model_name)
            preds = model(preprocessed, training=False)
            preds = tf.cast(preds, tf.float32)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
        grads = tape.gradient(loss, images)
        per_sample = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        norms.extend(per_sample.numpy().tolist())
    return np.array(norms)


def gradient_norm_analysis(models, dataset):
    """Compute grad-norm stats for all models; plot histogram."""
    records = {}
    grad_data = {}
    for name in MODEL_NAMES:
        norms = compute_gradient_norms(models[name], dataset, name)
        grad_data[name] = norms
        near_zero_frac = float(np.mean(norms < 1e-6))
        records[name] = {
            'mean_grad_norm': float(np.mean(norms)),
            'median_grad_norm': float(np.median(norms)),
            'std_grad_norm': float(np.std(norms)),
            'near_zero_fraction': near_zero_frac,
            'min': float(np.min(norms)),
            'max': float(np.max(norms)),
            'diagnosis': 'masked' if near_zero_frac > 0.10 else 'ok',
        }
        print(f"  {name}: mean ‖∇‖ = {records[name]['mean_grad_norm']:.6f}, "
              f"near-zero = {near_zero_frac:.2%}")

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, name in enumerate(MODEL_NAMES):
        axes[i].hist(grad_data[name], bins=50, alpha=0.8, edgecolor='black')
        axes[i].axvline(np.median(grad_data[name]), color='red', ls='--',
                        label=f'median={np.median(grad_data[name]):.4f}')
        axes[i].set_title(name)
        axes[i].set_xlabel('‖∇_x L‖')
        axes[i].set_ylabel('Count')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'gradient_norm_histogram.png', dpi=150)
    plt.close()
    print(f"  Saved gradient_norm_histogram.png")
    return records


# ────────────────────────────────────────────────────────────────
# (b) Restart PGD superiority
# ────────────────────────────────────────────────────────────────
def restart_pgd(logits_model, model, images, labels, epsilon,
                steps, step_size, model_name, n_restarts):
    """Run PGD with N random restarts; keep worst-case (lowest accuracy) images."""
    best_adv = None
    best_correct = images.shape[0] + 1  # start higher than possible

    for _ in range(n_restarts):
        adv_images = pgd_attack(logits_model, images, labels, epsilon,
                                steps, step_size, model_name)
        adv_pre = preprocess_for_model(adv_images, model_name)
        preds = tf.cast(model(adv_pre, training=False), tf.float32)
        adv_classes = tf.argmax(preds, axis=1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(adv_classes, tf.cast(labels, tf.int32)), tf.int32)).numpy()

        if correct < best_correct:
            best_correct = correct
            best_adv = adv_images

    return best_adv, best_correct


def restart_pgd_analysis(models, logits_models, dataset):
    """Compare single-restart PGD vs N-restart PGD."""
    records = {}
    for name in MODEL_NAMES:
        model = models[name]
        logits_model = logits_models[name]
        row = {}
        for eps in ADV_EPSILONS:
            step_size = eps / 4.0
            total = 0
            single_correct = 0
            restart_correct = 0

            for images, labels in dataset:
                bs = tf.shape(images)[0].numpy()
                labels_int = tf.cast(labels, tf.int32)

                # Single-restart PGD
                adv_single = pgd_attack(logits_model, images, labels, eps,
                                        PGD_STEPS, step_size, name)
                pre_s = preprocess_for_model(adv_single, name)
                preds_s = tf.cast(model(pre_s, training=False), tf.float32)
                cls_s = tf.argmax(preds_s, axis=1, output_type=tf.int32)
                single_correct += tf.reduce_sum(
                    tf.cast(tf.equal(cls_s, labels_int), tf.int32)).numpy()

                # Restart PGD
                _, worst_correct = restart_pgd(
                    logits_model, model, images, labels, eps,
                    PGD_STEPS, step_size, name, N_RESTARTS)
                restart_correct += worst_correct
                total += bs

            single_acc = single_correct / total
            restart_acc = restart_correct / total
            gap = single_acc - restart_acc
            row[f'eps_{eps}'] = {
                'single_restart_acc': round(single_acc, 4),
                'multi_restart_acc': round(restart_acc, 4),
                'gap': round(gap, 4),
                'diagnosis': 'masked' if gap > SUPERIORITY_THRESHOLD else 'ok',
            }
            print(f"  {name} ε={eps}: single={single_acc:.4f} "
                  f"multi={restart_acc:.4f} gap={gap:.4f}")
        records[name] = row
    return records


# ────────────────────────────────────────────────────────────────
# (c) Loss-landscape monotonicity
# ────────────────────────────────────────────────────────────────
def monotonicity_check(models, dataset):
    """Check that FGSM accuracy strictly decreases as epsilon grows."""
    records = {}
    for name in MODEL_NAMES:
        model = models[name]
        accuracies = []
        for eps in [0.0] + ADV_EPSILONS:
            total = 0
            correct = 0
            for images, labels in dataset:
                labels_int = tf.cast(labels, tf.int32)
                bs = tf.shape(images)[0].numpy()
                if eps > 0:
                    adv_images = fgsm_attack(model, images, labels, eps, name)
                else:
                    adv_images = images
                pre = preprocess_for_model(adv_images, name)
                preds = tf.cast(model(pre, training=False), tf.float32)
                cls = tf.argmax(preds, axis=1, output_type=tf.int32)
                correct += tf.reduce_sum(
                    tf.cast(tf.equal(cls, labels_int), tf.int32)).numpy()
                total += bs
            acc = correct / total
            accuracies.append((eps, acc))
            print(f"  {name} ε={eps}: acc={acc:.4f}")

        # check monotonicity
        non_mono_pairs = []
        for i in range(len(accuracies) - 1):
            e1, a1 = accuracies[i]
            e2, a2 = accuracies[i + 1]
            if a2 > a1 + 1e-4:
                non_mono_pairs.append({
                    'eps_low': round(e1, 4),
                    'eps_high': round(e2, 4),
                    'acc_low': round(a1, 4),
                    'acc_high': round(a2, 4),
                    'increase': round(a2 - a1, 4),
                })
        records[name] = {
            'accuracies': [{'epsilon': round(e, 4), 'accuracy': round(a, 4)}
                           for e, a in accuracies],
            'non_monotonic_pairs': non_mono_pairs,
            'diagnosis': 'non_monotonic' if non_mono_pairs else 'monotonic',
        }
    return records


# ────────────────────────────────────────────────────────────────
# Combined verdict
# ────────────────────────────────────────────────────────────────
def generate_verdict(grad_norms, restart_pgd_res, mono_res):
    """Final verdict per model: gradient_masking_detected (bool)."""
    verdict = {}
    for name in MODEL_NAMES:
        grad_diag = grad_norms[name]['diagnosis']
        pgd_diags = [v['diagnosis'] for v in restart_pgd_res[name].values()]
        mono_diag = mono_res[name]['diagnosis']

        flags = []
        if grad_diag == 'masked':
            flags.append('gradient_norms_near_zero')
        if 'masked' in pgd_diags:
            flags.append('restart_pgd_superiority')
        if mono_diag == 'non_monotonic':
            flags.append('loss_landscape_non_monotonic')

        detected = len(flags) >= 1
        verdict[name] = {
            'gradient_masking_detected': detected,
            'flags': flags,
            'severity': 'high' if len(flags) >= 2 else (
                'moderate' if len(flags) == 1 else 'none'),
        }
    return verdict


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TASK 1: Gradient Masking Diagnostic")
    print("=" * 60)
    seed_everything(SEED)

    print("\n[1/5] Loading models...")
    models = load_models()
    logits_models = build_all_logits_models(models)

    print("\n[2/5] Loading test dataset...")
    test_paths, test_labels, class_names = load_test_split()
    raw_test_ds = build_raw_test_dataset(test_paths, test_labels)
    print(f"  Test samples: {len(test_labels)}")

    print("\n[3/5] Sub-test (a): Gradient-norm histogram...")
    t0 = time.time()
    grad_norm_results = gradient_norm_analysis(models, raw_test_ds)
    print(f"  Done in {time.time()-t0:.1f} s")

    print("\n[4/5] Sub-test (b): Restart PGD superiority...")
    t0 = time.time()
    restart_results = restart_pgd_analysis(models, logits_models, raw_test_ds)
    print(f"  Done in {time.time()-t0:.1f} s")

    print("\n[5/5] Sub-test (c): Loss-landscape monotonicity...")
    t0 = time.time()
    mono_results = monotonicity_check(models, raw_test_ds)
    print(f"  Done in {time.time()-t0:.1f} s")

    # ── verdict ──
    verdict = generate_verdict(grad_norm_results, restart_results, mono_results)

    # ── save all results ──
    full_output = {
        'gradient_norms': grad_norm_results,
        'restart_pgd': restart_results,
        'monotonicity': mono_results,
        'verdict': verdict,
        'config': {
            'n_restarts': N_RESTARTS,
            'superiority_threshold': SUPERIORITY_THRESHOLD,
            'pgd_steps': PGD_STEPS,
            'epsilons': ADV_EPSILONS,
            'seed': SEED,
        },
    }
    out_file = OUT_DIR / 'gradient_masking_results.json'
    with open(out_file, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    # ── mono plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in MODEL_NAMES:
        accs = mono_results[name]['accuracies']
        eps_vals = [r['epsilon'] for r in accs]
        acc_vals = [r['accuracy'] for r in accs]
        ax.plot(eps_vals, acc_vals, 'o-', label=name, linewidth=2, markersize=6)
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('FGSM Accuracy')
    ax.set_title('Loss Landscape Monotonicity Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'monotonicity_check.png', dpi=150)
    plt.close()

    # ── summary table ──
    rows = []
    for name in MODEL_NAMES:
        v = verdict[name]
        rows.append({
            'Model': name,
            'Gradient_Masking': v['gradient_masking_detected'],
            'Severity': v['severity'],
            'Flags': ', '.join(v['flags']) if v['flags'] else 'none',
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'gradient_masking_summary.csv', index=False)
    print(f"\n  Summary table:")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    for name in MODEL_NAMES:
        v = verdict[name]
        emoji = '⚠' if v['gradient_masking_detected'] else '✓'
        print(f"  {emoji} {name}: masking={v['gradient_masking_detected']} "
              f"({v['severity']})")
    print("=" * 60)


if __name__ == '__main__':
    main()
