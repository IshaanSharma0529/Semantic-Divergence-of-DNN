"""
Task 4 — Confusion Direction Analysis
======================================
Build adversarial confusion matrices and extract semantic structure.

For each model × attack × epsilon, this script:
  1. Builds the confusion matrix: C[i,j] = # samples with true=i, adv_pred=j
  2. Computes Semantic Structure Score (SSS):
       SSS = 1 - H(C_off_diag) / H_max
     where H is Shannon entropy of the off-diagonal confusion distribution.
     SSS ≈ 1 → adversarial misclassifications concentrate on few target classes (semantic).
     SSS ≈ 0 → uniform scatter = random noise.
  3. Extracts top-K confused class pairs per model.
  4. Checks cross-model overlap: do VGG19 and ResNet50 produce the same
     confused pairs?  High overlap → shared representation vulnerabilities.

Outputs:
    Model Training/fgsm_results/confusion_analysis/
        confusion_matrices_{model}_{attack}_eps{eps}.csv
        semantic_structure_scores.csv
        top_confused_pairs.json
        cross_model_overlap.json
        figures/

Run from project root:
    python experiments/confusion_direction_analysis.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    load_test_split, build_raw_test_dataset, load_models,
    build_all_logits_models, preprocess_for_model,
    fgsm_attack, pgd_attack,
    MODEL_NAMES, ADV_EPSILONS, PGD_STEPS, SEED, NUM_CLASSES,
    seed_everything, RESULTS_DIR, get_class_names,
)

OUT_DIR = RESULTS_DIR / 'confusion_analysis'
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20  # top confused pairs to report


# ────────────────────────────────────────────────────────────────
# Build confusion matrix under attack
# ────────────────────────────────────────────────────────────────
def build_adversarial_confusion(model, logits_model, raw_test_ds,
                                 model_name, attack, epsilon):
    """Build confusion matrix: C[true, adv_pred].

    Returns:
        C: (num_classes × num_classes) integer confusion matrix
        records: list of per-sample dicts
    """
    num_classes = NUM_CLASSES
    C = np.zeros((num_classes, num_classes), dtype=np.int32)
    records = []

    for images, labels in raw_test_ds:
        labels_int = tf.cast(labels, tf.int32)
        bs = tf.shape(images)[0].numpy()

        # Generate adversarials
        if attack == 'fgsm':
            adv_images = fgsm_attack(model, images, labels, epsilon, model_name)
        elif attack == 'pgd':
            step_size = epsilon / 4.0
            adv_images = pgd_attack(logits_model, images, labels, epsilon,
                                    PGD_STEPS, step_size, model_name)
        else:
            raise ValueError(f"Unknown attack: {attack}")

        adv_pre = preprocess_for_model(adv_images, model_name)
        adv_preds_prob = tf.cast(model(adv_pre, training=False), tf.float32)
        adv_classes = tf.argmax(adv_preds_prob, axis=1, output_type=tf.int32)
        adv_confs = tf.reduce_max(adv_preds_prob, axis=1)

        for j in range(bs):
            t = int(labels_int[j].numpy())
            p = int(adv_classes[j].numpy())
            C[t, p] += 1
            if t != p:
                records.append({
                    'true_label': t,
                    'adv_pred': p,
                    'adv_confidence': round(float(adv_confs[j].numpy()), 4),
                })

    return C, records


# ────────────────────────────────────────────────────────────────
# Semantic Structure Score
# ────────────────────────────────────────────────────────────────
def compute_semantic_structure_score(C):
    """Compute SSS from confusion matrix.

    SSS = 1 - H(off_diagonal) / H_max_possible
    """
    num_classes = C.shape[0]
    # off-diagonal elements
    mask = ~np.eye(num_classes, dtype=bool)
    off_diag = C[mask].flatten().astype(float)
    total_off = off_diag.sum()

    if total_off == 0:
        return 1.0  # no misclassifications = perfectly structured

    probs = off_diag / total_off
    H = float(entropy(probs + 1e-12, base=2))
    # maximum entropy = log2(num_off_diagonal_cells)
    H_max = np.log2(len(off_diag))

    sss = 1.0 - (H / H_max) if H_max > 0 else 1.0
    return round(sss, 6)


def extract_top_confused_pairs(C, class_names, top_k=TOP_K):
    """Extract top confused (true_class, adv_class) pairs by count."""
    pairs = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i != j and C[i, j] > 0:
                pairs.append({
                    'true_label': i,
                    'true_class': class_names[i],
                    'adv_label': j,
                    'adv_class': class_names[j],
                    'count': int(C[i, j]),
                })
    pairs.sort(key=lambda x: x['count'], reverse=True)
    return pairs[:top_k]


# ────────────────────────────────────────────────────────────────
# Cross-model overlap
# ────────────────────────────────────────────────────────────────
def compute_cross_model_overlap(all_top_pairs):
    """Compute Jaccard similarity of top confused pairs across models.

    all_top_pairs: dict[model_name] → list of top confused dicts
    """
    overlap = {}
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            if i >= j:
                continue
            set1 = set((p['true_label'], p['adv_label'])
                       for p in all_top_pairs.get(m1, []))
            set2 = set((p['true_label'], p['adv_label'])
                       for p in all_top_pairs.get(m2, []))

            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0.0
            overlap[f"{m1}_vs_{m2}"] = {
                'jaccard': round(jaccard, 4),
                'shared_pairs': intersection,
                'union': union,
                'shared_pair_list': [
                    {'true': p[0], 'adv': p[1]}
                    for p in sorted(set1 & set2)
                ],
            }
    return overlap


# ────────────────────────────────────────────────────────────────
# Visualizations
# ────────────────────────────────────────────────────────────────
def plot_confusion_matrix(C, class_names, model_name, attack, eps, out_path):
    """Plot confusion matrix (top 20 most active classes for readability)."""
    # Select top-20 classes by total misclassification involvement
    row_sums = C.sum(axis=1) - np.diag(C)
    col_sums = C.sum(axis=0) - np.diag(C)
    activity = row_sums + col_sums
    top_idx = np.argsort(activity)[-20:]

    C_sub = C[np.ix_(top_idx, top_idx)]
    labels = [class_names[i][:12] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(C_sub, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Adversarial Prediction')
    ax.set_ylabel('True Class')
    ax.set_title(f'{model_name} {attack.upper()} Confusion (ε={eps}, top-20 classes)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sss_comparison(sss_data, out_path):
    """Bar chart comparing SSS across models, attacks, epsilons."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, attack in enumerate(['fgsm', 'pgd']):
        ax = axes[idx]
        for name in MODEL_NAMES:
            key_prefix = f"{name}_{attack}"
            eps_vals = []
            sss_vals = []
            for eps in ADV_EPSILONS:
                key = f"{name}_{attack}_eps{eps}"
                if key in sss_data:
                    eps_vals.append(eps)
                    sss_vals.append(sss_data[key])
            ax.plot(eps_vals, sss_vals, 'o-', label=name, linewidth=2)

        ax.set_xlabel('Epsilon (ε)')
        ax.set_ylabel('Semantic Structure Score')
        ax.set_title(f'{attack.upper()} SSS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TASK 4: Confusion Direction Analysis")
    print("=" * 60)
    seed_everything(SEED)

    print("\n[1/3] Loading models & data...")
    models = load_models()
    logits_models = build_all_logits_models(models)
    test_paths, test_labels, class_names = load_test_split()
    raw_test_ds = build_raw_test_dataset(test_paths, test_labels)
    print(f"  Test samples: {len(test_labels)}, Classes: {len(class_names)}")

    sss_data = {}
    all_top_pairs = {}
    all_results = {}

    print("\n[2/3] Building confusion matrices...")
    for attack in ['fgsm', 'pgd']:
        for eps in ADV_EPSILONS:
            print(f"\n  --- {attack.upper()} ε={eps} ---")
            eps_top_pairs = {}

            for name in MODEL_NAMES:
                t0 = time.time()
                print(f"    {name} ... ", end='', flush=True)

                C, records = build_adversarial_confusion(
                    models[name], logits_models[name], raw_test_ds,
                    name, attack, eps)

                sss = compute_semantic_structure_score(C)
                top_pairs = extract_top_confused_pairs(C, class_names)

                key = f"{name}_{attack}_eps{eps}"
                sss_data[key] = sss
                eps_top_pairs[name] = top_pairs

                # Save confusion matrix CSV
                df_C = pd.DataFrame(C, index=class_names, columns=class_names)
                df_C.to_csv(OUT_DIR / f'confusion_{name}_{attack}_eps{eps}.csv')

                # Plot
                plot_confusion_matrix(C, class_names, name, attack, eps,
                                      FIG_DIR / f'confusion_{name}_{attack}_eps{eps}.png')

                all_results[key] = {
                    'sss': sss,
                    'total_misclassified': int(C.sum() - np.trace(C)),
                    'top_pairs': top_pairs[:10],
                }
                print(f"SSS={sss:.4f} ({time.time()-t0:.1f}s)")

            # Cross-model overlap for this (attack, eps)
            overlap = compute_cross_model_overlap(eps_top_pairs)
            all_results[f'overlap_{attack}_eps{eps}'] = overlap

            # Store top pairs for highest eps for final analysis
            if eps == ADV_EPSILONS[-1]:
                all_top_pairs.update(
                    {f"{n}_{attack}": eps_top_pairs[n] for n in MODEL_NAMES})

    # ── Save SSS table ──
    print("\n[3/3] Generating summaries...")
    sss_rows = []
    for key, sss in sss_data.items():
        parts = key.split('_')
        model = parts[0]
        attack = parts[1]
        eps = float(parts[2].replace('eps', ''))
        sss_rows.append({
            'Model': model, 'Attack': attack, 'Epsilon': eps, 'SSS': sss,
        })
    df_sss = pd.DataFrame(sss_rows)
    df_sss.to_csv(OUT_DIR / 'semantic_structure_scores.csv', index=False)

    # ── SSS plot ──
    plot_sss_comparison(sss_data, FIG_DIR / 'sss_comparison.png')

    # ── Save comprehensive results ──
    with open(OUT_DIR / 'confusion_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Top confused pairs summary ──
    with open(OUT_DIR / 'top_confused_pairs.json', 'w') as f:
        json.dump({k: v for k, v in all_top_pairs.items()}, f, indent=2)

    # ── LaTeX table for SSS ──
    tex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Semantic Structure Score (SSS) across models and attacks}',
        r'\begin{tabular}{ll' + 'c' * len(ADV_EPSILONS) + '}',
        r'\toprule',
        'Model & Attack & ' + ' & '.join(
            [f'$\\varepsilon={e}$' for e in ADV_EPSILONS]) + r' \\',
        r'\midrule',
    ]
    for attack in ['fgsm', 'pgd']:
        for name in MODEL_NAMES:
            cells = []
            for eps in ADV_EPSILONS:
                key = f"{name}_{attack}_eps{eps}"
                cells.append(f'{sss_data.get(key, 0):.3f}')
            tex_lines.append(
                f'{name} & {attack.upper()} & ' +
                ' & '.join(cells) + r' \\')
        tex_lines.append(r'\midrule')
    tex_lines[-1] = r'\bottomrule'
    tex_lines += [r'\end{tabular}', r'\end{table}']
    with open(OUT_DIR / 'sss_table.tex', 'w') as f:
        f.write('\n'.join(tex_lines))

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  CONFUSION DIRECTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n  Semantic Structure Scores (ε={ADV_EPSILONS[-1]}):")
    for name in MODEL_NAMES:
        for attack in ['fgsm', 'pgd']:
            key = f"{name}_{attack}_eps{ADV_EPSILONS[-1]}"
            print(f"    {name}/{attack.upper()}: SSS = {sss_data.get(key, 0):.4f}")

    print(f"\n  Top-3 confused pairs (FGSM, ε={ADV_EPSILONS[-1]}):")
    for name in MODEL_NAMES:
        pairs = all_top_pairs.get(f"{name}_fgsm", [])[:3]
        for p in pairs:
            print(f"    {name}: {p['true_class']} → {p['adv_class']} "
                  f"(n={p['count']})")
    print("=" * 60)


if __name__ == '__main__':
    main()
