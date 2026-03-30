"""
Task 3 — Transfer Attack Matrix
================================
Build a 3×3 source→target robustness matrix for FGSM and PGD attacks.

Protocol:
  For each source model S and target model T:
    1. Generate adversarial examples on S  (already in Adversarial Bank)
    2. Evaluate those adversarial examples on T  (cross-model transfer)
    3. Record: transfer_fooling_rate = frac of T-correct samples misclassified

This quantifies:
  - Self-attack diagonal  (same as original results — sanity check)
  - Off-diagonal transfer rates
  - Boundary-distance correlation  (using DeepFool L2 norms)

Outputs:
    Model Training/fgsm_results/transfer_matrix/
        transfer_matrix_fgsm.csv   — 3×3 per epsilon
        transfer_matrix_pgd.csv    — 3×3 per epsilon
        transfer_matrix_figures/   — heatmaps
        transfer_summary.json      — all raw results

Run from project root:
    python experiments/transfer_attack_matrix.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    load_test_split, build_raw_test_dataset, load_models,
    build_all_logits_models, preprocess_for_model,
    fgsm_attack, pgd_attack,
    MODEL_NAMES, ADV_EPSILONS, PGD_STEPS, SEED,
    seed_everything, RESULTS_DIR,
)

OUT_DIR = RESULTS_DIR / 'transfer_matrix'
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# Core transfer evaluation
# ────────────────────────────────────────────────────────────────
def evaluate_transfer(target_model, adv_images, true_labels,
                      target_name):
    """Evaluate adversarial images (from source) on target model.

    Returns:
        accuracy: fraction correctly classified by target
        fooling_rate: fraction where target pred ≠ true label
    """
    pre = preprocess_for_model(adv_images, target_name)
    preds = tf.cast(target_model(pre, training=False), tf.float32)
    pred_classes = tf.argmax(preds, axis=1, output_type=tf.int32)
    labels_int = tf.cast(true_labels, tf.int32)

    correct = tf.reduce_sum(tf.cast(tf.equal(pred_classes, labels_int), tf.int32)).numpy()
    return int(correct), int(tf.shape(adv_images)[0].numpy())


def build_transfer_matrix_fgsm(models, raw_test_ds, epsilon):
    """Build 3×3 FGSM transfer matrix for one epsilon."""
    matrix = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))

    for i, src in enumerate(MODEL_NAMES):
        print(f"    Source={src} generating adversarials ... ", flush=True)

        for j, tgt in enumerate(MODEL_NAMES):
            total_correct = 0
            total_samples = 0
            # Generate and evaluate batch-by-batch to avoid OOM
            for images, labels in raw_test_ds:
                adv = fgsm_attack(models[src], images, labels, epsilon, src)
                c, n = evaluate_transfer(models[tgt], adv,
                                         labels, tgt)
                total_correct += c
                total_samples += n
                del adv  # free GPU memory immediately
            acc = total_correct / total_samples
            fooling = 1.0 - acc
            matrix[i, j] = round(fooling, 4)
            diag_mark = ' (self)' if i == j else ''
            print(f"      → Target={tgt}: fooling={fooling:.4f}{diag_mark}")

    return matrix


def build_transfer_matrix_pgd(models, logits_models, raw_test_ds, epsilon):
    """Build 3×3 PGD transfer matrix for one epsilon."""
    step_size = epsilon / 4.0
    matrix = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))

    for i, src in enumerate(MODEL_NAMES):
        print(f"    Source={src} generating PGD adversarials ... ", flush=True)

        for j, tgt in enumerate(MODEL_NAMES):
            total_correct = 0
            total_samples = 0
            # Generate and evaluate batch-by-batch to avoid OOM
            for images, labels in raw_test_ds:
                adv = pgd_attack(logits_models[src], images, labels, epsilon,
                                 PGD_STEPS, step_size, src)
                c, n = evaluate_transfer(models[tgt], adv,
                                         labels, tgt)
                total_correct += c
                total_samples += n
                del adv  # free GPU memory immediately
            acc = total_correct / total_samples
            fooling = 1.0 - acc
            matrix[i, j] = round(fooling, 4)
            diag_mark = ' (self)' if i == j else ''
            print(f"      → Target={tgt}: fooling={fooling:.4f}{diag_mark}")

    return matrix


# ────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────
def plot_transfer_heatmap(matrix, attack_name, epsilon, out_path):
    """Plot 3×3 transfer matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES,
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Target Model')
    ax.set_ylabel('Source Model')
    ax.set_title(f'{attack_name} Transfer Fooling Rate (ε={epsilon})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────
# Analysis
# ────────────────────────────────────────────────────────────────
def analyse_transferability(all_matrices, attack_name):
    """Compute transferability metrics from matrices."""
    results = {}
    for eps, matrix in all_matrices.items():
        diagonal = np.diag(matrix)
        off_diag = matrix[~np.eye(len(MODEL_NAMES), dtype=bool)]

        # Average transfer rate (off-diagonal)
        avg_transfer = float(np.mean(off_diag))
        # Transfer efficiency: off-diag / diagonal mean
        avg_self = float(np.mean(diagonal))
        transfer_efficiency = avg_transfer / avg_self if avg_self > 0 else 0

        # Most/least transferable pairs
        pairs = []
        for i, src in enumerate(MODEL_NAMES):
            for j, tgt in enumerate(MODEL_NAMES):
                if i != j:
                    pairs.append({
                        'source': src, 'target': tgt,
                        'fooling_rate': round(float(matrix[i, j]), 4),
                    })
        pairs.sort(key=lambda x: x['fooling_rate'], reverse=True)

        results[str(eps)] = {
            'self_attack_rates': {n: round(float(diagonal[k]), 4)
                                  for k, n in enumerate(MODEL_NAMES)},
            'avg_transfer_rate': round(avg_transfer, 4),
            'avg_self_attack_rate': round(avg_self, 4),
            'transfer_efficiency': round(transfer_efficiency, 4),
            'most_transferable': pairs[0] if pairs else None,
            'least_transferable': pairs[-1] if pairs else None,
            'all_pairs': pairs,
        }
    return results


def main():
    print("=" * 60)
    print("TASK 3: Transfer Attack Matrix")
    print("=" * 60)
    seed_everything(SEED)

    print("\n[1/4] Loading models...")
    models = load_models()
    logits_models = build_all_logits_models(models)

    print("\n[2/4] Loading test dataset...")
    test_paths, test_labels, class_names = load_test_split()
    raw_test_ds = build_raw_test_dataset(test_paths, test_labels)
    print(f"  Test samples: {len(test_labels)}")

    all_results = {}

    # ── FGSM transfer ──
    print("\n[3/4] Building FGSM transfer matrices...")
    fgsm_matrices = {}
    for eps in ADV_EPSILONS:
        print(f"\n  ε = {eps}")
        matrix = build_transfer_matrix_fgsm(models, raw_test_ds, eps)
        fgsm_matrices[eps] = matrix

        # Save CSV
        df = pd.DataFrame(matrix, index=MODEL_NAMES, columns=MODEL_NAMES)
        df.to_csv(OUT_DIR / f'transfer_fgsm_eps{eps}.csv')
        plot_transfer_heatmap(matrix, 'FGSM', eps,
                              FIG_DIR / f'transfer_fgsm_eps{eps}.png')

    fgsm_analysis = analyse_transferability(fgsm_matrices, 'FGSM')
    all_results['fgsm'] = {
        'matrices': {str(e): m.tolist() for e, m in fgsm_matrices.items()},
        'analysis': fgsm_analysis,
    }

    # ── PGD transfer ──
    print("\n[4/4] Building PGD transfer matrices...")
    pgd_matrices = {}
    for eps in ADV_EPSILONS:
        print(f"\n  ε = {eps}")
        matrix = build_transfer_matrix_pgd(models, logits_models,
                                            raw_test_ds, eps)
        pgd_matrices[eps] = matrix

        df = pd.DataFrame(matrix, index=MODEL_NAMES, columns=MODEL_NAMES)
        df.to_csv(OUT_DIR / f'transfer_pgd_eps{eps}.csv')
        plot_transfer_heatmap(matrix, 'PGD', eps,
                              FIG_DIR / f'transfer_pgd_eps{eps}.png')

    pgd_analysis = analyse_transferability(pgd_matrices, 'PGD')
    all_results['pgd'] = {
        'matrices': {str(e): m.tolist() for e, m in pgd_matrices.items()},
        'analysis': pgd_analysis,
    }

    # ── Save comprehensive results ──
    with open(OUT_DIR / 'transfer_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Combined summary heatmap (highest ε) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_eps = ADV_EPSILONS[-1]
    for idx, (attack_name, matrices) in enumerate([('FGSM', fgsm_matrices),
                                                     ('PGD', pgd_matrices)]):
        m = matrices[max_eps]
        sns.heatmap(m, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES,
                    vmin=0, vmax=1, ax=axes[idx])
        axes[idx].set_xlabel('Target Model')
        axes[idx].set_ylabel('Source Model')
        axes[idx].set_title(f'{attack_name} Transfer (ε={max_eps})')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'transfer_combined_heatmap.png', dpi=150)
    plt.close()

    # ── LaTeX table ──
    for attack_name, matrices in [('fgsm', fgsm_matrices), ('pgd', pgd_matrices)]:
        m = matrices[max_eps]
        tex_lines = [
            r'\begin{table}[h]',
            r'\centering',
            f'\\caption{{{attack_name.upper()} Transfer Fooling Rate '
            f'($\\varepsilon={max_eps}$)}}',
            r'\begin{tabular}{l' + 'c' * len(MODEL_NAMES) + '}',
            r'\toprule',
            'Source $\\rightarrow$ Target & ' +
            ' & '.join(MODEL_NAMES) + r' \\',
            r'\midrule',
        ]
        for i, src in enumerate(MODEL_NAMES):
            cells = []
            for j in range(len(MODEL_NAMES)):
                val = f'{m[i,j]:.3f}'
                if i == j:
                    val = f'\\textbf{{{val}}}'
                cells.append(val)
            tex_lines.append(f'{src} & ' + ' & '.join(cells) + r' \\')
        tex_lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
        with open(OUT_DIR / f'transfer_{attack_name}_table.tex', 'w') as f:
            f.write('\n'.join(tex_lines))

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  TRANSFER MATRIX SUMMARY")
    print("=" * 60)
    for attack_name, analysis in [('FGSM', fgsm_analysis), ('PGD', pgd_analysis)]:
        print(f"\n  {attack_name} (ε={max_eps}):")
        a = analysis[str(max_eps)]
        print(f"    Self-attack rates: {a['self_attack_rates']}")
        print(f"    Avg transfer rate: {a['avg_transfer_rate']:.4f}")
        print(f"    Transfer efficiency: {a['transfer_efficiency']:.4f}")
        if a['most_transferable']:
            mt = a['most_transferable']
            print(f"    Most transferable: {mt['source']}→{mt['target']} "
                  f"({mt['fooling_rate']:.4f})")
    print("=" * 60)


if __name__ == '__main__':
    main()
