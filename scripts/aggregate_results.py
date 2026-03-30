"""
Task 5 (Part 2) — Aggregate Multi-Seed Results
===============================================
Load per-sample results from multi_seed_runner.py and compute:
  1. Bootstrap 95% CIs on accuracy and fooling rate
  2. Wilcoxon signed-rank tests: pairwise model comparisons
  3. Mean ± std across seeds
  4. LaTeX table with CIs

Run from project root:
    python scripts/aggregate_results.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    MODEL_NAMES, EPSILONS, ADV_EPSILONS, RESULTS_DIR
)

IN_DIR = RESULTS_DIR / 'multi_seed'
OUT_DIR = IN_DIR  # write alongside


N_BOOTSTRAP = 10000
CI_ALPHA = 0.05


def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA):
    """Compute bootstrap percentile CI for the mean."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    means = np.array([
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return round(float(np.mean(data)), 6), round(float(lo), 6), round(float(hi), 6)


def main():
    print("=" * 60)
    print("AGGREGATE: Multi-Seed Statistical Analysis")
    print("=" * 60)

    # ── Load per-sample data ──
    per_sample_file = IN_DIR / 'multi_seed_per_sample.json'
    if not per_sample_file.exists():
        print(f"  ERROR: {per_sample_file} not found. Run multi_seed_runner.py first.")
        return

    with open(per_sample_file, 'r') as f:
        all_seed_results = json.load(f)

    seeds = list(all_seed_results.keys())
    print(f"  Seeds: {seeds}")

    # ── 1. Bootstrap CIs ──
    print("\n[1/3] Computing bootstrap CIs...")
    ci_rows = []
    for name in MODEL_NAMES:
        for attack in ['fgsm', 'pgd']:
            eps_list = EPSILONS if attack == 'fgsm' else ADV_EPSILONS
            for eps in eps_list:
                key = f"{name}_{attack}_eps{eps}"
                accs = []
                for seed_str in seeds:
                    if key in all_seed_results[seed_str]:
                        accs.append(all_seed_results[seed_str][key]['accuracy'])

                if len(accs) < 2:
                    continue

                mean_val = np.mean(accs)
                std_val = np.std(accs, ddof=1)

                # Per-seed bootstrap (pool per-sample from all seeds)
                pooled = []
                for seed_str in seeds:
                    if key in all_seed_results[seed_str]:
                        pooled.extend(all_seed_results[seed_str][key]['per_sample'])

                _, ci_lo, ci_hi = bootstrap_ci(pooled)

                ci_rows.append({
                    'Model': name,
                    'Attack': attack.upper(),
                    'Epsilon': eps,
                    'Mean_Accuracy': round(mean_val, 4),
                    'Std': round(std_val, 4),
                    'CI_95_Lo': ci_lo,
                    'CI_95_Hi': ci_hi,
                    'CI_Width': round(ci_hi - ci_lo, 4),
                })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(OUT_DIR / 'bootstrap_confidence_intervals.csv', index=False)
    print(f"  Saved {len(ci_rows)} CI rows")

    # ── 2. Wilcoxon signed-rank tests ──
    print("\n[2/3] Wilcoxon signed-rank tests...")
    wilcoxon_rows = []
    pairs = [
        ('VGG19', 'ResNet50'),
        ('VGG19', 'DenseNet121'),
        ('ResNet50', 'DenseNet121'),
    ]
    for attack in ['fgsm', 'pgd']:
        eps_list = EPSILONS if attack == 'fgsm' else ADV_EPSILONS
        for eps in eps_list:
            for m1, m2 in pairs:
                key1 = f"{m1}_{attack}_eps{eps}"
                key2 = f"{m2}_{attack}_eps{eps}"

                # Collect per-sample from seed 42 for paired test
                # (Wilcoxon requires paired data — same test set)
                seed_str = '42'
                if (seed_str in all_seed_results and
                        key1 in all_seed_results[seed_str] and
                        key2 in all_seed_results[seed_str]):

                    s1 = np.array(all_seed_results[seed_str][key1]['per_sample'])
                    s2 = np.array(all_seed_results[seed_str][key2]['per_sample'])

                    diff = s1 - s2
                    if np.all(diff == 0):
                        p_val = 1.0
                        stat = 0.0
                    else:
                        try:
                            stat, p_val = wilcoxon(s1, s2)
                        except ValueError:
                            stat, p_val = 0.0, 1.0

                    wilcoxon_rows.append({
                        'Attack': attack.upper(),
                        'Epsilon': eps,
                        'Model_A': m1,
                        'Model_B': m2,
                        'Mean_A': round(float(np.mean(s1)), 4),
                        'Mean_B': round(float(np.mean(s2)), 4),
                        'Wilcoxon_Stat': round(float(stat), 2),
                        'p_value': round(float(p_val), 6),
                        'Significant_005': p_val < 0.05,
                        'Significant_001': p_val < 0.01,
                    })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    df_wilcoxon.to_csv(OUT_DIR / 'wilcoxon_tests.csv', index=False)
    print(f"  Saved {len(wilcoxon_rows)} test rows")

    # ── 3. LaTeX table ──
    print("\n[3/3] Generating LaTeX tables...")
    max_eps = ADV_EPSILONS[-1]
    for attack in ['fgsm', 'pgd']:
        eps_list = EPSILONS if attack == 'fgsm' else ADV_EPSILONS
        tex_lines = [
            r'\begin{table}[h]',
            r'\centering',
            f'\\caption{{{attack.upper()} Robust Accuracy '
            f'(Mean $\\pm$ 95\\% CI, {len(seeds)} seeds)}}',
            r'\begin{tabular}{l' + 'c' * len(eps_list) + '}',
            r'\toprule',
            'Model & ' + ' & '.join(
                [f'$\\varepsilon={e}$' for e in eps_list]) + r' \\',
            r'\midrule',
        ]
        for name in MODEL_NAMES:
            cells = []
            for eps in eps_list:
                match = df_ci[(df_ci['Model'] == name) &
                              (df_ci['Attack'] == attack.upper()) &
                              (df_ci['Epsilon'] == eps)]
                if len(match) > 0:
                    row = match.iloc[0]
                    cell = (f'{row["Mean_Accuracy"]:.3f}'
                            f'$_{{[{row["CI_95_Lo"]:.3f},{row["CI_95_Hi"]:.3f}]}}$')
                else:
                    cell = '-'
                cells.append(cell)
            tex_lines.append(f'{name} & ' + ' & '.join(cells) + r' \\')
        tex_lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
        with open(OUT_DIR / f'{attack}_ci_table.tex', 'w') as f:
            f.write('\n'.join(tex_lines))

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  STATISTICAL VALIDATION SUMMARY")
    print("=" * 60)

    # Cross-seed stability
    print("\n  Cross-Seed Accuracy Stability:")
    for name in MODEL_NAMES:
        for attack in ['fgsm', 'pgd']:
            key = f"{name}_{attack}_eps{max_eps}"
            accs = []
            for seed_str in seeds:
                if key in all_seed_results[seed_str]:
                    accs.append(all_seed_results[seed_str][key]['accuracy'])
            if accs:
                print(f"    {name}/{attack.upper()} ε={max_eps}: "
                      f"{np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")

    sig_tests = df_wilcoxon[df_wilcoxon['Significant_005']]
    print(f"\n  Significant Wilcoxon tests (p<0.05): {len(sig_tests)}/{len(df_wilcoxon)}")
    if len(sig_tests) > 0:
        for _, r in sig_tests.iterrows():
            print(f"    {r['Attack']} ε={r['Epsilon']}: "
                  f"{r['Model_A']}({r['Mean_A']:.4f}) vs "
                  f"{r['Model_B']}({r['Mean_B']:.4f}), "
                  f"p={r['p_value']:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
