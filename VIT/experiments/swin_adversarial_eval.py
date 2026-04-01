"""
Swin Transformer — Adversarial Robustness Evaluation
=====================================================
White-box FGSM and PGD attacks directly on Swin-Tiny.

For each attack × epsilon:
  - Adversarial accuracy
  - Accuracy drop (Δ)
  - Fooling rate
  - Mean clean / adversarial confidence

Outputs:
    Swin_Training/adversarial_results/swin_tiny_fgsm.json
    Swin_Training/adversarial_results/swin_tiny_pgd.json
    Swin_Training/adversarial_results/swin_tiny_summary.json

Run from project root:
    python VIT/experiments/swin_adversarial_eval.py [--max-samples N] [--epsilons 0.01 0.02]
"""

import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.swin_utils import (
    load_swin_config, load_swin_model, get_dataloader,
    seed_everything, get_device, PROJECT_ROOT,
    fgsm_attack_swin, pgd_attack_swin, evaluate_under_attack,
)


def main():
    parser = argparse.ArgumentParser(description='Swin-Tiny adversarial evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit test samples (for quick debugging)')
    parser.add_argument('--epsilons', type=float, nargs='+', default=None,
                        help='Override epsilons (e.g. --epsilons 0.01 0.02)')
    parser.add_argument('--skip-pgd', action='store_true',
                        help='Skip PGD (slower than FGSM)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    args = parser.parse_args()

    print("=" * 60)
    print("Swin-Tiny — Adversarial Robustness Evaluation")
    print("=" * 60)

    config = load_swin_config()
    seed_everything(config['evaluation']['seed'])
    device = get_device()

    # ── Load model ──
    print("\n[1/3] Loading model...")
    model, processor = load_swin_model(
        checkpoint=config['model']['hf_checkpoint'],
        device=device,
    )

    # ── Load data ──
    print("\n[2/3] Loading ImageNet-100 test set...")
    batch_size = args.batch_size or config['evaluation']['batch_size']
    data_dir = PROJECT_ROOT / config['dataset']['data_dir']
    dataloader = get_dataloader(
        data_dir=data_dir,
        split=config['dataset']['splits']['test'],
        processor=processor,
        batch_size=batch_size,
        num_workers=config['evaluation']['num_workers'],
        max_samples=args.max_samples,
    )

    # ── Run attacks ──
    print("\n[3/3] Running adversarial evaluation...")
    out_dir = PROJECT_ROOT / config['output']['adversarial_results']
    out_dir.mkdir(parents=True, exist_ok=True)

    fgsm_epsilons = args.epsilons or config['attacks']['fgsm']['epsilons']
    pgd_epsilons = args.epsilons or config['attacks']['pgd']['epsilons']
    pgd_steps = config['attacks']['pgd']['steps']
    pgd_step_ratio = config['attacks']['pgd']['step_size_ratio']

    all_results = {}

    # ── FGSM ──
    print("\n--- FGSM Attack ---")
    fgsm_results = []
    for eps in fgsm_epsilons:
        print(f"  ε = {eps} ... ", end='', flush=True)
        t0 = time.time()
        result = evaluate_under_attack(
            model=model,
            dataloader=dataloader,
            device=device,
            attack_fn=fgsm_attack_swin,
            epsilon=eps,
        )
        elapsed = time.time() - t0
        result['time_seconds'] = round(elapsed, 2)
        fgsm_results.append(result)
        print(f"adv_acc={result['adv_accuracy']:.4f}  "
              f"Δ={result['accuracy_drop']:.4f}  "
              f"fooled={result['fooling_rate']:.4f}  "
              f"({elapsed:.1f}s)")

    fgsm_file = out_dir / 'swin_tiny_fgsm.json'
    with open(fgsm_file, 'w') as f:
        json.dump({'attack': 'fgsm', 'model': config['model']['name'],
                   'results': fgsm_results}, f, indent=2)
    print(f"  → Saved to {fgsm_file}")
    all_results['fgsm'] = fgsm_results

    # ── PGD ──
    if not args.skip_pgd:
        print("\n--- PGD Attack ---")
        pgd_results = []
        for eps in pgd_epsilons:
            step_size = eps * pgd_step_ratio
            print(f"  ε = {eps} (steps={pgd_steps}, α={step_size:.4f}) ... ",
                  end='', flush=True)
            t0 = time.time()
            result = evaluate_under_attack(
                model=model,
                dataloader=dataloader,
                device=device,
                attack_fn=pgd_attack_swin,
                epsilon=eps,
                steps=pgd_steps,
                step_size=step_size,
            )
            elapsed = time.time() - t0
            result['time_seconds'] = round(elapsed, 2)
            result['pgd_steps'] = pgd_steps
            result['pgd_step_size'] = step_size
            pgd_results.append(result)
            print(f"adv_acc={result['adv_accuracy']:.4f}  "
                  f"Δ={result['accuracy_drop']:.4f}  "
                  f"fooled={result['fooling_rate']:.4f}  "
                  f"({elapsed:.1f}s)")

        pgd_file = out_dir / 'swin_tiny_pgd.json'
        with open(pgd_file, 'w') as f:
            json.dump({'attack': 'pgd', 'model': config['model']['name'],
                       'results': pgd_results}, f, indent=2)
        print(f"  → Saved to {pgd_file}")
        all_results['pgd'] = pgd_results

    # ── Summary ──
    summary_file = out_dir / 'swin_tiny_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'model': config['model']['hf_checkpoint'],
            'model_name': config['model']['name'],
            'dataset': config['dataset']['name'],
            'seed': config['evaluation']['seed'],
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Summary saved to {summary_file}")

    # ── Print summary table ──
    print("\n" + "=" * 60)
    print(f"  {'Attack':<10} {'ε':<8} {'Clean%':<10} {'Adv%':<10} {'Δ%':<10} {'Fooled%':<10}")
    print("-" * 60)
    for attack_name, results_list in all_results.items():
        for r in results_list:
            print(f"  {attack_name:<10} {r['epsilon']:<8.3f} "
                  f"{r['clean_accuracy']*100:<10.2f} "
                  f"{r['adv_accuracy']*100:<10.2f} "
                  f"{r['accuracy_drop']*100:<10.2f} "
                  f"{r['fooling_rate']*100:<10.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
