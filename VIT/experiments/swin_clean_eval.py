"""
Swin Transformer — Clean Baseline Evaluation
=============================================
Run pretrained Swin-Tiny on ImageNet-100 test set and record:
  - Top-1 accuracy
  - Per-class accuracy
  - Mean confidence
  - Confusion matrix (saved as PNG)

Outputs:
    Swin_Training/clean_baselines/swin_tiny_clean.json
    Swin_Training/clean_baselines/swin_tiny_confusion_matrix.png

Run from project root:
    python VIT/experiments/swin_clean_eval.py [--max-samples N]
"""

import os, sys, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.swin_utils import (
    load_swin_config, load_swin_model, get_dataloader,
    load_class_names, seed_everything, get_device, PROJECT_ROOT,
)


def plot_confusion_matrix(all_preds, all_labels, num_classes, save_path):
    """Generate and save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Swin-Tiny Clean Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Swin-Tiny clean evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit test samples (for quick debugging)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    args = parser.parse_args()

    print("=" * 60)
    print("Swin-Tiny — Clean Baseline Evaluation")
    print("=" * 60)

    config = load_swin_config()
    seed_everything(config['evaluation']['seed'])
    device = get_device()

    # ── Load model ──
    print("\n[1/4] Loading model...")
    model, processor = load_swin_model(
        checkpoint=config['model']['hf_checkpoint'],
        device=device,
    )

    # ── Load data ──
    print("\n[2/4] Loading ImageNet-100 test set...")
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

    # ── Evaluate ──
    print("\n[3/4] Running clean evaluation...")
    import torch
    import torch.nn.functional as F

    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    num_classes = config['dataset']['num_classes']

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (pixel_values, labels) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            probs = F.softmax(outputs.logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
            all_confs.extend(confs.cpu().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx + 1}/{len(dataloader)}")
    elapsed = time.time() - t0

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    mean_conf = float(np.mean(all_confs))

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[str(c)] = float((all_preds[mask] == c).mean())

    print(f"\n  Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Mean Confidence: {mean_conf:.4f}")
    print(f"  Total Samples: {len(all_labels)}")
    print(f"  Inference Time: {elapsed:.1f}s ({len(all_labels)/elapsed:.1f} img/s)")

    # ── Save results ──
    print("\n[4/4] Saving results...")
    out_dir = PROJECT_ROOT / config['output']['clean_baselines']
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model': config['model']['hf_checkpoint'],
        'model_name': config['model']['name'],
        'dataset': config['dataset']['name'],
        'num_classes': num_classes,
        'total_samples': len(all_labels),
        'accuracy': float(accuracy),
        'mean_confidence': mean_conf,
        'inference_time_seconds': round(elapsed, 2),
        'throughput_img_per_sec': round(len(all_labels) / elapsed, 1),
        'per_class_accuracy': per_class_acc,
        'seed': config['evaluation']['seed'],
    }

    results_file = out_dir / 'swin_tiny_clean.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_file}")

    # Confusion matrix
    try:
        cm_file = out_dir / 'swin_tiny_confusion_matrix.png'
        plot_confusion_matrix(all_preds, all_labels, num_classes, cm_file)
    except ImportError:
        print("  sklearn not available — skipping confusion matrix")

    print("\n" + "=" * 60)
    print(f"  Clean baseline complete: {accuracy*100:.2f}% accuracy")
    print("=" * 60)


if __name__ == '__main__':
    main()
