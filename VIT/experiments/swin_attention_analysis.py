"""
Swin Transformer — Attention & Representation Analysis
=======================================================
ViT-specific analysis that CNNs cannot provide:

  1. Token drift per Swin stage (hidden_states comparison)
  2. Attention map visualization (clean vs adversarial)
  3. GradCAM saliency maps via pytorch-grad-cam

Outputs:
    Swin_Training/attention_analysis/token_drift.json
    Swin_Training/figures/token_drift_by_stage.png
    Swin_Training/figures/attention_maps_sample_*.png
    Swin_Training/figures/gradcam_clean_vs_adv_*.png

Run from project root:
    python VIT/experiments/swin_attention_analysis.py [--num-samples 10]
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.swin_utils import (
    load_swin_config, load_swin_model, seed_everything,
    get_device, PROJECT_ROOT, fgsm_attack_swin, ImageNet100Dataset,
)


# ────────────────────────────────────────────────────────────────
# 1. TOKEN DRIFT ANALYSIS
# ────────────────────────────────────────────────────────────────

def compute_token_drift(
    model, pixel_values_clean, pixel_values_adv, device
) -> dict:
    """Compute per-stage token drift between clean and adversarial inputs.

    Returns dict mapping stage index → mean L2 drift across all tokens.
    """
    model.eval()
    with torch.no_grad():
        clean_out = model(
            pixel_values=pixel_values_clean.to(device),
            output_hidden_states=True,
        )
        adv_out = model(
            pixel_values=pixel_values_adv.to(device),
            output_hidden_states=True,
        )

    drift_per_stage = {}
    for stage_i, (h_c, h_a) in enumerate(
        zip(clean_out.hidden_states, adv_out.hidden_states)
    ):
        # h_c, h_a shape: [batch, num_tokens, embed_dim]
        drift = (h_c - h_a).norm(dim=-1).mean().item()
        drift_per_stage[stage_i] = round(drift, 6)

    return drift_per_stage


def run_token_drift_analysis(model, dataset, device, num_samples, epsilon):
    """Aggregate token drift over multiple samples."""
    print("\n--- Token Drift Analysis ---")
    all_drifts = {}

    for i in range(min(num_samples, len(dataset))):
        pixel_values, label = dataset[i]
        pv_clean = pixel_values.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)

        # Generate adversarial
        pv_adv = fgsm_attack_swin(model, pv_clean, label_t, epsilon)

        drift = compute_token_drift(model, pv_clean, pv_adv, device)

        for stage, val in drift.items():
            if stage not in all_drifts:
                all_drifts[stage] = []
            all_drifts[stage].append(val)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    # Average
    avg_drift = {stage: round(np.mean(vals), 6) for stage, vals in all_drifts.items()}
    std_drift = {stage: round(np.std(vals), 6) for stage, vals in all_drifts.items()}

    print("\n  Stage-wise token drift (FGSM ε={:.3f}):".format(epsilon))
    for stage in sorted(avg_drift.keys()):
        print(f"    Stage {stage}: {avg_drift[stage]:.4f} ± {std_drift[stage]:.4f}")

    return {
        'epsilon': epsilon,
        'num_samples': min(num_samples, len(dataset)),
        'mean_drift_per_stage': avg_drift,
        'std_drift_per_stage': std_drift,
    }


def plot_token_drift(drift_data, save_path):
    """Plot token drift across Swin stages."""
    stages = sorted(drift_data['mean_drift_per_stage'].keys())
    means = [drift_data['mean_drift_per_stage'][s] for s in stages]
    stds = [drift_data['std_drift_per_stage'][s] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    stage_labels = [f"Stage {s}" for s in stages]
    bars = ax.bar(stage_labels, means, yerr=stds, capsize=5,
                  color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'],
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Swin Stage', fontsize=12)
    ax.set_ylabel('Mean Token Drift (L2)', fontsize=12)
    ax.set_title(f'Token Drift per Stage (FGSM ε={drift_data["epsilon"]})',
                 fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Token drift plot saved to {save_path}")


# ────────────────────────────────────────────────────────────────
# 2. ATTENTION MAP VISUALIZATION
# ────────────────────────────────────────────────────────────────

def visualize_attention_maps(model, dataset, device, num_samples, epsilon, fig_dir):
    """Generate clean vs adversarial attention map comparisons."""
    print("\n--- Attention Map Visualization ---")

    for i in range(min(num_samples, len(dataset), 5)):  # Max 5 examples
        pixel_values, label = dataset[i]
        pv_clean = pixel_values.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)

        pv_adv = fgsm_attack_swin(model, pv_clean, label_t, epsilon)

        with torch.no_grad():
            clean_out = model(pixel_values=pv_clean, output_attentions=True)
            adv_out = model(pixel_values=pv_adv, output_attentions=True)

        # Plot attention from last layer, averaged across heads
        n_layers = len(clean_out.attentions)
        last_attn_clean = clean_out.attentions[-1][0].mean(0).cpu().numpy()
        last_attn_adv = adv_out.attentions[-1][0].mean(0).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(last_attn_clean, cmap='viridis', aspect='auto')
        axes[0].set_title('Clean (last layer, head-avg)', fontsize=11)
        axes[0].set_xlabel('Key token')
        axes[0].set_ylabel('Query token')

        axes[1].imshow(last_attn_adv, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Adversarial ε={epsilon} (last layer, head-avg)', fontsize=11)
        axes[1].set_xlabel('Key token')
        axes[1].set_ylabel('Query token')

        plt.suptitle(f'Sample {i} (true label: {label})', fontsize=13)
        plt.tight_layout()
        save_path = fig_dir / f'attention_maps_sample_{i}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")


# ────────────────────────────────────────────────────────────────
# 3. GRADCAM
# ────────────────────────────────────────────────────────────────

class HFSwinWrapper(torch.nn.Module):
    """Wrapper so pytorch-grad-cam gets raw logits tensor."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        return self.model(pixel_values=x).logits


def run_gradcam(model, dataset, device, num_samples, epsilon, fig_dir):
    """Generate GradCAM visualizations: clean vs adversarial."""
    print("\n--- GradCAM Visualization ---")
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("  pytorch-grad-cam not installed — skipping GradCAM.")
        print("  Install with: pip install grad-cam")
        return

    wrapped = HFSwinWrapper(model)
    wrapped.eval()

    # Target the last layernorm of the Swin backbone
    target_layers = [wrapped.model.swin.layernorm]
    cam = GradCAM(model=wrapped, target_layers=target_layers)

    for i in range(min(num_samples, len(dataset), 5)):
        pixel_values, label = dataset[i]
        pv_clean = pixel_values.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)

        pv_adv = fgsm_attack_swin(model, pv_clean, label_t, epsilon)

        # GradCAM on clean
        grayscale_clean = cam(input_tensor=pv_clean)
        # GradCAM on adversarial
        grayscale_adv = cam(input_tensor=pv_adv)

        # Denormalize for overlay
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img_clean = (pv_clean[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        img_adv = (pv_adv[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(img_clean)
        axes[0, 0].set_title('Clean Image', fontsize=11)
        axes[0, 0].axis('off')

        cam_clean = show_cam_on_image(img_clean, grayscale_clean[0], use_rgb=True)
        axes[0, 1].imshow(cam_clean)
        axes[0, 1].set_title('Clean GradCAM', fontsize=11)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(img_adv)
        axes[1, 0].set_title(f'Adversarial (ε={epsilon})', fontsize=11)
        axes[1, 0].axis('off')

        cam_adv = show_cam_on_image(img_adv, grayscale_adv[0], use_rgb=True)
        axes[1, 1].imshow(cam_adv)
        axes[1, 1].set_title('Adversarial GradCAM', fontsize=11)
        axes[1, 1].axis('off')

        plt.suptitle(f'Sample {i} — GradCAM (true label: {label})', fontsize=13)
        plt.tight_layout()
        save_path = fig_dir / f'gradcam_clean_vs_adv_{i}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")

    cam = None  # release


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Swin attention analysis')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to analyze (default: from config)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='FGSM epsilon for adversarial comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("Swin-Tiny — Attention & Representation Analysis")
    print("=" * 60)

    config = load_swin_config()
    seed_everything(config['evaluation']['seed'])
    device = get_device()

    # Load model
    print("\n[1/5] Loading model...")
    model, processor = load_swin_model(
        checkpoint=config['model']['hf_checkpoint'],
        device=device,
    )

    # Load dataset (without DataLoader — we need per-sample access)
    print("\n[2/5] Loading dataset...")
    data_dir = PROJECT_ROOT / config['dataset']['data_dir']
    dataset = ImageNet100Dataset(
        data_dir=data_dir,
        split=config['dataset']['splits']['test'],
        processor=processor,
    )

    num_samples = args.num_samples or config['attention_analysis']['num_samples']
    epsilon = args.epsilon

    # Create output dirs
    fig_dir = PROJECT_ROOT / config['output']['figures']
    analysis_dir = PROJECT_ROOT / config['output']['attention_analysis']
    fig_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── Token Drift ──
    print("\n[3/5] Token drift analysis...")
    drift_data = run_token_drift_analysis(
        model, dataset, device, num_samples, epsilon
    )
    drift_file = analysis_dir / 'token_drift.json'
    with open(drift_file, 'w') as f:
        json.dump(drift_data, f, indent=2)
    print(f"  Saved to {drift_file}")

    plot_token_drift(drift_data, fig_dir / 'token_drift_by_stage.png')

    # ── Attention Maps ──
    print("\n[4/5] Attention map visualization...")
    visualize_attention_maps(
        model, dataset, device, num_samples, epsilon, fig_dir
    )

    # ── GradCAM ──
    print("\n[5/5] GradCAM visualization...")
    run_gradcam(model, dataset, device, num_samples, epsilon, fig_dir)

    print("\n" + "=" * 60)
    print("  Attention analysis complete.")
    print(f"  Figures: {fig_dir}")
    print(f"  Data:    {analysis_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
