"""
Script: VIT/train_swin.py
Purpose: Fine-tune Swin-Base on ImageNet-100 (100-class head) for adversarial
         robustness experiments.

Mixed precision: Uses torch.cuda.amp (GradScaler + autocast) — ~1.5–2× speedup
on Ampere / Ada Lovelace GPUs (RTX 3080, RTX 4060).

IMPORTANT for adversarial attacks:
  autocast is used for the forward/backward pass during training ONLY.
  Attack functions in swin_utils.py do NOT use autocast — gradients are
  always computed in float32 to avoid precision loss.

Inputs:
  - ImageNet100_Training/data/{train,val}/{synset}/*.JPEG
  - microsoft/swin-base-patch4-window7-224 (HuggingFace pretrained)

Outputs:
  - Swin_Training/checkpoints/swin_base_imagenet100_best.pt
  - Swin_Training/checkpoints/swin_base_imagenet100_final.pt
  - Swin_Training/clean_baselines/swin_base_clean.json

Run from project root:
    python VIT/train_swin.py [--epochs 20] [--batch-size 32] [--lr 1e-4]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import SwinForImageClassification, AutoImageProcessor

# ── add project root to sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "VIT" / "src"))
from swin_utils import (
    seed_everything, get_device, ImageNet100Dataset, load_class_names
)

# ── Constants ──
SEED            = 42
CHECKPOINT      = "microsoft/swin-base-patch4-window7-224"
NUM_CLASSES     = 100
DATA_DIR        = PROJECT_ROOT / "ImageNet100_Training" / "data"
OUT_DIR         = PROJECT_ROOT / "Swin_Training"
CKPT_DIR        = OUT_DIR / "checkpoints"
BASELINES_DIR   = OUT_DIR / "clean_baselines"

# Default hyperparameters (override via CLI)
DEFAULT_EPOCHS     = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-4
DEFAULT_FROZEN_EP  = 5       # Epochs to train only the new head
MIN_LR             = 1e-6


# ────────────────────────────────────────────────────────────────
# MODEL  — replace 1000-class head with 100-class head
# ────────────────────────────────────────────────────────────────

def build_model(num_classes: int, checkpoint: str, device: torch.device):
    """Load pretrained Swin-Base and replace classification head."""
    print(f"  Loading {checkpoint} ...")
    model = SwinForImageClassification.from_pretrained(checkpoint)

    # Replace the head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.classifier.weight, std=0.02)
    nn.init.zeros_(model.classifier.bias)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total | {trainable:,} trainable")
    return model


def freeze_backbone(model):
    """Freeze all layers except the new classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen backbone. Trainable params: {trainable:,} (head only)")


def unfreeze_backbone(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfrozen backbone. Trainable params: {trainable:,}")


# ────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, epoch):
    """One training epoch with mixed-precision autocast."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_i, (pixel_values, labels) in enumerate(loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── mixed precision forward ──
        with autocast():
            outputs = model(pixel_values=pixel_values)
            loss    = F.cross_entropy(outputs.logits, labels)

        # ── scaled backward ──
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * pixel_values.size(0)
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += pixel_values.size(0)

        if (batch_i + 1) % 50 == 0:
            print(f"    Epoch {epoch} [{batch_i+1}/{len(loader)}] "
                  f"loss={total_loss/total:.4f} acc={correct/total:.3f}", end="\r")

    print()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Clean accuracy evaluation. No autocast needed — just inference."""
    model.eval()
    correct = 0
    total   = 0
    losses  = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=pixel_values)
        loss    = F.cross_entropy(outputs.logits, labels)
        losses.append(loss.item())

        preds    = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += pixel_values.size(0)

    return np.mean(losses), correct / total


# ────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scaler, epoch, val_acc, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
    }, path)
    print(f"  Saved checkpoint: {path.name} (val_acc={val_acc:.4f})")


def save_clean_baseline(model, val_loader, class_mapping: dict, device, out_path: Path):
    """Run clean evaluation and save per-class accuracy to JSON."""
    model.eval()
    synsets      = sorted(class_mapping.keys(), key=lambda s: class_mapping[s]["index"])
    num_classes  = len(synsets)
    per_cls_correct = [0] * num_classes
    per_cls_total   = [0] * num_classes
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels_t     = labels.to(device)
            outputs      = model(pixel_values=pixel_values)
            probs        = F.softmax(outputs.logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_t.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

            for i in range(len(labels_t)):
                lbl = labels_t[i].item()
                per_cls_total[lbl]   += 1
                if preds[i].item() == lbl:
                    per_cls_correct[lbl] += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = float((all_preds == all_labels).mean())

    per_class_acc = {}
    for i, synset in enumerate(synsets):
        if per_cls_total[i] > 0:
            per_class_acc[class_mapping[synset]["human_name"]] = round(
                per_cls_correct[i] / per_cls_total[i], 4
            )

    result = {
        "model":          "swin-base-patch4-window7-224",
        "dataset":        "ImageNet-100",
        "accuracy":       round(accuracy, 4),
        "mean_confidence": round(float(np.mean(all_confs)), 4),
        "total_samples":  len(all_labels),
        "per_class_accuracy": per_class_acc,
        "timestamp":      datetime.now().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Clean baseline saved: {out_path}")
    return accuracy


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--frozen-epochs", type=int, default=DEFAULT_FROZEN_EP,
                        help="Epochs to train only the head before unfreezing backbone")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp",     action="store_true",
                        help="Disable mixed-precision training")
    args = parser.parse_args()

    seed_everything(SEED)
    device = get_device()
    use_amp = not args.no_amp and device.type == "cuda"

    print("=" * 60)
    print("  Swin-Base Fine-tuning on ImageNet-100")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  AMP       : {'enabled (float16 Tensor Cores)' if use_amp else 'disabled'}")
    print(f"  Epochs    : {args.epochs} (head-only: {args.frozen_epochs})")
    print(f"  Batch     : {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print()

    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("  Run:  python scripts/setup_imagenet100.py")
        sys.exit(1)

    # ── Load processor ──
    print("Loading processor and datasets...")
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

    train_loader = DataLoader(
        ImageNet100Dataset(DATA_DIR, split="train", processor=processor),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ImageNet100Dataset(DATA_DIR, split="val", processor=processor),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Load class mapping ──
    mapping_path = PROJECT_ROOT / "ImageNet100_Training" / "class_mapping.json"
    with open(mapping_path) as f:
        class_mapping = json.load(f)

    # ── Build model ──
    print("\nBuilding model...")
    model = build_model(NUM_CLASSES, CHECKPOINT, device)

    # ── Phase 1: train head only ──
    best_val_acc  = 0.0
    best_ckpt     = CKPT_DIR / "swin_base_imagenet100_best.pt"
    scaler        = GradScaler(enabled=use_amp)

    if args.frozen_epochs > 0:
        print(f"\n[Phase 1] Head-only training for {args.frozen_epochs} epochs...")
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.frozen_epochs, eta_min=MIN_LR
        )

        for epoch in range(1, args.frozen_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            print(f"  Epoch {epoch:2d}/{args.frozen_epochs} | "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scaler, epoch, val_acc, best_ckpt)

    # ── Phase 2: full fine-tuning ──
    remaining_epochs = args.epochs - args.frozen_epochs
    if remaining_epochs > 0:
        print(f"\n[Phase 2] Full fine-tuning for {remaining_epochs} epochs...")
        unfreeze_backbone(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr / 10, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=MIN_LR
        )

        for epoch in range(args.frozen_epochs + 1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            print(f"  Epoch {epoch:2d}/{args.epochs} | "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scaler, epoch, val_acc, best_ckpt)

    # ── Save final checkpoint ──
    final_ckpt = CKPT_DIR / "swin_base_imagenet100_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"\nFinal model saved: {final_ckpt}")
    print(f"Best val accuracy : {best_val_acc:.4f}")

    # ── Load best and save clean baseline ──
    print("\nSaving clean baseline from best checkpoint...")
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model"])

    baseline_path = BASELINES_DIR / "swin_base_clean.json"
    acc = save_clean_baseline(model, val_loader, class_mapping, device, baseline_path)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Final checkpoint  : {final_ckpt}")
    print(f"  Best checkpoint   : {best_ckpt}")
    print(f"  Clean baseline    : {baseline_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
