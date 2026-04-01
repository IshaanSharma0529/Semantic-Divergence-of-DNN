"""
Swin Transformer utilities for the CrossVision-Attacks project.

PyTorch/HuggingFace counterpart to shared_utils.py (which is TensorFlow-based).
Provides model loading, data loading, attack functions, and helpers
specifically for Swin Transformer evaluation on ImageNet-100.
"""

import os, json, yaml, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import SwinForImageClassification, AutoImageProcessor

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # VIT/src → VIT → project root
VIT_DIR = PROJECT_ROOT / 'VIT'
CONFIG_FILE = VIT_DIR / 'configs' / 'swin_config.yaml'


# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

def load_swin_config(config_path: Optional[Path] = None) -> dict:
    """Load Swin configuration from YAML."""
    path = config_path or CONFIG_FILE
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ────────────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────
# DEVICE
# ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────

def load_swin_model(
    checkpoint: str = "microsoft/swin-tiny-patch4-window7-224",
    device: Optional[torch.device] = None,
) -> Tuple[SwinForImageClassification, AutoImageProcessor]:
    """Load pretrained Swin model and image processor from HuggingFace.

    Returns:
        (model, processor) — model on device in eval mode, processor for
        image preprocessing.
    """
    device = device or get_device()
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SwinForImageClassification.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {checkpoint}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Classes: {model.config.num_labels}")
    print(f"    Device: {device}")
    return model, processor


# ────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────

class ImageNet100Dataset(Dataset):
    """PyTorch Dataset for ImageNet-100 prepared by prepare_imagenet100.py.

    Directory structure expected:
        data_dir/{split}/{synset_id}/image.JPEG

    The dataset maps each synset to [0, num_classes) using sorted order,
    matching the mapping from prepare_imagenet100.py.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = 'test',
        processor: Optional[AutoImageProcessor] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir) / split
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.data_dir}\n"
                f"Run scripts/prepare_imagenet100.py first."
            )

        self.processor = processor

        # Discover classes (sorted synsets — same order as prepare_imagenet100)
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths + labels
        self.samples: List[Tuple[Path, int]] = []
        valid_ext = {'.jpeg', '.jpg', '.png', '.JPEG'}
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix in valid_ext:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  ImageNet100Dataset({split}): {len(self.samples)} images, "
              f"{len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].squeeze(0)  # [3, 224, 224]
        else:
            # Fallback: manual resize + normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)

        return pixel_values, label


def get_dataloader(
    data_dir: str | Path,
    split: str = 'test',
    processor: Optional[AutoImageProcessor] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for a given split."""
    dataset = ImageNet100Dataset(
        data_dir=data_dir,
        split=split,
        processor=processor,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ────────────────────────────────────────────────────────────────
# CLASS NAME MAPPING
# ────────────────────────────────────────────────────────────────

def load_class_names(config: dict) -> List[str]:
    """Load human-readable class names from class_names.txt."""
    path = PROJECT_ROOT / config['dataset']['class_names_file']
    if path.exists():
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback: use synset IDs from the data directory
    data_dir = PROJECT_ROOT / config['dataset']['data_dir'] / 'test'
    return sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])


def load_imagenet1k_labels(model) -> Dict[int, str]:
    """Get ImageNet-1K label mapping from the HF model config."""
    return model.config.id2label


# ────────────────────────────────────────────────────────────────
# ATTACK FUNCTIONS  (PyTorch / HF Swin)
# ────────────────────────────────────────────────────────────────

def fgsm_attack_swin(
    model: SwinForImageClassification,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """FGSM attack on HuggingFace Swin model.

    Args:
        model: HF SwinForImageClassification (eval mode OK, we need grads on input)
        pixel_values: [B, 3, 224, 224] normalized input
        labels: [B] integer labels
        epsilon: perturbation budget

    Returns:
        adv_pixel_values: [B, 3, 224, 224]
    """
    pixel_values = pixel_values.clone().detach().requires_grad_(True)
    outputs = model(pixel_values=pixel_values)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()

    adv = pixel_values + epsilon * pixel_values.grad.sign()
    adv = adv.clamp(pixel_values.min().item(), pixel_values.max().item())
    return adv.detach()


def pgd_attack_swin(
    model: SwinForImageClassification,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    step_size: Optional[float] = None,
) -> torch.Tensor:
    """PGD attack (Madry et al.) on HuggingFace Swin model.

    Args:
        model: HF SwinForImageClassification
        pixel_values: [B, 3, 224, 224] normalized input
        labels: [B] integer labels
        epsilon: perturbation budget (L∞)
        steps: number of PGD steps
        step_size: per-step size (default: epsilon/4)

    Returns:
        adv_pixel_values: [B, 3, 224, 224]
    """
    if step_size is None:
        step_size = epsilon / 4.0

    orig = pixel_values.clone().detach()
    noise = torch.empty_like(pixel_values).uniform_(-epsilon, epsilon)
    adv = (orig + noise).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(pixel_values=adv)
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()

        with torch.no_grad():
            adv = adv + step_size * adv.grad.sign()
            # Project back into L∞ ball
            perturbation = (adv - orig).clamp(-epsilon, epsilon)
            adv = (orig + perturbation).detach()

    return adv


# ────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_clean(
    model: SwinForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
) -> dict:
    """Evaluate clean accuracy over dataloader.

    Returns dict with: accuracy, top5_accuracy, per_class_accuracy,
    per_class_counts, mean_confidence, predictions, labels.
    """
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels_t = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        labels_t = labels_t.to(device)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        # Top-1
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_t.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

        for i in range(len(labels_t)):
            lbl = labels_t[i].item()
            per_class_total[lbl] += 1
            if preds[i].item() == lbl:
                per_class_correct[lbl] += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    mean_conf = np.mean(all_confs)

    per_class_acc = {}
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc[c] = float(per_class_correct[c] / per_class_total[c])

    return {
        'accuracy': float(accuracy),
        'mean_confidence': float(mean_conf),
        'total_samples': len(all_labels),
        'per_class_accuracy': per_class_acc,
    }


def evaluate_under_attack(
    model: SwinForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    attack_fn,
    epsilon: float,
    **attack_kwargs,
) -> dict:
    """Evaluate model under an adversarial attack.

    Returns dict with: clean_accuracy, adv_accuracy, accuracy_drop,
    fooling_rate, mean_clean_conf, mean_adv_conf.
    """
    model.eval()
    total = 0
    correct_clean = 0
    correct_adv = 0
    fooled = 0
    clean_confs, adv_confs = [], []

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels_t = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        labels_t = labels_t.to(device)
        bs = pixel_values.size(0)

        # Clean predictions
        with torch.no_grad():
            clean_out = model(pixel_values=pixel_values)
            clean_probs = F.softmax(clean_out.logits, dim=-1)
            clean_conf, clean_preds = clean_probs.max(dim=-1)

        # Adversarial
        adv_images = attack_fn(
            model, pixel_values, labels_t, epsilon, **attack_kwargs
        )

        with torch.no_grad():
            adv_out = model(pixel_values=adv_images)
            adv_probs = F.softmax(adv_out.logits, dim=-1)
            adv_conf, adv_preds = adv_probs.max(dim=-1)

        correct_clean += (clean_preds == labels_t).sum().item()
        correct_adv += (adv_preds == labels_t).sum().item()
        fooled += (adv_preds != clean_preds).sum().item()
        clean_confs.extend(clean_conf.cpu().tolist())
        adv_confs.extend(adv_conf.cpu().tolist())
        total += bs

    clean_acc = correct_clean / total
    adv_acc = correct_adv / total

    return {
        'epsilon': epsilon,
        'clean_accuracy': clean_acc,
        'adv_accuracy': adv_acc,
        'accuracy_drop': clean_acc - adv_acc,
        'fooling_rate': fooled / total,
        'mean_clean_conf': float(np.mean(clean_confs)),
        'mean_adv_conf': float(np.mean(adv_confs)),
        'total_samples': total,
    }
