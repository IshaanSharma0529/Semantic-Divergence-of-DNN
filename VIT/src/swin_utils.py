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
from scipy.stats import entropy as scipy_entropy
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


# ────────────────────────────────────────────────────────────────
# DEEPFOOL ATTACK
# ────────────────────────────────────────────────────────────────

def deepfool_attack_swin(
    model: SwinForImageClassification,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    overshoot: float = 0.02,
    num_classes: int = 100,
    top_k_candidates: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepFool minimal-perturbation attack for HuggingFace Swin.

    Iteratively finds the nearest decision boundary and steps across it.
    Only evaluates gradients for the top-K candidate classes per iteration
    for efficiency (typically finds the same boundary as full evaluation).

    Args:
        model: Swin in eval mode
        pixel_values: [B, 3, 224, 224] ImageNet-normalized
        labels: [B] true class indices
        max_iter: max perturbation steps per sample
        overshoot: extra step beyond the boundary (0.02 = 2%)
        num_classes: total number of classes
        top_k_candidates: only consider top-K logit classes per step

    Returns:
        (adv_pixel_values [B, 3, 224, 224], l2_distances [B])
    """
    device = pixel_values.device
    data_min = pixel_values.min().item()
    data_max = pixel_values.max().item()
    model.eval()
    results, l2_dists = [], []

    for i in range(pixel_values.size(0)):
        x_orig = pixel_values[i:i+1].clone().detach()  # [1, 3, 224, 224]
        xi = x_orig.clone()
        true_lbl = int(labels[i].item())

        for _ in range(max_iter):
            xi_var = xi.detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(pixel_values=xi_var)
                logits = out.logits[0]  # [num_classes]

            pred = int(logits.argmax().item())
            if pred != true_lbl:
                break  # already fooled

            # Gradient of true-class logit
            with torch.enable_grad():
                g_true = torch.autograd.grad(
                    logits[true_lbl], xi_var,
                    create_graph=False, retain_graph=True
                )[0].detach()

            # Top-K candidates (exclude true class)
            topk_scores, topk_idx = logits.detach().topk(top_k_candidates + 1)
            candidate_classes = [int(k) for k in topk_idx.tolist() if k != true_lbl][:top_k_candidates]

            min_ratio = float('inf')
            best_pert = torch.zeros_like(xi)

            for k in candidate_classes:
                xi_k = xi.detach().requires_grad_(True)
                with torch.enable_grad():
                    out_k = model(pixel_values=xi_k).logits[0]
                    g_k = torch.autograd.grad(
                        out_k[k], xi_k, create_graph=False
                    )[0].detach()

                w = (g_k - g_true)
                f = float((out_k[k] - out_k[true_lbl]).detach())
                w_norm = float(w.norm()) + 1e-8
                ratio = abs(f) / w_norm
                if ratio < min_ratio:
                    min_ratio = ratio
                    best_pert = (abs(f) / (w_norm ** 2)) * w

            xi = (xi.detach() + (1 + overshoot) * best_pert).clamp(data_min, data_max)

        l2_dists.append(float((xi - x_orig).norm().item()))
        results.append(xi.detach())

    return torch.cat(results, dim=0), torch.tensor(l2_dists, device=device)


# ────────────────────────────────────────────────────────────────
# NORMALIZATION UTILITIES
# ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] raw images to ImageNet-normalized space.

    x: [B, 3, H, W] or [3, H, W] in [0, 1]
    Returns same shape, normalized.
    """
    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization back to [0, 1] space."""
    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation regularization.

    x: [B, 3, H, W] in [0, 1]
    Returns scalar.
    """
    diff_h = (x[..., 1:, :] - x[..., :-1, :]).abs().sum()
    diff_w = (x[..., :, 1:] - x[..., :, :-1]).abs().sum()
    return diff_h + diff_w


# ────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_swin_features(
    model: SwinForImageClassification,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Extract GAP features (pooler_output) from Swin: [B, 1024].

    These are the features fed directly into the classifier head.
    pixel_values should already be ImageNet-normalized.
    """
    outputs = model.swin(pixel_values=pixel_values)
    return outputs.pooler_output  # [B, 1024]


@torch.no_grad()
def extract_swin_spatial_features(
    model: SwinForImageClassification,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Extract spatial token features from Swin: [B, 49, 1024].

    Pre-pooling token features — 49 tokens, each covering a
    32×32-pixel receptive field in the 224×224 input.
    pixel_values should already be ImageNet-normalized.
    """
    outputs = model.swin(pixel_values=pixel_values)
    return outputs.last_hidden_state  # [B, 49, 1024]


# ────────────────────────────────────────────────────────────────
# FEATURE INVERSION
# ────────────────────────────────────────────────────────────────

class _SwinInverter:
    """Batch feature inverter for Swin Transformer (PyTorch).

    Optimizes N raw [0,1] images in parallel so that their
    ImageNet-normalized GAP features match the given target vectors.
    Uses Adam with TV regularization, same design as the TF CNN inverter.
    """

    def invert_batch(
        self,
        model: SwinForImageClassification,
        target_features: np.ndarray,   # (N, 1024) GAP feature targets
        init_images_01: np.ndarray,    # (N, H, W, 3) in [0,1], HWC
        device: torch.device,
        steps: int = 100,
        lr: float = 0.01,
        tv_weight: float = 1e-4,
    ) -> np.ndarray:                   # (N, H, W, 3) in [0,1], HWC
        """Optimize N images in parallel to match N target feature vectors."""
        # Convert HWC → CHW, numpy → tensor
        init_chw = init_images_01.transpose(0, 3, 1, 2).astype(np.float32)
        img_param = torch.nn.Parameter(
            torch.tensor(init_chw, device=device)
        )
        target_t = torch.tensor(target_features, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([img_param], lr=lr)

        prev_loss = float('inf')
        model.eval()
        for step in range(steps):
            optimizer.zero_grad()
            # Clamp to [0,1], then normalize for the model
            img_clamped = img_param.clamp(0.0, 1.0)
            pixel_values = normalize_imagenet(img_clamped)
            # Extract features (with grad)
            outputs = model.swin(pixel_values=pixel_values)
            feats = outputs.pooler_output  # [N, 1024]

            feat_loss = F.mse_loss(feats, target_t)
            tv_loss   = total_variation_loss(img_clamped)
            loss      = feat_loss + tv_weight * tv_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                img_param.clamp_(0.0, 1.0)

            if step % 10 == 0:
                lv = loss.item()
                if lv < 1e-4 or (prev_loss - lv) < 1e-7 * max(prev_loss, 1e-8):
                    break
                prev_loss = lv

        result = img_param.detach().cpu().numpy()   # [N, 3, H, W]
        return result.transpose(0, 2, 3, 1)         # [N, H, W, 3]


_swin_inverter = _SwinInverter()


def invert_features_batch_swin(
    model: SwinForImageClassification,
    target_features: np.ndarray,
    init_images_01: np.ndarray,
    device: torch.device,
    steps: int = 100,
    lr: float = 0.01,
    tv_weight: float = 1e-4,
) -> np.ndarray:
    """Invert N feature vectors to N images in parallel.

    Args:
        model: Swin in eval mode
        target_features: (N, 1024) GAP feature targets
        init_images_01: (N, H, W, 3) in [0, 1] HWC initialization
        device: torch device
        steps: max Adam steps
        lr: Adam learning rate
        tv_weight: total variation regularization weight

    Returns:
        (N, H, W, 3) images in [0, 1]
    """
    return _swin_inverter.invert_batch(
        model, target_features, init_images_01, device, steps, lr, tv_weight
    )


def invert_features_swin(
    model: SwinForImageClassification,
    target_feature: np.ndarray,
    init_image_01: np.ndarray,
    device: torch.device,
    steps: int = 100,
    lr: float = 0.01,
    tv_weight: float = 1e-4,
) -> np.ndarray:
    """Invert a single feature vector to an image. Calls invert_features_batch_swin."""
    return invert_features_batch_swin(
        model,
        target_feature[np.newaxis, :],
        init_image_01[np.newaxis, :],
        device, steps, lr, tv_weight,
    )[0]


# ────────────────────────────────────────────────────────────────
# ADAIN STYLE TRANSFER  (spatial, avoids 1-D degeneracy)
# ────────────────────────────────────────────────────────────────

def adain_style_transfer_swin(
    model: SwinForImageClassification,
    content_img_01: np.ndarray,                    # (H, W, 3) in [0,1]
    style_img_01: np.ndarray,                      # (H, W, 3) in [0,1]
    device: torch.device,
    alpha: float = 1.0,
    content_spatial_feat: Optional[np.ndarray] = None,  # (49, 1024) pre-computed
    style_spatial_feat: Optional[np.ndarray] = None,    # (49, 1024) pre-computed
    steps: int = 75,
    lr: float = 0.01,
    tv_weight: float = 1e-4,
) -> np.ndarray:
    """Spatial AdaIN style transfer using Swin token features.

    Uses pre-pooling token features [49, 1024] so AdaIN computes
    per-channel statistics across tokens (not a degenerate scalar op).
    The style target is then GAP'd for the inversion step.

    Returns:
        (H, W, 3) adversarial image in [0, 1]
    """
    model.eval()

    def _to_pixel_values(img_01: np.ndarray) -> torch.Tensor:
        t = torch.tensor(img_01.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32, device=device)
        return normalize_imagenet(t)  # [1, 3, 224, 224]

    if content_spatial_feat is None:
        with torch.no_grad():
            content_spatial_feat = (
                model.swin(pixel_values=_to_pixel_values(content_img_01))
                .last_hidden_state[0].cpu().numpy()
            )  # (49, 1024)

    if style_spatial_feat is None:
        with torch.no_grad():
            style_spatial_feat = (
                model.swin(pixel_values=_to_pixel_values(style_img_01))
                .last_hidden_state[0].cpu().numpy()
            )  # (49, 1024)

    # Per-channel statistics over 49 token positions
    c_mean = content_spatial_feat.mean(axis=0)        # (1024,)
    c_std  = content_spatial_feat.std(axis=0) + 1e-8  # (1024,)
    s_mean = style_spatial_feat.mean(axis=0)           # (1024,)
    s_std  = style_spatial_feat.std(axis=0) + 1e-8    # (1024,)

    # AdaIN: per-token, per-channel normalization with style stats
    adain_tokens = (content_spatial_feat - c_mean) / c_std * s_std + s_mean  # (49, 1024)
    blended_tokens = alpha * adain_tokens + (1 - alpha) * content_spatial_feat

    # GAP blended tokens → single 1024-d target for the inverter
    blended_gap = blended_tokens.mean(axis=0)  # (1024,)

    return invert_features_batch_swin(
        model,
        blended_gap[np.newaxis, :],         # (1, 1024)
        content_img_01[np.newaxis, :],      # (1, H, W, 3)
        device, steps, lr, tv_weight,
    )[0]  # (H, W, 3)


# ────────────────────────────────────────────────────────────────
# SEMANTIC STRUCTURE SCORE
# ────────────────────────────────────────────────────────────────

def compute_sss_from_confusion(confusion_matrix: np.ndarray) -> float:
    """Compute Semantic Structure Score from a confusion matrix.

    SSS = 1 - H(off_diagonal) / H_max
    where H is Shannon entropy of the off-diagonal element distribution.

    SSS ≈ 1: misclassifications concentrate on few target classes (semantic).
    SSS ≈ 0: uniform scatter across all classes (random noise).
    """
    num_classes = confusion_matrix.shape[0]
    mask = ~np.eye(num_classes, dtype=bool)
    off_diag = confusion_matrix[mask].flatten().astype(float)
    total_off = off_diag.sum()

    if total_off == 0:
        return 1.0

    probs = off_diag / total_off
    H     = float(scipy_entropy(probs + 1e-12, base=2))
    H_max = float(np.log2(len(off_diag)))

    return round(1.0 - (H / H_max), 6) if H_max > 0 else 1.0


def build_adversarial_confusion_swin(
    model: SwinForImageClassification,
    dataloader: DataLoader,
    device: torch.device,
    attack_fn,
    epsilon: float,
    num_classes: int = 100,
    **attack_kwargs,
) -> Tuple[np.ndarray, List[dict]]:
    """Build confusion matrix under adversarial attack.

    Returns:
        C: (num_classes, num_classes) integer confusion matrix — C[true, adv_pred]
        records: per-sample misclassification dicts (true_label, adv_pred, adv_confidence)
    """
    C = np.zeros((num_classes, num_classes), dtype=np.int32)
    records = []
    model.eval()

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels_t = (
            labels.to(device)
            if isinstance(labels, torch.Tensor)
            else torch.tensor(labels, device=device)
        )

        adv = attack_fn(model, pixel_values, labels_t, epsilon, **attack_kwargs)

        with torch.no_grad():
            adv_probs = F.softmax(model(pixel_values=adv).logits, dim=-1)
        adv_conf, adv_pred = adv_probs.max(dim=-1)

        for j in range(labels_t.size(0)):
            t = int(labels_t[j].item())
            p = int(adv_pred[j].item())
            C[t, p] += 1
            if t != p:
                records.append({
                    'true_label': t,
                    'adv_pred': p,
                    'adv_confidence': round(float(adv_conf[j].item()), 4),
                })

    return C, records
