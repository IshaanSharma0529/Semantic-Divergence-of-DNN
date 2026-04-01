# 🪟 Swin Transformer — Course of Action

## Context

Swin Transformer is the first ViT we are implementing in the **CNN → ViT → VLM** adversarial propagation pipeline.

In this project Swin sits in the **supporting analysis** layer:
- It receives the **same adversarial images** already generated against CNNs (FGSM, DeepFool, UAP, GAN/VAE)
- We measure how well those attacks **transfer** (white-box on Swin, black-box from CNNs)
- We extract **attention maps & token representations** to show *where* perturbations cause instability
- This feeds directly into the VLM failure analysis (the core ACL contribution)

---

## 🔧 Step 1 — Environment & Model Setup

### 1.1 Install / verify dependencies

```bash
# Activate your venv first
source .venv/bin/activate

pip install transformers accelerate  # HuggingFace ecosystem
pip install torch torchvision        # already in requirement.txt
```

### 1.2 Choose a Swin variant

| HF Checkpoint | Params | Recommended use |
|---|---|---|
| `microsoft/swin-tiny-patch4-window7-224` | ~28M | Fast iteration / ablations |
| `microsoft/swin-base-patch4-window7-224` | ~88M | **← Use this for paper results** |
| `microsoft/swin-large-patch4-window7-224` | ~197M | If compute allows |

Use **`microsoft/swin-base-patch4-window7-224`** — pretrained on ImageNet-22k and fine-tuned on ImageNet-1k.

### 1.3 Load via HuggingFace Transformers

```python
from transformers import SwinForImageClassification, AutoImageProcessor
import torch

CKPT = "microsoft/swin-base-patch4-window7-224"

processor = AutoImageProcessor.from_pretrained(CKPT)
model = SwinForImageClassification.from_pretrained(CKPT)
model.eval()
```

> `AutoImageProcessor` handles resizing, normalization, and tensor conversion — no manual transforms needed.
> Use `output_attentions=True` and `output_hidden_states=True` in the forward pass to get full internal state access.

---

## 📦 Step 2 — Dataset Prep (Caltech-101, same split as CNNs)

```python
# Reuse frozen_split_indices.json from Model Training/
import json
from PIL import Image
from transformers import AutoImageProcessor

with open("Model Training/frozen_split_indices.json") as f:
    split = json.load(f)

processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Usage per image:
# inputs = processor(images=pil_image, return_tensors="pt")
# pixel_values = inputs["pixel_values"]   # shape: [1, 3, 224, 224]
```

> ⚠️ **Critical:** Use the **exact same** `frozen_split_indices.json` and random seeds as the CNN experiments so transfer-attack comparisons are apples-to-apples.

---

## ✅ Step 3 — Clean Baseline Evaluation

Run inference on the clean Caltech-101 test set and record:

| Metric | Target |
|---|---|
| Top-1 Accuracy | ≥ 85% (pretrained Swin) |
| Per-class accuracy | Save for later confusion analysis |
| Inference time | Log for Table footnote |

Save results to `Model Training/clean_baselines/swin_base_clean.json`.

---

## ⚔️ Step 4 — Adversarial Attack Evaluation

### 4.1 Load pre-generated adversarial bank

The adversarial images built by `experiments/build_adversarial_bank.py` should already exist.  
Run Swin inference over:

- `fgsm_ε=0.01 / 0.03 / 0.05` images
- `deepfool` images
- `UAP` images
- `GAN / VAE` adversarial images

### 4.2 White-box FGSM directly on Swin (needed for transfer matrix)

```python
import torch, torch.nn.functional as F

def fgsm_swin(model, pixel_values, labels, eps):
    pixel_values = pixel_values.clone().requires_grad_(True)
    outputs = model(pixel_values=pixel_values)   # HF forward
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()
    adv = (pixel_values + eps * pixel_values.grad.sign()).clamp(0, 1)
    return adv.detach()
```

> Run at ε ∈ {0.01, 0.03, 0.05} — **same ε values as CNN attacks** so the transfer matrix rows are directly comparable.

### 4.3 Record for each attack

- Adversarial accuracy
- Accuracy drop Δ = clean − adversarial
- Fooling rate

Save to `experiments/transfer_attack_matrix.py` output (Swin row).

---

## 🔍 Step 5 — Representation & Attention Analysis

This is the key **ViT-specific analysis** that CNNs cannot provide.

### 5.1 Token instability (CLS token drift)

```python
with torch.no_grad():
    clean_out = model(
        pixel_values=x_clean,
        output_hidden_states=True,
        output_attentions=True
    )
    adv_out = model(
        pixel_values=x_adv,
        output_hidden_states=True,
        output_attentions=True
    )

# Token drift per Swin stage (reshaped spatial tokens)
for stage_i, (h_c, h_a) in enumerate(
        zip(clean_out.hidden_states, adv_out.hidden_states)):
    drift = (h_c - h_a).norm(dim=-1).mean()
    print(f"Stage {stage_i} | token drift = {drift:.4f}")
```

### 5.2 Attention map visualization

```python
# clean_out.attentions → list of tensors, one per attention layer
# Shape per layer: [batch, heads, tokens, tokens]
attn_clean = clean_out.attentions
attn_adv   = adv_out.attentions

import matplotlib.pyplot as plt

for layer_i, (a_c, a_a) in enumerate(zip(attn_clean, attn_adv)):
    avg_c = a_c[0].mean(0).cpu().numpy()   # head-averaged, first sample
    avg_a = a_a[0].mean(0).cpu().numpy()
    # save / plot avg_c and avg_a as heatmaps
```

### 5.3 Window-level attention collapse

Swin uses **shifted-window attention** — adversarial patches that span window boundaries are particularly effective. Note in qualitative analysis whether the attack disrupts cross-window information flow.

---

## 📊 Step 6 — Explainability (Saliency Maps)

Use **GradCAM** or **Attention Rollout** via `pytorch-grad-cam` — it supports HF Swin directly:

```bash
pip install grad-cam          # pytorch-grad-cam by Jacob Gildenblat
```

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# For HF SwinForImageClassification, target the last layernorm before the classifier
target_layers = [model.swin.layernorm]

# Wrapper needed because HF models return an object, not a raw tensor
class HFSwinWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    def forward(self, x):
        return self.model(pixel_values=x).logits

wrapped = HFSwinWrapper(model)
cam = GradCAM(model=wrapped, target_layers=[wrapped.model.swin.layernorm])
grayscale_cam = cam(input_tensor=x_clean)
```

Save side-by-side `clean | adversarial` GradCAM figures → goes into **Section J (Figures)** of the paper.

---

## 🔄 Step 7 — Transfer Attack Matrix (Swin Row)

Update `experiments/transfer_attack_matrix.py` to include Swin.

The matrix should look like:

|  | VGG19 | ResNet50 | DenseNet121 | **Swin-B** |
|---|---|---|---|---|
| FGSM (CNN-sourced) | — | — | — | **record here** |
| DeepFool (CNN-sourced) | — | — | — | **record here** |
| UAP (CNN-sourced) | — | — | — | **record here** |
| GAN/VAE | — | — | — | **record here** |
| FGSM (Swin-sourced) | record | record | record | — |

The **Swin-sourced row** shows how well ViT-generated adversarial examples fool CNNs (and eventually VLMs).

---

## 💾 Step 8 — Save Everything

```
Model Training/
  clean_baselines/
    swin_base_clean.json          ← clean accuracy + per-class
  swin_adversarial_results.json   ← per-attack accuracy/drop/fooling-rate
  figures/
    swin_gradcam_clean.png
    swin_gradcam_adv.png
    swin_attention_clean.png
    swin_attention_adv.png
    swin_token_drift_by_stage.png
```

---

## 📋 Checklist (tick as you go)

- [ ] Dependencies installed (`transformers`, `accelerate`, `grad-cam`)
- [ ] Model loaded, clean baseline recorded
- [ ] Adversarial bank images run through Swin (FGSM, DeepFool, UAP)
- [ ] GAN/VAE adversarial images run through Swin
- [ ] White-box FGSM generated directly on Swin
- [ ] Transfer matrix Swin row filled
- [ ] Token drift (CLS) per stage computed and saved
- [ ] Attention maps (clean vs adversarial) saved
- [ ] GradCAM / Attention Rollout figures saved
- [ ] All JSON results committed to repo
- [ ] `Experiment_Checklist.md` ViT section updated

---

## 🧭 What Comes After Swin

Once Swin is done, the next model in the pipeline is:

1. **SAM** (Segment Anything Model) — segmentation-based representation analysis
2. **CLIP** → **BLIP** → **PaLI-Gemma** → **LLaVA-1.6** — the VLM core

The adversarial images generated here (both CNN-sourced **and** Swin-sourced) will be used as inputs to the VLMs in a black-box / surrogate transfer setting.
