# ✅ EXPERIMENT CHECKLIST — PAPER 1 (ACL-ORIENTED)

## Goal

Evaluate how adversarial visual perturbations propagate from **CNNs → ViTs → VLMs** and cause **semantic and linguistic failures**.

---

## 🔹 A. DATASET & TASK SETUP (FOUNDATION)

### ☐ Dataset Selection

- ☐ Caltech101 (for CNN & ViT baselines)
- ☐ Image–text datasets for VLMs (e.g., COCO, VQA-style datasets)
- ☐ Ensure the **same images** flow through CNN → ViT → VLM where possible

### ☐ Prompt Design (VLMs)

- ☐ Fixed prompts (no prompt attack)
- ☐ Multiple prompt types:
  - ☐ Captioning
  - ☐ VQA
  - ☐ Instruction-following  
    *(e.g., “Describe…”, “Count…”, “Locate…”)*
- ☐ Prompts logged and versioned

### ☐ Train / Test Split

- ☐ Clean evaluation set
- ☐ Attacked evaluation set
- ☐ Same random seed across models

---

## 🔹 B. MODEL SETUP (NO CONFUSION ALLOWED)

### CNNs

- ☐ VGG19 (pretrained, frozen)
- ☐ ResNet50 (pretrained, frozen)
- ☐ DenseNet121 (pretrained, frozen)

### Vision Transformers (ViTs)

- ☐ Swin Transformer
- ☐ SAM *(segmentation output used for downstream analysis)*

### Vision–Language Models (VLMs)

- ☐ CLIP
- ☐ BLIP
- ☐ PaLI-Gemma
- ☐ LLaVA-1.6

### ☐ Versioning & Configuration

- ☐ Model versions documented
- ☐ Checkpoints recorded
- ☐ Inference settings fixed  
  *(temperature, max tokens, decoding strategy)*

---

## 🔹 C. THREAT MODEL (REVIEWER MUST UNDERSTAND)

### ☐ Clearly Defined

- ☐ Evasion attacks only (test-time)
- ☐ Image-only perturbation
- ☐ Prompt unchanged
- ☐ White-box access (CNNs, ViTs)
- ☐ Black-box / surrogate access (VLMs)

### ☐ Perturbation Constraints

- ☐ ε values specified
- ☐ Norm used (L∞ / L2)
- ☐ Imperceptibility verified visually

---

## 🔹 D. ADVERSARIAL ATTACK IMPLEMENTATION

### Classical Attacks

- ☐ FGSM
- ☐ DeepFool
- ☐ Universal Adversarial Perturbation (UAP)

For each attack:

- ☐ Correct implementation
- ☐ Hyperparameters logged
- ☐ Attack success verified on CNNs

### Generative AI Attacks

- ☐ GAN-based adversarial image generation
- ☐ VAE / AE latent-space perturbation
- ☐ Image realism verified

### ☐ Attack Transferability

- ☐ CNN → ViT
- ☐ ViT → VLM
- ☐ Generative attacks outperform gradient-based attacks

---

## 🔹 E. CNN & ViT EVALUATION (SUPPORTING EVIDENCE)

### ☐ Metrics

- ☐ Clean accuracy
- ☐ Adversarial accuracy
- ☐ Accuracy drop (%)

### ☐ Representation Analysis

- ☐ Feature drift (layer-wise)
- ☐ Token instability (ViTs)
- ☐ SAM mask distortion

### ☐ Explainability

- ☐ Saliency maps (clean vs adversarial)
- ☐ Attention heatmaps
- ☐ Qualitative comparisons saved

---

## 🔹 F. VLM EVALUATION (ACL CORE)

### ☐ Language Failure Taxonomy Defined

- ☐ Hallucination
- ☐ Wrong entity grounding
- ☐ Attribute errors (color, count, size)
- ☐ Instruction violation
- ☐ Partial compliance
- ☐ Overconfident incorrect answers

### ☐ Annotation Protocol

- ☐ Manual or semi-automatic labeling
- ☐ Clear definitions per category
- ☐ Representative examples saved

### ☐ Metrics

- ☐ Failure rate per category
- ☐ Overall semantic failure rate
- ☐ Comparison across attack types

### ☐ Qualitative Analysis

- ☐ Clean vs adversarial output pairs
- ☐ Failure explanations per example

---

## 🔹 G. GENERATIVE ATTACK QUALITY CHECK

### ☐ Image Realism

- ☐ FID score computed
- ☐ Visual inspection performed
- ☐ Human-imperceptibility confirmed

### ☐ Transferability Proof

- ☐ Same image fools multiple VLMs
- ☐ Attack generalizes across different prompts

---

## 🔹 H. STATISTICAL RIGOR (VERY IMPORTANT)

- ☐ Multiple runs (≥ 3 seeds)
- ☐ Mean ± standard deviation reported
- ☐ Significance testing (t-test / Wilcoxon)
- ☐ Confidence intervals where applicable

---

## 🔹 I. ABLATION STUDIES (SMALL BUT POWERFUL)

- ☐ Attack strength vs language failure
- ☐ Gradient-based vs generative attacks
- ☐ CNN-robust ≠ VLM-robust demonstration
- ☐ SAM-based vs non-SAM pipelines

---

## 🔹 J. FIGURES & TABLES (REVIEWER BAIT)

### ☐ Required Plots

- ☐ Accuracy drop (CNN / ViT)
- ☐ Language failure rates (VLMs)
- ☐ Attack comparison heatmaps

### ☐ Visual Examples

- ☐ Clean vs adversarial images
- ☐ Corresponding VLM outputs
- ☐ Attention / saliency overlays

---

## 🔹 K. REPRODUCIBILITY CHECK

- ☐ Code runs end-to-end
- ☐ Configuration files saved
- ☐ Random seeds fixed
- ☐ README explains experiments
- ☐ Hardware and environment documented

---

## 🔹 L. SCOPE SAFETY CHECK (DO NOT FAIL THIS)

- ☐ No defenses included
- ☐ No watermarking methods
- ☐ No control objectives
- ☐ Watermarking mentioned **only as future work**

---

## 🔹 M. FINAL SANITY QUESTIONS (ASK YOURSELF)

- ☐ Is **language failure** the main result?
- ☐ Are CNNs / ViTs supporting evidence, not the headline?
- ☐ Can a linguistics reviewer understand the failure taxonomy?
- ☐ Is the contribution clear in **three sentences**?

**If YES to all → you are ready to write.**

---

## 🧠 Advisor-Level Honesty

If you complete **80–90%** of this checklist, your paper will:

- Feel **deliberate**, not exploratory  
- Be hard to dismiss as *“just benchmarking”*  
- Fit **ACL Findings / EMNLP Findings** cleanly
