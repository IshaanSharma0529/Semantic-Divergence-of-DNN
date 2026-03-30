# Model confidence, stability, and semantic understanding diverge

This project accompanies the paper:

> **Model confidence, stability, and semantic understanding diverge**  
> This paper demonstrates that adversarial vulnerability in CNN architectures is structured rather than random, governed by architectural connectivity patterns that determine decision boundary geometry. Models with dense connectivity exhibit boundary-proximate representations and early-onset confidence inversion, while residual architectures show intermediate stability despite similar clean accuracy. These findings establish that model confidence, adversarial stability, and representational robustness are empirically separable objectives — a result with direct implications for deploying vision models in security-critical systems.

---

## Project Overview

This repository provides a systematic evaluation of how adversarial perturbations at the visual level propagate through different model hierarchies — from **Convolutional Neural Networks (CNNs)** to **Vision Transformers (ViTs)** to **Vision–Language Models (VLMs)** — and induce semantic and linguistic failures in downstream multimodal reasoning tasks.

- **Emphasis:** Language-level failures caused by vision perturbations, not just image classification accuracy.
- **Target venues:** ACL Findings / EMNLP Findings.

---

## Models Under Study

- **CNNs:** VGG19, ResNet50, DenseNet121  
  (Pixel- and feature-level robustness baselines)
- **ViTs:** Swin Transformer, SAM (Segment Anything Model)  
  (Token- and attention-level robustness)
- **VLMs:** CLIP, BLIP, PaLI-Gemma, LLaVA-1.6  
  (Semantic and linguistic robustness — core focus)

CNNs and ViTs are used to analyze **representation instability**; VLMs are the **core evaluation target** (failures measured in language outputs).

---

## Threat Model

- **Attack type:** Evasion attacks (test-time only)
- **Attacker capability:**
  - White-box for CNNs/ViTs
  - Black-box/surrogate for VLMs
- **Attack surface:** Image only
- **Prompt:** Remains unchanged
- **Goal:** Induce failures in multimodal language reasoning

---

## Adversarial Attacks Considered

- **Classical Gradient-Based:**
  - FGSM (single-step)
  - DeepFool (minimal perturbation)
  - Universal Adversarial Perturbations
- **Generative AI–Based:**
  - GAN-based adversarial image generation
  - Autoencoder/VAE-based latent-space perturbations
  - *(Optional: diffusion-based attacks)*

Generative attacks are emphasized for producing **natural-looking perturbations** that transfer to VLMs.

---

## Evaluation Objectives

- **CNNs & ViTs:**
  - Accuracy degradation
  - Feature instability
  - Attention/token drift
  - Saliency/heatmap analysis
- **VLMs (Core):**
  - Hallucination rate
  - Wrong entity grounding
  - Attribute errors (color, count, size, location)
  - Instruction-following violations
  - Overconfident incorrect responses
  - Partial/inconsistent answers

A **linguistic error taxonomy** is used to categorize failures.

---

## Metrics

- Standard robustness metrics (accuracy drop, fooling rate)
- Semantic consistency (CLIP similarity, caption drift)
- Image quality (FID for generative attacks)
- Qualitative: attention maps, saliency visualizations

---

## Project Structure

- `Model Training/` — Training scripts, checkpoints, and adversarial banks
- `experiments/` — Scripts for building adversarial banks, transfer analysis, confusion analysis, etc.
- `src/` — Shared utilities and attack implementations
- `scripts/` — Data preparation and multi-seed runner scripts
- `configs/` — Training and experiment configuration files

---

## Installation & Requirements

Install dependencies (Python 3.8+ recommended):

```bash
pip install -r requirement.txt
```

---

## Running Experiments

- **Train/fine-tune models:**
  - See `Model Training/Model_Training.ipynb` and configs in `configs/`
- **Build adversarial bank:**
  - `python experiments/build_adversarial_bank.py [--save-images]`
- **Run multi-seed evaluation:**
  - `python scripts/multi_seed_runner.py`
- **FGSM/PGD/DeepFool attacks:**
  - See `Model Training/cnn-attacks.ipynb`
- **Generative attacks:**
  - See `Model Training/genai-attacks.ipynb`

---

## Notes

- This project is **purely evaluative and analytical** — no new attacks or defenses are proposed.
- Adversarial watermarking is mentioned only as future work.
- For full context, see `Project_Context.md`.

---

## Citation

If you use this code or analysis, please cite the paper:

> **Model confidence, stability, and semantic understanding diverge**

---

## Contact

For questions or collaborations, please open an issue or contact the authors.
