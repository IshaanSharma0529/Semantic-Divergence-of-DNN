# 🧠 MASTER PROJECT PROMPT FOR PAPER 1 (ACL-ORIENTED)

## Context

I am working on a research paper (**Paper 1**) whose goal is to systematically evaluate how adversarial perturbations applied at the visual level propagate across different model hierarchies—from **Convolutional Neural Networks (CNNs)** to **Vision Transformers (ViTs)** to **Vision–Language Models (VLMs)**—and induce **semantic and linguistic failures** in downstream multimodal reasoning tasks.

This project is intended for submission to **ACL Findings / EMNLP Findings**, so the emphasis is on **language-level failures caused by vision perturbations**, not merely image classification accuracy.

---

## 🔹 Models Under Study

### CNNs  

*(Pixel- and feature-level robustness baselines)*

- VGG19  
- ResNet50  
- DenseNet121  

### Vision Transformers (ViTs)  

*(Token- and attention-level robustness)*

- Swin Transformer  
- SAM (Segment Anything Model)  

### Vision–Language Models (VLMs)  

*(Semantic and linguistic robustness — core focus)*

- CLIP  
- BLIP  
- PaLI-Gemma  
- LLaVA-1.6  

CNNs and ViTs are used primarily to analyze **representation instability**, while VLMs are the **core evaluation target**, where failures are measured in terms of **language outputs**.

---

## 🔹 Threat Model

- **Attack type:** Evasion attacks (test-time only)  
- **Attacker capability:**
  - White-box access for CNNs and ViTs  
  - Black-box / surrogate-based access for VLMs  
- **Attack surface:** Image only  
- **Prompt:** Remains unchanged  
- **Goal:** Induce failures in multimodal language reasoning  

This threat model reflects realistic misuse scenarios where visual content is adversarially manipulated to corrupt downstream AI reasoning.

---

## 🔹 Adversarial Attacks Considered

### Classical Gradient-Based Attacks

- FGSM (single-step)
- DeepFool (minimal perturbation)
- Universal Adversarial Perturbations (dataset-level noise)

### Generative AI–Based Attacks

- GAN-based adversarial image generation
- Autoencoder / VAE-based latent-space perturbations
- *(Optional future extension: diffusion-based attacks)*

Generative attacks are emphasized because they produce **natural-looking perturbations** that transfer more effectively to **Vision–Language Models**.

---

## 🔹 Evaluation Objectives

### For CNNs & ViTs *(Supporting Analysis)*

- Accuracy degradation
- Feature instability
- Attention / token drift
- Saliency map and heatmap analysis

These models are **not the final goal**, but are used to understand **where and how perturbations distort representations**.

---

### For Vision–Language Models *(Core Contribution)*

Evaluation focuses on **language-level failures**, not accuracy alone:

- Hallucination rate
- Wrong entity grounding
- Attribute errors (color, count, size, location)
- Instruction-following violations
- Overconfident incorrect responses
- Partial or inconsistent answers

A **linguistic error taxonomy** is used to systematically categorize failures.

---

## 🔹 Metrics

- Standard robustness metrics (accuracy drop, fooling rate)
- Semantic consistency metrics (CLIP similarity, caption drift)
- Image quality metrics for generative attacks (FID)
- Qualitative analysis via attention maps and saliency visualizations

---

## 🔹 Paper 1 Scope (IMPORTANT)

This paper is **purely evaluative and analytical**.

It:

- ❌ DOES NOT propose new attacks  
- ❌ DOES NOT introduce adversarial watermarking  
- ❌ DOES NOT include defense mechanisms  

Adversarial watermarking is mentioned **only as future work**, based on insights from robustness propagation.

---

## 🔹 High-Level Research Questions

1. How do adversarial perturbations at the pixel level propagate through CNN and ViT representations?
2. Why do generative adversarial attacks transfer more effectively to VLMs than gradient-based attacks?
3. How do failures in vision representations manifest as linguistic and semantic errors in multimodal models?
4. Are certain architectures (e.g., SAM-based pipelines) more vulnerable to grounding failures?

---

## 🔹 Future Direction (Context Only)

Insights from this paper will later be used to design **adversarial watermarking mechanisms for multimodal models**, but **no watermarking methods or objectives are included in this paper**.

---

## 🔹 Expectation from Copilot / Assistant

When generating code, experiments, analysis, or text:

- Prioritize **clarity, reproducibility, and evaluation rigor**
- Treat **VLM language failures** as the primary signal
- Use CNNs and ViTs as **supporting analysis layers**
- Avoid introducing defenses or watermarking prematurely
- Align explanations and outputs with **ACL-style research writing**
