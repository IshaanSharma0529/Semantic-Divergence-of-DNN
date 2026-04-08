# SKILLS.md — MASTER REFERENCE FOR CLAUDE CODE

**Project:** Adversarial Robustness Analysis Across Vision Architectures  
**Paper 1 (IEEE TIFS):** "Confidence, Geometric Robustness, and Semantic Coherence Diverge Under Adversarial Attack"  
**Current Phase:** Week 1 — Caltech-101 CNN Analysis Complete, Building Adversarial Bank  
**Timeline:** 12 weeks to submission  
**Last Updated:** 2026-04-04

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Research Questions & Hypothesis](#2-research-questions--hypothesis)
3. [Paper 1 Scope & Contributions](#3-paper-1-scope--contributions)
4. [Current Status](#4-current-status)
5. [Existing Codebase](#5-existing-codebase)
6. [Technical Stack](#6-technical-stack)
7. [Critical Implementation Rules](#7-critical-implementation-rules)
8. [Task Specifications (Week 1)](#8-task-specifications-week-1)
9. [Common Code Patterns](#9-common-code-patterns)
10. [Quality Assurance](#10-quality-assurance)
11. [Troubleshooting Guide](#11-troubleshooting-guide)

---

## 1. PROJECT OVERVIEW

### 1.1 Core Thesis

**Adversarial vulnerability in vision architectures is not random but structurally determined by architectural connectivity patterns that govern decision boundary geometry and semantic failure modes.**

### 1.2 The Three Divergences (Central Contribution)

We demonstrate that three robustness measures are **empirically separable**:

1. **Confidence ≠ Adversarial Robustness**
   - High clean accuracy does not predict adversarial stability
   - Example: DenseNet121 (92% clean → 0% adversarial)

2. **Adversarial Accuracy ≠ Semantic Coherence** ← **NOVEL CONTRIBUTION**
   - Two models with same adversarial accuracy show different semantic failure patterns
   - Architecture determines whether failures are semantically structured or random
   - Example: Model A confuses "dog→cat→wolf" (low entropy, semantic), Model B confuses "dog→keyboard→stoplight" (high entropy, random)

3. **Boundary Geometry ≠ Semantic Failure Mode**
   - Similar DeepFool distances, different confusion patterns
   - Geometric robustness doesn't predict semantic coherence

### 1.3 Why This Matters

**For security-critical systems:** Confusing "stop sign → speed limit sign" is safer than "stop sign → green light." Traditional metrics (accuracy, L2 distance) don't capture this.

**For model deployment:** Practitioners need to know not just IF models fail, but HOW they fail semantically.

---

## 2. RESEARCH QUESTIONS & HYPOTHESIS

### 2.1 Research Questions

**RQ1:** How does architectural connectivity determine decision boundary geometry?

**RQ2:** Do models with similar adversarial accuracy exhibit different semantic failure patterns?

**RQ3:** Can we predict semantic coherence of failures from architectural features?

**RQ4:** Do generative attacks reveal semantic vulnerabilities missed by gradient attacks?

**RQ5:** How do confusion patterns transfer across architectures?

### 2.2 Core Hypothesis

**Dense connectivity → boundary-proximate representations → early confidence inversion → random semantic failures**

**Residual connectivity → boundary-distant representations → late confidence inversion → structured semantic failures**

**Attention-based → different boundary geometry → different semantic pattern**

### 2.3 Novelty Confirmation

**Literature review conducted 2026-04-04 confirms:**

- ✅ No one has studied semantic coherence divergence in CNNs vs ViTs
- ✅ No one has used GAN/VAE attacks for CNN vs ViT comparison
- ✅ Existing work shows "ViT > CNN" using only gradient attacks
- ✅ Our contribution: Same adversarial accuracy ≠ same semantic failure mode

---

## 3. PAPER 1 SCOPE & CONTRIBUTIONS

### 3.1 Full Title

**"Confidence, Geometric Robustness, and Semantic Coherence Diverge Under Adversarial Attack: An Architectural Analysis of CNNs and Vision Transformers"**

### 3.2 Target Venue

**IEEE Transactions on Information Forensics and Security (TIFS)**

- Q1 journal, impact factor ~6.8
- Acceptance probability: 75-85%
- Timeline: 12 weeks

### 3.3 Models Under Study

**CNNs (Pixel and Feature-Level Robustness):**

- VGG19 (dense connectivity)
- ResNet50 (residual connectivity)
- DenseNet121 (extreme dense connectivity)

**Vision Transformer:**

- Swin Transformer (hierarchical attention)

**All models trained on Caltech-101 (101 classes, 869 test images)**

### 3.4 Attack Methods

**Gradient-Based Attacks:**

- FGSM (Fast Gradient Sign Method) — single-step
- PGD (Projected Gradient Descent) — iterative, ε ∈ {0.005, 0.01, 0.02, 0.04}
- DeepFool — minimal perturbation to decision boundary

**Generative Attacks (User Implementing):**

- Latent Interpolation Attack (GAN/VAE)
- Style Transfer Attack (AdaIN)
- Prototype Shift Attack
- Diffusion-like Noise Attack

### 3.5 Novel Contributions

1. **First demonstration that adversarial accuracy ≠ semantic coherence**
   - Confusion entropy as new robustness metric
   - Architecture determines semantic structure of failures

2. **Confidence inversion onset metric**
   - Measures epsilon at which model confidence flips from correct→wrong class
   - Architecture-dependent: DenseNet early onset, ResNet late onset

3. **Boundary accessibility index**
   - Ratio of DeepFool perturbation magnitude across models
   - Quantifies geometric vulnerability independent of accuracy

4. **First GAN/VAE attack comparison on CNN vs ViT**
   - Hypothesis: Generative attacks transfer better than gradient attacks
   - Semantic perturbations exploit different vulnerabilities

5. **Transfer attack matrices**
   - Correlation between semantic similarity and transferability
   - Architecture-dependent propagation patterns

### 3.6 What This Paper Does NOT Include

❌ Vision-Language Models (VLMs) — saved for Paper 2

❌ Defense mechanisms — purely analytical

❌ Watermarking — mentioned only as future work

❌ Multiple datasets — Caltech-101 for CNNs, ImageNet-100 for ViT (Week 3+)

---

## 4. CURRENT STATUS

### 4.1 Completed Work

✅ **CNNs Trained on Caltech-101**

- VGG19: 87.2% clean accuracy
- ResNet50: 90.1% clean accuracy
- DenseNet121: 92.3% clean accuracy

✅ **Gradient Attacks Implemented and Run**

- FGSM: All models evaluated at ε ∈ {0.005, 0.01, 0.02, 0.04}
- PGD: All models evaluated at same epsilon values
- DeepFool: Minimal perturbations computed for all test samples

✅ **Gradient Masking Diagnostic Completed**

- Test flagged false positive (heavy-tailed gradient distribution)
- Results confirmed valid: DenseNet extreme vulnerability is genuine

✅ **Frozen Test Split Created**

- Seed 42, 869 test images
- Saved to: `frozen_split_indices.json`

✅ **Clean Baselines Established**

- Per-sample predictions, confidences, features saved
- File: `clean_baselines.json`

### 4.2 Key Results So Far

**FGSM Results (ε=0.01):**

- VGG19: 25.8% adversarial accuracy
- ResNet50: 38.2% adversarial accuracy
- DenseNet121: 0.0% adversarial accuracy ← **Extreme vulnerability**

**PGD Results (ε=0.01):**

- VGG19: 18.3% adversarial accuracy
- ResNet50: 31.6% adversarial accuracy
- DenseNet121: 0.0% adversarial accuracy

**DeepFool Mean Perturbation:**

- VGG19: L2 = 0.0421, L∞ = 0.0089
- ResNet50: L2 = 0.0634, L∞ = 0.0134
- DenseNet121: L2 = 0.0198, L∞ = 0.0047 ← **Boundary-proximate**

**Interpretation:**

- DenseNet121 has smallest perturbation budget BUT worst robustness
- This proves: Boundary proximity ≠ just accuracy drop
- Dense connectivity creates easily exploitable decision boundaries

### 4.3 In Progress

🔄 **User Implementing Generative Attacks**

- Location: `Model Training/genai-attacks.ipynb`
- User must complete by end of Week 1 or generative attacks dropped from paper
- User has RTX 4060 laptop + RTX 3080 workstation access

### 4.4 Pending Tasks (Your Work This Week)

⏳ **Task 2:** Build adversarial image bank with complete metadata

⏳ **Task 3:** Transfer attack matrix (CNN→CNN, later CNN→ViT)

⏳ **Task 4:** Confusion direction analysis ← **Critical for semantic contribution**

⏳ **Task 5:** Three-seed statistical validation

---

## 5. EXISTING CODEBASE

### 5.1 Directory Structure

```
Model Training/
├── Model_Training.ipynb          # Original CNN training notebook
├── cnn-attacks.ipynb             # FGSM, PGD, DeepFool implementations
├── genai-attacks.ipynb           # User implementing generative attacks
├── checkpoints/
│   ├── VGG19_best.h5            # Trained model weights
│   ├── ResNet50_best.h5
│   └── DenseNet121_best.h5
├── fgsm_results/
│   ├── fgsm_results.csv         # FGSM evaluation results
│   ├── pgd_full_results.json    # PGD detailed results
│   ├── deepfool_results/
│   │   └── boundary_geometry.csv
│   └── gradient_masking/
│       └── gradient_masking_results.json
├── clean_baselines/
│   └── clean_baselines.json     # Clean test set predictions
└── frozen_split_indices.json    # Frozen test split (seed 42)

configs/
└── training_config.yaml          # Model training hyperparameters

src/
└── (shared utilities — to be created)

scripts/
└── (data preparation — to be created)

experiments/
└── (new experiment notebooks — to be created)

results/
└── (aggregated analysis — to be created)
```

### 5.2 Key Files and Their Contents

**`frozen_split_indices.json`:**
```json
{
  "seed": 42,
  "test_indices": [12, 45, 78, ...],
  "dataset": "Caltech-101",
  "num_classes": 101
}
```

**`fgsm_results.csv`:**
```csv
model,epsilon,clean_accuracy,adversarial_accuracy,fooling_rate,mean_confidence_clean,mean_confidence_adv,mean_l2_distance,mean_linf_distance
VGG19,0.005,0.872,0.456,0.477,0.892,0.634,0.0124,0.005
VGG19,0.01,0.872,0.258,0.704,0.892,0.512,0.0243,0.01
...
```

**`pgd_full_results.json`:**
```json
{
  "VGG19": {
    "eps_0.01": {
      "adversarial_accuracy": 0.183,
      "fooling_rate": 0.790,
      "mean_confidence_adv": 0.487,
      "per_sample_results": [...]
    }
  }
}
```

**`deepfool_results/boundary_geometry.csv`:**
```csv
model,sample_id,true_class,perturbation_l2,perturbation_linf,num_iterations,fooled
VGG19,0,accordion,0.0421,0.0089,12,True
...
```

**`clean_baselines.json`:**
```json
{
  "VGG19": {
    "predictions": [0, 1, 1, ...],
    "confidences": [0.95, 0.87, ...],
    "correct": [True, True, False, ...]
  }
}
```

---

## 6. TECHNICAL STACK

### 6.1 Core Dependencies

**Deep Learning Framework:**

- TensorFlow 2.15.1 (NO standalone Keras)
- tensorflow-estimator 2.15.0
- tensorboard 2.15.2

**Scientific Computing:**

- numpy 1.26.4
- scipy 1.11.4
- pandas 2.1.4

**Visualization:**

- matplotlib 3.8.4
- pillow 10.4.0

**Full requirements:** See `requirement.txt`

### 6.2 Hardware Information

**User's Available Hardware:**

- RTX 4060 laptop (8GB VRAM) — primary development
- RTX 3080 workstation (10GB VRAM) — college access for training

**Memory Management Needed:**

- Batch process adversarial generation (don't load all 869 at once)
- Clear GPU memory between models: `tf.keras.backend.clear_session()`

### 6.3 Data Specifications

**Current Dataset:** Caltech-101

- Classes: 101
- Test set: 869 images (frozen seed 42)
- Image size: 224×224×3
- Pixel range: [0, 1] (float32)

**Future Dataset (Week 3+):** ImageNet-100

- For ViT experiments only
- Do NOT implement ImageNet-100 code this week

---

## 7. CRITICAL IMPLEMENTATION RULES

### 7.1 File Reading Protocol

**BEFORE editing ANY existing file:**

1. Use `view` tool to read ENTIRE file first
2. Understand existing structure completely
3. Match code style exactly
4. **NEVER include line number prefixes in `str_replace` old_str**

### 7.2 Format Matching

**All new results must match existing formats EXACTLY:**

- CSV column names identical
- JSON structure identical
- Precision: 3-4 decimal places
- Data types must match

### 7.3 No Placeholders

**Absolutely forbidden:**

- `pass` statements
- `TODO` comments
- Placeholder functions
- Incomplete implementations

### 7.4 Model Loading

**Models located at:**
```python
model_paths = {
    'VGG19': 'Model Training/checkpoints/VGG19_best.h5',
    'ResNet50': 'Model Training/checkpoints/ResNet50_best.h5',
    'DenseNet121': 'Model Training/checkpoints/DenseNet121_best.h5'
}
```

**DO NOT retrain models. Use existing checkpoints.**

### 7.5 Data Split Usage

**ALWAYS use frozen split from `frozen_split_indices.json`**

### 7.6 Documentation Requirements

**Every file must have header documenting purpose, inputs, outputs**

---

## 8. TASK SPECIFICATIONS (WEEK 1)

### 8.1 TASK 2: Build Adversarial Image Bank

**Goal:** Save ALL adversarial examples with complete metadata including predicted labels.

**Critical Fix:** Current code throws away `adv_pred` and `adv_pred_idx`. Must SAVE these.

**See CLAUDE.md for complete implementation details.**

### 8.2 TASK 3: Transfer Attack Matrix

**Goal:** Test if adversarial examples from model S fool model T.

**Critical for propagation narrative.**

**See CLAUDE.md for complete implementation details.**

### 8.3 TASK 4: Confusion Direction Analysis

**Goal:** Extract WHICH classes models confuse to.

**THIS IS THE CORE SEMANTIC CONTRIBUTION.**

**Confusion entropy reveals whether failures are structured or random.**

**See CLAUDE.md for complete implementation details.**

### 8.4 TASK 5: Three-Seed Statistical Validation

**Goal:** Validate results across different test splits.

**Re-run attacks only, NOT model training.**

**See CLAUDE.md for complete implementation details.**

---

## 9. COMMON CODE PATTERNS

### 9.1 Loading Frozen Split

```python
with open('Model Training/frozen_split_indices.json', 'r') as f:
    split = json.load(f)
test_indices = split['test_indices']
```

### 9.2 Batch Processing

```python
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    process_batch(batch)
    if i % 100 == 0:
        tf.keras.backend.clear_session()
```

### 9.3 Progress Reporting

```python
from tqdm import tqdm
for item in tqdm(items, desc="Processing"):
    # work
```

### 9.4 GPU Memory Management

```python
tf.keras.backend.clear_session()
gc.collect()
```

### 9.5 Standard Notebook Header

```python
# ============================================================
# Task N: [Task Name]
# Purpose: [What this notebook does]
# Inputs:  [Files read]
# Outputs: [Files written]
# ============================================================

import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(gpus) > 0}")
```

---

## 10. QUALITY ASSURANCE

**Before marking task complete:**

- [ ] All output files exist
- [ ] Formats match exactly
- [ ] No NaN values
- [ ] Spot-check 5 samples
- [ ] Results reasonable
- [ ] Console prints completion message

---

## 11. TROUBLESHOOTING GUIDE

**OOM Error:** Reduce batch size, clear GPU memory

**File not found:** Use absolute paths, verify existence

**Format mismatch:** Load existing file, match structure exactly

**Random results:** Set all seeds (random, numpy, tensorflow)

**TensorFlow import errors:** Use `import tensorflow as tf` only — no standalone Keras

**`.h5` load errors:** Ensure `tensorflow` version matches training environment (2.15.1)

---

**END OF SKILLS.md**

For complete task implementation details, see CLAUDE.md.
