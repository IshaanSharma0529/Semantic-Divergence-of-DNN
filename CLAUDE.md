# CLAUDE CODE INSTRUCTIONS — PAPER 1 EXECUTION

**Paper:** "Confidence, Geometric Robustness, and Semantic Coherence Diverge Under Adversarial Attack"  
**Venue:** IEEE Transactions on Information Forensics and Security (TIFS)  
**Phase:** Week 1 — Building Adversarial Bank and Running Analysis  
**Timeline:** 12 weeks total  
**Last Updated:** 2026-04-04

---

## QUICK START

**Before starting ANY task:**
1. Read SKILLS.md for full project context
2. Read this file for task implementation details
3. Use `view` tool to examine existing code
4. Match existing formats EXACTLY

**Your tasks this week:**
- Task 2: Build adversarial image bank
- Task 3: Transfer attack matrix
- Task 4: Confusion direction analysis (CRITICAL for paper)
- Task 5: Three-seed statistical validation

---

## PROJECT SUMMARY

### What We're Proving

**Three robustness measures diverge:**

1. Confidence ≠ Robustness (DenseNet: 92% clean → 0% adversarial)
2. **Adversarial Accuracy ≠ Semantic Coherence** ← OUR NOVEL CONTRIBUTION
3. Boundary Geometry ≠ Semantic Failure Mode

### Why This Matters

Traditional metrics don't tell you HOW models fail semantically. Two models with same adversarial accuracy can have completely different semantic failure patterns:

- Model A: Confuses "dog→cat→wolf" (semantically structured)
- Model B: Confuses "dog→keyboard→stoplight" (random chaos)

**For security systems:** This distinction matters. Confusing "stop sign→speed limit" is safer than "stop sign→green light."

---

## WHAT EXISTS (Completed Work)

✅ **3 CNNs trained on Caltech-101:**
- VGG19, ResNet50, DenseNet121
- Checkpoints: `Model Training/checkpoints/`

✅ **Gradient attacks run:**
- FGSM, PGD (ε ∈ {0.005, 0.01, 0.02, 0.04})
- DeepFool (minimal perturbation)
- Results: `Model Training/fgsm_results/`

✅ **Key findings:**
- DenseNet121: 0% adversarial accuracy (extreme vulnerability)
- DenseNet121: Smallest DeepFool distance (boundary-proximate)
- This proves: Close boundary ≠ high accuracy

✅ **Frozen test split:**
- 869 samples, seed 42
- File: `frozen_split_indices.json`

---

## WHAT YOU'LL BUILD (This Week)

### Task 2: Adversarial Image Bank
**Save all adversarial examples permanently**

Why: Current code throws away predicted labels. Need them for semantic analysis.

### Task 3: Transfer Attack Matrix
**Test if adversarial examples transfer between models**

Why: Critical for understanding propagation to ViT later.

### Task 4: Confusion Direction Analysis
**Extract WHICH classes models confuse to**

Why: **THIS IS THE PAPER'S CORE CONTRIBUTION.** Proves semantic coherence diverges from accuracy.

### Task 5: Three-Seed Validation
**Re-run attacks with different test splits**

Why: Prove results aren't random artifacts.

---

## CRITICAL RULES

### Rule 1: Read Before Edit
**ALWAYS use `view` tool to read files before editing**

### Rule 2: Match Formats EXACTLY
Load existing CSV/JSON, match structure precisely

### Rule 3: No Placeholders
No `pass`, no `TODO`, complete code only

### Rule 4: Don't Retrain Models
Use existing checkpoints from `Model Training/checkpoints/`

### Rule 5: Use Frozen Split
Always load from `frozen_split_indices.json` (seed 42)

### Rule 6: No Line Numbers in str_replace
```python
# WRONG
old_str = "    42\tdef function():"

# CORRECT  
old_str = "def function():"
```

---

## TASK 2: BUILD ADVERSARIAL IMAGE BANK

### Goal
Save ALL adversarial examples with complete metadata including predicted class names (currently being thrown away).

### Directory Structure to Create
```
adversarial_bank/
  VGG19/
    FGSM/
      eps_0.005/
        images/
          0000_accordion.npy
          0001_airplanes.npy
          ...
        metadata.csv
      eps_0.01/
        (same structure)
    PGD/
      (same structure)
    DeepFool/
      images/
      metadata.csv
  ResNet50/
    (same structure)
  DenseNet121/
    (same structure)
```

### metadata.csv Format (CRITICAL)

**Must include ALL these columns:**
```csv
sample_id,true_class,true_class_idx,clean_pred,clean_pred_idx,clean_confidence,adv_pred,adv_pred_idx,adv_confidence,fooled,l2_distance,linf_distance,epsilon,attack_name,model_name
```

**The problem:** Current code computes `adv_pred` and `adv_pred_idx` but throws them away. YOU MUST SAVE THEM.

### Implementation Steps

1. **Load model and data:**
```python
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Load frozen split
with open('Model Training/frozen_split_indices.json', 'r') as f:
    split = json.load(f)
test_indices = split['test_indices']

# Load model
model = tf.keras.models.load_model('Model Training/checkpoints/VGG19_best.h5')

# Load test images and labels
# (Use existing data loading code from cnn-attacks.ipynb)
```

2. **For each attack and epsilon:**
```python
models = ['VGG19', 'ResNet50', 'DenseNet121']
attacks = {
    'FGSM': [0.005, 0.01, 0.02, 0.04],
    'PGD': [0.005, 0.01, 0.02, 0.04],
    'DeepFool': [None]  # No epsilon
}

for model_name in models:
    model = load_model(model_name)
    
    for attack_name, epsilons in attacks.items():
        for eps in epsilons:
            # Generate adversarial examples
            adv_images = run_attack(model, attack_name, eps)
            
            # Get predictions on adversarial images
            adv_preds = model.predict(adv_images)
            adv_pred_idx = np.argmax(adv_preds, axis=1)
            adv_confidence = np.max(adv_preds, axis=1)
            
            # Map indices to class names
            adv_pred_names = [class_names[idx] for idx in adv_pred_idx]
            
            # Save each image and build metadata
            metadata_rows = []
            for i, (adv_img, true_idx, true_name) in enumerate(zip(adv_images, true_labels, true_names)):
                # Save image
                img_path = f'adversarial_bank/{model_name}/{attack_name}/eps_{eps}/images/{i:04d}_{true_name}.npy'
                Path(img_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(img_path, adv_img)
                
                # Build metadata row
                metadata_rows.append({
                    'sample_id': i,
                    'true_class': true_name,
                    'true_class_idx': true_idx,
                    'clean_pred': clean_pred_names[i],
                    'clean_pred_idx': clean_pred_idx[i],
                    'clean_confidence': clean_confidences[i],
                    'adv_pred': adv_pred_names[i],  # DON'T THROW THIS AWAY!
                    'adv_pred_idx': adv_pred_idx[i],  # DON'T THROW THIS AWAY!
                    'adv_confidence': adv_confidence[i],
                    'fooled': (adv_pred_idx[i] != true_idx),
                    'l2_distance': np.linalg.norm(adv_img - clean_images[i]),
                    'linf_distance': np.max(np.abs(adv_img - clean_images[i])),
                    'epsilon': eps if eps else 'minimal',
                    'attack_name': attack_name,
                    'model_name': model_name
                })
            
            # Save metadata CSV
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_path = f'adversarial_bank/{model_name}/{attack_name}/eps_{eps}/metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
```

3. **Verify the bank:**
```python
# Count total images
total_images = 0
for model_dir in Path('adversarial_bank').iterdir():
    for attack_dir in model_dir.iterdir():
        for eps_dir in attack_dir.iterdir():
            img_count = len(list((eps_dir / 'images').glob('*.npy')))
            total_images += img_count
            assert img_count == 869, f"Expected 869 images in {eps_dir}, got {img_count}"

# Spot-check 5 random samples
import random
random.seed(42)
samples_to_check = random.sample(range(869), 5)

for sample_id in samples_to_check:
    # Load image
    img = np.load(f'adversarial_bank/VGG19/FGSM/eps_0.01/images/{sample_id:04d}_*.npy')
    
    # Load metadata
    metadata = pd.read_csv('adversarial_bank/VGG19/FGSM/eps_0.01/metadata.csv')
    row = metadata[metadata['sample_id'] == sample_id].iloc[0]
    
    # Re-evaluate
    model = load_model('VGG19')
    pred = model.predict(img.reshape(1, 224, 224, 3))
    pred_idx = np.argmax(pred)
    
    # Verify matches metadata
    assert pred_idx == row['adv_pred_idx'], f"Sample {sample_id} prediction mismatch"

print(f"✓ Verification passed: {total_images} images")
```

4. **Save verification report:**
```python
verification = {
    'total_images': total_images,
    'total_size_gb': total_size / (1024**3),
    'models': 3,
    'attacks': 3,
    'epsilons_per_attack': {
        'FGSM': 4,
        'PGD': 4,
        'DeepFool': 1
    },
    'spot_checks_passed': 5,
    'timestamp': datetime.now().isoformat()
}

with open('results/adversarial_bank_verification.json', 'w') as f:
    json.dump(verification, f, indent=2)

print(f"TASK 2 COMPLETE — {total_images} adversarial images saved, {total_size_gb:.2f} GB")
```

### Expected Outputs
- `adversarial_bank/` directory with 3 × 3 × (4+4+1) = 27 subdirectories
- Each subdirectory: 869 .npy images + metadata.csv
- `results/adversarial_bank_verification.json`

---

## TASK 3: TRANSFER ATTACK MATRIX

### Goal
Test if adversarial examples crafted on model S fool model T (S ≠ T).

**This is CRITICAL:** If mean transfer < 25%, propagation narrative weakens.

### Create File
`experiments/transfer_attack_matrix.ipynb`

### Method

```python
# For each source model S
for source_model in ['VGG19', 'ResNet50', 'DenseNet121']:
    
    # For each target model T
    for target_model in ['VGG19', 'ResNet50', 'DenseNet121']:
        
        if source_model == target_model:
            continue  # Skip white-box (already known)
        
        # Load adversarial images from source
        source_dir = f'adversarial_bank/{source_model}/FGSM/eps_0.01/images/'
        adv_images = [np.load(f) for f in sorted(Path(source_dir).glob('*.npy'))]
        
        # Load target model
        target = tf.keras.models.load_model(f'checkpoints/{target_model}_best.h5')
        
        # Evaluate (no gradient computation, just inference)
        target_preds = target.predict(np.array(adv_images))
        target_pred_idx = np.argmax(target_preds, axis=1)
        
        # Load true labels from metadata
        metadata = pd.read_csv(f'adversarial_bank/{source_model}/FGSM/eps_0.01/metadata.csv')
        true_labels = metadata['true_class_idx'].values
        
        # Compute transfer fooling rate
        fooled = (target_pred_idx != true_labels)
        transfer_rate = np.mean(fooled) * 100
        
        print(f"Transfer: {source_model} → {target_model}: {transfer_rate:.1f}%")
```

### Output Format

**Save:** `results/transfer/transfer_matrix_FGSM_eps0.01.csv`
```csv
Source\Target,VGG19,ResNet50,DenseNet121
VGG19,-,45.2,38.7
ResNet50,52.1,-,41.3
DenseNet121,48.9,43.2,-
```

**Save:** `results/transfer/transferability_ranking.json`
```json
{
  "FGSM": {
    "eps_0.01": {
      "VGG19_mean_transfer": 41.95,
      "ResNet50_mean_transfer": 46.70,
      "DenseNet121_mean_transfer": 46.05,
      "most_transferable_source": "ResNet50",
      "least_transferable_source": "VGG19",
      "overall_mean_transfer": 44.90
    }
  }
}
```

### Interpretation (add to notebook)

```markdown
## Transfer Analysis Results

**Overall transfer rate: 44.90%**

**Implication:**
- Transfer rate > 40% suggests CNN→ViT propagation is plausible
- Use ResNet50 (highest transfer) for ViT experiments

**Next:** Investigate if semantic structure affects transfer (Task 4)
```

---

## TASK 4: CONFUSION DIRECTION ANALYSIS

### Goal
Extract WHICH classes models confuse to, not just that they fail.

**THIS IS THE PAPER'S CORE SEMANTIC CONTRIBUTION.**

### Create File
`experiments/confusion_direction_analysis.ipynb`

### Confusion Entropy Concept

**Low entropy (< 1.5):** Semantically structured failures
- Model confuses to few similar classes
- Example: dog → {cat: 40%, wolf: 30%, fox: 20%, coyote: 10%}

**High entropy (> 2.5):** Random failures
- Model confuses to many unrelated classes  
- Example: dog → {keyboard: 8%, stoplight: 7%, ceiling_fan: 6%, ...}

### Method

```python
from scipy.stats import entropy

# For each model × attack × epsilon
for model_name in ['VGG19', 'ResNet50', 'DenseNet121']:
    for attack in ['FGSM', 'PGD', 'DeepFool']:
        for eps in epsilons:
            
            # Load metadata
            metadata = pd.read_csv(f'adversarial_bank/{model_name}/{attack}/eps_{eps}/metadata.csv')
            
            # Build confusion matrix
            num_classes = 101
            confusion_matrix = np.zeros((num_classes, num_classes))
            
            for _, row in metadata.iterrows():
                if row['fooled']:
                    true_idx = row['true_class_idx']
                    adv_idx = row['adv_pred_idx']
                    confusion_matrix[true_idx, adv_idx] += 1
            
            # Compute per-class entropy
            entropies = {}
            for class_idx in range(num_classes):
                confusions = confusion_matrix[class_idx, :]
                if confusions.sum() > 0:
                    probs = confusions / confusions.sum()
                    H = entropy(probs)
                    entropies[class_names[class_idx]] = H
            
            mean_entropy = np.mean(list(entropies.values()))
            
            # Save results
            results = {
                'mean_entropy': mean_entropy,
                'median_entropy': np.median(list(entropies.values())),
                'per_class_entropy': entropies
            }
            
            with open(f'results/confusion/{model_name}/{attack}/eps_{eps}_entropy.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Extract top-10 confusion pairs
            top_confusions = []
            for true_idx in range(num_classes):
                for adv_idx in range(num_classes):
                    if true_idx != adv_idx and confusion_matrix[true_idx, adv_idx] > 0:
                        top_confusions.append({
                            'true_class': class_names[true_idx],
                            'confused_class': class_names[adv_idx],
                            'count': confusion_matrix[true_idx, adv_idx],
                            'rate': confusion_matrix[true_idx, adv_idx] / confusion_matrix[true_idx, :].sum()
                        })
            
            top_confusions = sorted(top_confusions, key=lambda x: x['count'], reverse=True)[:10]
            
            # Save
            pd.DataFrame(top_confusions).to_csv(
                f'results/confusion/{model_name}/{attack}/eps_{eps}_top_confusions.csv',
                index=False
            )
```

### Analysis to Include (markdown cell)

```markdown
## Semantic Coherence Results

### DenseNet121 (Dense Connectivity):
- Mean confusion entropy: 2.87 (HIGH)
- **Interpretation:** RANDOM semantic failures
- Example confusions: accordion → {stoplight, keyboard, ceiling_fan, ...}
- **Failure mode:** Semantically incoherent

### ResNet50 (Residual Connectivity):
- Mean confusion entropy: 1.63 (LOW)
- **Interpretation:** STRUCTURED semantic failures
- Example confusions: accordion → {grand_piano, guitar, violin, ...}
- **Failure mode:** Semantically related classes

### VGG19:
- Mean confusion entropy: 2.14 (MEDIUM)
- Mixed semantic structure

### KEY FINDING:
**DenseNet121 and VGG19 both have ~0% accuracy at eps=0.01**
**BUT: DenseNet shows random failures, VGG19 more structured**

**This proves: Adversarial accuracy ≠ semantic coherence**

This is our NOVEL CONTRIBUTION.
```

### Expected Outputs
- `results/confusion/{model}/{attack}/eps_{e}_entropy.json`
- `results/confusion/{model}/{attack}/eps_{e}_confusion_matrix.csv`
- `results/confusion/{model}/{attack}/eps_{e}_top_confusions.csv`
- `results/confusion/semantic_structure_scores.json` (aggregated)

---

## TASK 5: THREE-SEED STATISTICAL VALIDATION

### Goal
Prove results aren't random artifacts by re-running with different test splits.

**IMPORTANT:** Do NOT retrain models. Only re-run attacks on new splits.

### Create File
`scripts/multi_seed_runner.py`

### Method

```python
# For each seed
for seed in [42, 123, 456]:
    
    # Generate new frozen split
    np.random.seed(seed)
    test_indices = generate_split(seed)
    
    # Save split
    with open(f'frozen_splits/test_split_seed{seed}.json', 'w') as f:
        json.dump({'seed': seed, 'test_indices': test_indices}, f)
    
    # Re-run attacks (use existing trained models)
    for model_name in ['VGG19', 'ResNet50', 'DenseNet121']:
        model = load_model(model_name)
        
        # Load test data for this split
        test_images, test_labels = load_data(test_indices)
        
        for attack in ['FGSM', 'PGD', 'DeepFool']:
            for eps in epsilons:
                # Run attack
                adv_images = run_attack(model, attack, eps, test_images, test_labels)
                
                # Evaluate
                results = evaluate(model, adv_images, test_labels)
                
                # Save to results/seed_{seed}/
                save_results(results, seed, model_name, attack, eps)
```

### Aggregate Results

Create: `scripts/aggregate_results.py`

```python
from scipy.stats import bootstrap, wilcoxon

# Aggregate metrics across seeds
for metric in ['fooling_rate', 'adv_accuracy', 'adv_confidence', 'l2_distance']:
    
    values_42 = load_metric(seed=42, metric=metric)
    values_123 = load_metric(seed=123, metric=metric)
    values_456 = load_metric(seed=456, metric=metric)
    
    mean = np.mean([values_42, values_123, values_456])
    std = np.std([values_42, values_123, values_456])
    
    # Bootstrap 95% CI
    ci_lower, ci_upper = bootstrap_ci([values_42, values_123, values_456])
    
    # Check coefficient of variation
    cv = std / mean
    if cv > 0.15:
        warnings.append(f"High variance: {metric}, CV={cv:.2f}")
    
    # Save
    aggregated_results.append({
        'metric': metric,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'seed_42': values_42,
        'seed_123': values_123,
        'seed_456': values_456
    })

# Run significance tests
stat, p_value = wilcoxon(vgg19_values, resnet50_values)
```

### Expected Outputs
- `frozen_splits/test_split_seed{42,123,456}.json`
- `results/seed_{seed}/` directories with all attack results
- `results/aggregated/all_metrics_aggregated.csv`
- `results/aggregated/paper_results_table.csv` (formatted as "mean ± std")
- `results/aggregated/significance_tests.csv`
- `results/aggregated/stability_warnings.txt`

---

## COMMUNICATION PROTOCOL

### When You Complete a Task

```
TASK {N} COMPLETE — {brief summary}

Output files:
- {file1_path} ({size})
- {file2_path} ({size})

Verification:
- {check1}: PASSED
- {check2}: PASSED

Next: Ready for Task {N+1}
```

### When You Encounter a Problem

```
ISSUE in Task {N}: {description}

Details:
- {what you tried}
- {error or unexpected result}
- {what you need}

Options:
1. {option 1}
2. {option 2}

Recommendation: {your suggestion}
```

### When You Need Clarification

**Ask specific questions, not vague ones.**

**GOOD:** "In metadata.csv, should 'epsilon' be float or string when attack is DeepFool?"

**BAD:** "I'm confused about the format."

---

## FINAL CHECKLIST

Before marking any task complete:

- [ ] All output files exist in correct locations
- [ ] File formats match existing patterns exactly
- [ ] Verification checks passed
- [ ] No placeholder code remains
- [ ] Console printed completion message
- [ ] Results are reasonable (spot-checked)

---

## CURRENT STATUS

**Completed:**
- ✅ CNNs trained
- ✅ Gradient attacks run
- ✅ Gradient masking diagnostic
- ✅ Frozen splits created
- ✅ Clean baselines established

**In Progress:**
- 🔄 User implementing generative attacks

**Pending (Your Work):**
- ⏳ Task 2: Adversarial image bank
- ⏳ Task 3: Transfer matrices
- ⏳ Task 4: Confusion analysis ← CRITICAL
- ⏳ Task 5: Multi-seed validation

**Start with Task 2 when ready.**

---

END OF CLAUDE CODE INSTRUCTIONS

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
