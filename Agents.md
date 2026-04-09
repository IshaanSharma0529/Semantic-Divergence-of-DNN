# AGENTS.md

**Project:** Adversarial Robustness Analysis Across Vision Architectures  
**Paper:** "Confidence, Geometric Robustness, and Semantic Coherence Diverge Under Adversarial Attack"  
**Venue:** IEEE TIFS (12 weeks) → ACL (10 weeks)  
**Phase:** Week 1 — Building Adversarial Bank

---

## Stack

- **Language:** Python 3.10+
- **Framework:** TensorFlow 2.15.1 (NO standalone Keras)
- **Scientific:** NumPy 1.26.4, Pandas 2.1.4, SciPy 1.11.4
- **Notebooks:** Jupyter, IPython 8.18.1
- **Hardware:** RTX 4060 (8GB) laptop, RTX 3080 (10GB) workstation

---

## Project Structure

```
Model Training/          # Trained models & attack implementations
├── checkpoints/         # .h5 model weights (DO NOT RETRAIN)
├── fgsm_results/        # Attack evaluation results (READ ONLY)
├── cnn-attacks.ipynb    # FGSM, PGD, DeepFool implementations
└── genai-attacks.ipynb  # User implementing (in progress)

adversarial_bank/        # YOU WILL CREATE (Task 2)
results/                 # YOU WILL CREATE (Tasks 3-5)
experiments/             # YOU WILL CREATE (analysis notebooks)
configs/                 # training_config.yaml
src/                     # Shared utilities (to be created)
scripts/                 # Data prep, multi-seed runner
```

---

## Critical Files (READ ONLY)

**DO NOT MODIFY these files:**
- `Model Training/checkpoints/*.h5` — Trained model weights
- `Model Training/frozen_split_indices.json` — Test split (seed 42, 869 samples)
- `Model Training/fgsm_results/*` — Existing attack results
- `Model Training/clean_baselines/clean_baselines.json` — Clean predictions

**Always use existing frozen split:**
```python
with open('Model Training/frozen_split_indices.json', 'r') as f:
    split = json.load(f)
test_indices = split['test_indices']  # Always use this
```

---

## Coding Standards

### File Headers
Every Python file must have:
```python
"""
Script: {filename}
Purpose: {brief description}

Dataset: Caltech-101 (or ImageNet-100 for ViT)
Models: VGG19, ResNet50, DenseNet121
Seed: 42

Inputs:
- {list input files/directories}

Outputs:
- {list output files/directories}

Author: Claude Code
Date: {date}
"""
```

### Function Docstrings
```python
def function_name(arg1, arg2):
    """
    Brief description.
    
    Args:
        arg1 (type): Description
        arg2 (type): Description
        
    Returns:
        type: Description
        
    Raises:
        ErrorType: When this occurs
        
    Example:
        >>> result = function_name(x, y)
    """
```

### Import Order
```python
# Standard library
import os
import json
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Project-specific
from src.utils import load_model, evaluate_attack
```

---

## Data Format Standards

### Images
- **Format:** `.npy` (float32, NOT PNG)
- **Range:** [0, 1]
- **Shape:** (224, 224, 3) or batch: (N, 224, 224, 3)
- **Why .npy:** Precision matters for re-evaluation

### CSV Results
**Match existing format exactly:**
```csv
model,attack,epsilon,metric,value
VGG19,FGSM,0.01,fooling_rate,0.742
```

**Required columns for metadata:**
```csv
sample_id,true_class,true_class_idx,clean_pred,clean_pred_idx,clean_confidence,adv_pred,adv_pred_idx,adv_confidence,fooled,l2_distance,linf_distance,epsilon,attack_name,model_name
```

### JSON Results
**Match existing structure:**
```json
{
  "VGG19": {
    "FGSM": {
      "eps_0.01": {
        "fooling_rate": 0.742,
        "adversarial_accuracy": 0.258,
        "mean_confidence": 0.512
      }
    }
  }
}
```

---

## Critical Rules

### 1. Read Before Edit
**ALWAYS use `view` tool before modifying existing files**
```python
# Example: Before editing cnn-attacks.ipynb
# 1. view('Model Training/cnn-attacks.ipynb')
# 2. Understand structure
# 3. Then edit
```

### 2. Never Retrain Models
```python
# CORRECT: Load existing
model = tf.keras.models.load_model('Model Training/checkpoints/VGG19_best.h5')

# WRONG: Retrain
model.fit(...)  # DO NOT DO THIS
```

### 3. Use Frozen Split
```python
# CORRECT
with open('Model Training/frozen_split_indices.json') as f:
    test_indices = json.load(f)['test_indices']

# WRONG
test_indices = np.random.choice(...)  # DO NOT create new splits
```

### 4. Match Existing Formats
```python
# Load existing to match structure
existing_df = pd.read_csv('Model Training/fgsm_results/fgsm_results.csv')
new_df = pd.DataFrame(columns=existing_df.columns)  # Same columns
```

### 5. No Placeholders
```python
# WRONG
def process_data():
    pass  # TODO: implement

# CORRECT
def process_data():
    # Complete implementation
    data = load_data()
    results = analyze(data)
    return results
```

### 6. GPU Memory Management
```python
# Clear memory between model loads
import gc
tf.keras.backend.clear_session()
gc.collect()
```

---

## Common Patterns

### Batch Processing
```python
from tqdm import tqdm

for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
    batch = data[i:i+batch_size]
    results = process_batch(batch)
    
    # Clear memory periodically
    if i % 100 == 0:
        tf.keras.backend.clear_session()
```

### Safe File Saving
```python
from pathlib import Path

output_path = Path('results/transfer/matrix.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

# Backup if exists
if output_path.exists():
    shutil.copy(output_path, str(output_path) + '.backup')

df.to_csv(output_path, index=False)
```

### Verification Pattern
```python
def verify_output(filepath, expected_rows=869):
    """Verify output file integrity."""
    df = pd.read_csv(filepath)
    assert len(df) == expected_rows, f"Expected {expected_rows}, got {len(df)}"
    assert df.isna().sum().sum() == 0, "Found NaN values"
    print(f"✓ Verified: {filepath}")
```

---

## Current Tasks (Week 1)

### Task 2: Build Adversarial Image Bank
**Create:** `adversarial_bank/{model}/{attack}/eps_{e}/images/*.npy` + `metadata.csv`

**Critical fix needed:**
- Current code throws away `adv_pred` and `adv_pred_idx`
- YOU MUST SAVE these in metadata.csv
- See CLAUDE.md for complete implementation

### Task 3: Transfer Attack Matrix
**Create:** `results/transfer/transfer_matrix_{attack}_eps{e}.csv`

**Test:** Do adversarial examples from model S fool model T?

### Task 4: Confusion Direction Analysis ← **CORE CONTRIBUTION**
**Create:** `results/confusion/{model}/{attack}/eps_{e}_confusion_matrix.csv`

**Compute:** Confusion entropy (semantic coherence metric)
- Low entropy (<1.5): Semantically structured failures
- High entropy (>2.5): Random failures

**This proves:** Adversarial accuracy ≠ semantic coherence

### Task 5: Three-Seed Validation
**Create:** `results/seed_{42,123,456}/` + `results/aggregated/`

**Re-run attacks only (NOT training) on different splits**

---

## Attack Implementations (Reference Only)

### FGSM
```python
def fgsm_attack(model, images, labels, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adv_images = images + epsilon * signed_grad
    return tf.clip_by_value(adv_images, 0, 1)
```

### Load from existing code in `cnn-attacks.ipynb`

---

## Testing Requirements

### Before Marking Task Complete
- [ ] All output files exist in correct locations
- [ ] File formats match existing patterns exactly
- [ ] No NaN values in results
- [ ] Spot-check 5 random samples
- [ ] Results are reasonable (no all-zeros)
- [ ] Console prints "TASK {N} COMPLETE"

### Verification Example
```python
# Verify adversarial bank
total_images = 0
for model_dir in Path('adversarial_bank').iterdir():
    for attack_dir in model_dir.iterdir():
        for eps_dir in attack_dir.iterdir():
            img_count = len(list((eps_dir / 'images').glob('*.npy')))
            assert img_count == 869, f"Expected 869, got {img_count}"
            total_images += img_count

print(f"✓ Total images verified: {total_images}")
```

---

## Error Handling

### Pattern
```python
try:
    result = operation()
    if result is None:
        print(f"WARNING: Operation returned None")
        return None
except FileNotFoundError as e:
    print(f"ERROR: File not found - {e}")
    raise
except Exception as e:
    print(f"ERROR in {function_name}: {e}")
    raise
```

---

## Progress Reporting

### When Task Complete
```
TASK {N} COMPLETE — {brief summary}

Output files:
- {file_path} ({size})

Verification:
- {check}: PASSED

Next: Task {N+1}
```

### When Issue Encountered
```
ISSUE in Task {N}: {description}

Details:
- Tried: {what you attempted}
- Error: {error message}
- Need: {what's needed}

Options:
1. {option 1}
2. {option 2}
```

---

## Key Results (Context)

**DenseNet121 Extreme Vulnerability:**
- Clean accuracy: 92.3%
- Adversarial accuracy (ε=0.01): 0.0%
- DeepFool L2: 0.0198 (smallest = boundary-proximate)

**This proves:** Small boundary distance ≠ high robustness

**Paper's novel contribution:**
Two models with same adversarial accuracy can have different semantic failure modes:
- DenseNet: Random confusions (high entropy)
- ResNet: Structured confusions (low entropy)

---

## Links to Full Documentation

- **SKILLS.md** — Complete project context, research questions, hypothesis
- **CLAUDE.md** — Detailed task implementation instructions
- **EXECUTION_PLAN.md** — 12-week timeline, decision rules

**Read SKILLS.md first for full context.**

---

## What NOT to Do

❌ Retrain models  
❌ Create new random splits  
❌ Use placeholder code (`pass`, `TODO`)  
❌ Modify existing result files  
❌ Use PNG for adversarial images (use .npy)  
❌ Include line numbers in str_replace  
❌ Over-format with excessive headers/bullets  

---

## MCP Servers (Future)

*Not configured yet. Will add when needed.*

---

**Last Updated:** 2026-04-04  
**Current Week:** 1 of 12 (Paper 1)  
**Current Phase:** Building adversarial bank and running semantic analysis

---

END OF AGENTS.MD

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
