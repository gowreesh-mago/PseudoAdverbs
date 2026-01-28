# Evaluation Metrics Documentation

This document explains all evaluation metrics used in the Pseudo-Adverbs training script.

## Overview

The model predicts (action, adverb) pairs from video inputs. During evaluation, we compute multiple metrics to assess different aspects of the model's performance.

## Score Tensors

The evaluator produces three different score tensors, each used for different evaluation purposes:

### 1. `scores` - Unconstrained Scores
- **Contains**: Raw prediction scores for ALL (adverb, action) pairs
- **Shape**: `[batch_size, num_pairs]` where `num_pairs = num_adverbs × num_actions`
- **Usage**: Evaluating unconstrained prediction tasks where the model must predict both action and adverb

### 2. `action_gt_scores` - Action-Constrained Scores
- **Contains**: Scores masked to keep only pairs with the ground truth action
- **Masking**: Sets all pairs where `action ≠ action_gt` to `-1e10`
- **Example**: If GT action = "run", keeps all `(*, "run")` pairs, masks others
- **Usage**: Evaluating adverb prediction when the action is known (not used in final metrics)

### 3. `antonym_action_gt_scores` - Antonym-Constrained Scores
- **Contains**: Scores masked to keep only `(adverb_gt OR antonym, action_gt)` pairs
- **Masking**: Binary choice between adverb and its antonym, with correct action
- **Example**: If GT is `("quickly", "run")`, keeps `("quickly", "run")` and `("slowly", "run")`
- **Usage**: Evaluating binary adverb disambiguation when action is known

## Metric Categories

### 1. Constrained Adverb Metrics
**Task**: Given the ground truth action, distinguish the correct adverb from its antonym.

**Score Tensor Used**: `antonym_action_gt_scores`

**Metrics**:
- `test/adverb_constrained/top1_accuracy`
  - Top-1 accuracy on binary adverb choice
  - Micro-averaged (all samples weighted equally)

- `test/adverb_constrained/balanced_accuracy`
  - Macro-averaged accuracy (each adverb class weighted equally)
  - Useful when adverb classes are imbalanced

- `test/adverb_constrained/top5_accuracy`
  - Top-5 accuracy (always 1.0 for binary choice, kept for consistency)

**Example**:
```
Video shows someone running quickly
GT: ("quickly", "run")
Model sees: [("quickly", "run"), ("slowly", "run")]
Correct if: Model ranks ("quickly", "run") higher
```

### 2. Unconstrained Adverb Metrics
**Task**: Predict the correct adverb from all possible (action, adverb) pairs.

**Score Tensor Used**: `scores` (raw, unmasked)

**How it works**:
1. Find the highest-scoring pair: `argmax(scores)`
2. Extract the adverb component from that pair
3. Check if it matches the ground truth adverb

**Metrics**:
- `test/adverb/top1_accuracy`
  - Top-1 adverb accuracy
  - Extracts adverb from the best-scoring pair

- `test/adverb/balanced_accuracy`
  - Macro-averaged across adverb classes

- `test/adverb/top5_accuracy`
  - Checks if correct adverb appears in top-5 pairs

**Example**:
```
Video shows someone running quickly
GT: ("quickly", "run")
Model sees: ALL (adverb, action) pairs
Top prediction: ("quickly", "walk")
Result: CORRECT for adverb, WRONG for pair
```

### 3. Action Metrics
**Task**: Predict the correct action from all possible (action, adverb) pairs.

**Score Tensor Used**: `scores` (raw, unmasked)

**How it works**:
1. Find the highest-scoring pair: `argmax(scores)`
2. Extract the action component from that pair
3. Check if it matches the ground truth action

**Metrics**:
- `test/action/top1_accuracy`
  - Top-1 action accuracy
  - Extracts action from the best-scoring pair

- `test/action/top5_accuracy`
  - Checks if correct action appears in top-5 pairs

**Example**:
```
Video shows someone running quickly
GT: ("quickly", "run")
Model sees: ALL (adverb, action) pairs
Top prediction: ("slowly", "run")
Result: CORRECT for action, WRONG for adverb
```

### 4. Pair Metrics
**Task**: Predict the exact (action, adverb) combination.

**Score Tensor Used**: `scores` (raw, unmasked)

**How it works**:
1. Find the highest-scoring pair: `argmax(scores)`
2. Check if BOTH action AND adverb match ground truth

**Metrics**:
- `test/pair/top1_accuracy`
  - Top-1 exact pair accuracy
  - Both components must be correct
  - **Note**: `pair_acc ≤ min(adverb_acc, action_acc)`

- `test/pair/top5_accuracy`
  - Checks if exact correct pair appears in top-5

**Example**:
```
Video shows someone running quickly
GT: ("quickly", "run")
Model sees: ALL (adverb, action) pairs
Top prediction: ("quickly", "run")
Result: CORRECT for both components ✓
```

## Metric Relationships

### Expected Relationships:
```
pair_accuracy ≤ min(adverb_accuracy, action_accuracy)
```
The pair must have both components correct, so it can't exceed either individual accuracy.

```
adverb_constrained_accuracy ≥ adverb_unconstrained_accuracy
```
Constrained task is easier (binary choice with known action) than unconstrained (choose from all pairs).

```
top5_accuracy ≥ top1_accuracy
```
Top-5 is more lenient than top-1.

### Debugging Guide:

**If action accuracy ≈ 100%**:
- Model has learned actions very well
- Check if dataset has few actions or strong action bias

**If adverb_constrained ≫ adverb_unconstrained**:
- Model struggles to predict actions
- Consider the action features may be weak

**If pair_accuracy ≈ adverb_accuracy × action_accuracy**:
- Action and adverb predictions are independent
- Model may not be learning action-adverb interactions

**If top5 ≈ top1**:
- Model is very confident in predictions
- Or: Limited diversity in top predictions

## Implementation Details

### Top-K Calculation
```python
# Top-1
pair_pred = np.argmax(scores.numpy(), axis=1)

# Top-5
top5_pred = np.argsort(scores.numpy(), axis=1)[:, -5:]
```

### Balanced Accuracy (Macro-averaged)
```python
# Per-class accuracies
per_class_accs = [accuracy for samples in each class]

# Average across classes (not samples)
balanced_acc = mean(per_class_accs)
```

### Pair Extraction
Given a pair index `i` from the predictions:
```python
# Pairs are structured as: (adverb, action)
adverb = dset.pairs[i][0]  # First element
action = dset.pairs[i][1]  # Second element
```

## Code Locations

- **Metric implementations**: `utils.py:26-95`
- **Test function**: `train.py:292-410`
- **Evaluator class**: `model.py:207-248`

## Changes from Original Implementation

### Fixed Bugs:
1. **Action accuracy was broken**: Originally used `action_gt_scores` (pre-masked with GT action), giving artificially high accuracy. Now correctly uses raw `scores`.

### Added Metrics:
1. Unconstrained adverb metrics (separate from constrained)
2. Action metrics (previously broken)
3. Pair metrics (new)
4. Top-5 variants for all metrics

### Improved Naming:
- WandB logs now use hierarchical naming: `test/{metric_type}/{variant}`
- Clear distinction between constrained and unconstrained tasks
- Consistent naming across TensorBoard and WandB
