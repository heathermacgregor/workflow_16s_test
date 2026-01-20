# Overfitting Prevention & Model Validation

## Overview

This document describes the comprehensive overfitting prevention and validation methods implemented in the workflow to ensure models generalize well to unseen data. These methods are critical for production ML systems, especially in microbiome research where batch effects and confounders are common.

---

## Problem Statement

**Overfitting** occurs when a model learns patterns specific to the training data (including noise and batch effects) rather than true biological signals. Symptoms include:
- ✗ High training accuracy but poor test performance
- ✗ Models that don't generalize to new datasets
- ✗ Inflated feature importances from spurious correlations
- ✗ Batch-specific patterns instead of true biological signals

**Nuclear contamination MCC of 0.973** is suspiciously high and warrants validation to ensure it reflects true biological signal rather than overfitting.

---

## Implemented Prevention Methods

### 1. **Model Complexity Constraints** (RandomForest)

**Location:** `machine_learning.py` lines 78-95

**Strategy:** Limit model capacity to prevent memorization

**Parameters:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,              # Limit tree depth (was unlimited)
    min_samples_split=10,      # Require 10 samples to split (was 2)
    min_samples_leaf=5,        # Require 5 samples per leaf (was 1)
    max_features='sqrt',       # Only sqrt(n_features) per split (was all)
    random_state=42,
    oob_score=True,
    n_jobs=-1
)
```

**Impact:**
- Prevents trees from memorizing individual samples
- Forces learning of generalizable patterns
- Reduces variance at cost of slight bias increase

**Validation:** Compare OOB score vs test score - gap > 0.10 triggers warning

---

### 2. **CatBoost Overfitting Detector** (CatBoost)

**Location:** `models/feature_selection/core.py` lines 25-140

**Strategy:** Monitor validation score during training and stop when overfitting begins

**Implementation:**
```python
CatBoostClassifier(
    od_type='Iter',               # Overfitting detector type
    od_wait=50,                   # Stop if no improvement for 50 iterations
    early_stopping_rounds=50,     # Early stopping threshold
    use_best_model=True           # Revert to best iteration, not final
)
```

**Validation:**
- Tracks train/validation gap across CV folds
- Reports overfitting gap: Train - Val score
- Warns if gap > 0.15 (high overfitting) or > 0.10 (moderate)
- Logs best iteration vs total iterations

**Example Output:**
```
Final Model Performance:
  Training Score: 0.932
  Test Score: 0.889
  Overfitting Gap: 0.043
  ✓ Good generalization (gap < 0.10)
  Best Iteration: 87 / 150
  ✓ Early stopping worked well (stopped at 58%)
```

---

### 3. **Nested Cross-Validation** (Both)

**Location:** `models/overfitting_prevention.py` lines 22-155

**Strategy:** Unbiased performance estimation with separate hyperparameter tuning

**Architecture:**
```
Outer Loop (5 folds):
  ├─ Split data into train/test
  ├─ Inner Loop (3 folds on train only):
  │   ├─ Hyperparameter tuning
  │   └─ Select best parameters
  ├─ Train model with best params on full train
  └─ Evaluate on held-out test (never seen during tuning)
```

**Metrics:**
- **Outer CV Score:** Unbiased estimate of generalization performance
- **Inner CV Score:** Optimistic estimate from hyperparameter search
- **Overfitting Gap:** Inner - Outer (should be < 0.10)

**When Triggered:**
- Automatically for priority targets (facility-related variables)
- For any model with suspiciously high performance (score > 0.9)
- For models with OOB-Test gap > 0.10

**Example Output:**
```
Nested CV Results (MCC):
  Outer CV (Unbiased): 0.873 ± 0.045
  Inner CV (Optimistic): 0.921
  Overfitting Gap: 0.048
  ✓ Model generalization appears good (gap < 0.10)
```

---

### 4. **Learning Curves**

**Location:** `models/overfitting_prevention.py` lines 158-282

**Strategy:** Diagnose overfitting by varying training set size

**Interpretation:**
- **High train, low val:** Overfitting - need regularization
- **Both low:** Underfitting - need more complexity
- **Both high and converging:** Good fit

**Visualization:** Interactive HTML plot showing:
- Training score vs dataset size
- Validation score vs dataset size
- ±1 std deviation bands
- Final gap quantification

**When Triggered:** Same as nested CV

**Example:**
```
Learning Curve Gap: 0.067 (Train: 0.945, Val: 0.878)
✓ Good fit: Training-Validation gap = 0.067 < 0.10
```

---

### 5. **Permutation Tests**

**Location:** `models/overfitting_prevention.py` lines 285-352

**Strategy:** Verify features are truly predictive, not random correlations

**Method:**
1. Record actual model performance
2. Shuffle target labels (destroys true signal)
3. Retrain model 100 times with shuffled labels
4. Compare actual score to shuffled distribution
5. Calculate p-value: probability result is due to chance

**Interpretation:**
- **p < 0.001:** Highly significant - features are predictive
- **p < 0.05:** Significant - features likely predictive
- **p ≥ 0.05:** **Not significant** - model may be overfitting!

**When Triggered:** For priority targets (optional for others due to computational cost)

**Example:**
```
Permutation Test Results:
  Actual MCC: 0.889
  Permuted MCC: 0.023 ± 0.045
  p-value: 0.0001
  ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)
```

---

### 6. **Stability Selection**

**Location:** `models/overfitting_prevention.py` lines 355-453

**Strategy:** Identify features that are consistently important across bootstrap samples

**Method:**
1. Create 50-100 bootstrap samples
2. Train model on each sample
3. Record top 20% features per sample
4. Calculate selection frequency for each feature
5. Mark features as "stable" if selected ≥70% of time

**Purpose:**
- Features selected once may be spurious
- Features selected consistently are likely real signals
- Reduces false discovery rate

**When Triggered:** For priority targets with feature importances available

**Example:**
```
Stability Selection Results:
  Total features: 250
  Stable features (freq >= 0.70): 18
  Top 5 stable features:
    - Aquicella: 94%
    - Cellulomonas: 89%
    - Sphingomonas: 87%
    - Bdellovibrio: 82%
    - Aquisphaera: 78%
```

---

### 7. **Batch-Aware Cross-Validation**

**Location:** `models/feature_selection/validation.py` + `core.py`

**Strategy:** Use GroupKFold to ensure batches never split across train/test

**Problem:** If samples from the same batch appear in both train and test, model can memorize batch effects rather than learning biology

**Solution:**
```python
# Before: Random split can put same batch in train AND test
X_train, X_test = train_test_split(X, y)  # BAD

# After: Batches strictly isolated
cv = GroupKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=batch_labels):
    # All samples from batch_A only in train OR test, never both
```

**Impact:**
- Prevents "batch memorization"
- Forces model to learn cross-batch generalizable patterns
- More conservative performance estimates (good!)

---

## Integration into Workflow

### RandomForest (machine_learning.py)

**Automatic Checks:**
1. OOB vs Test score gap computed for all models
2. Gap > 0.10 → Warning logged
3. Gap > 0.15 → Overfitting warning

**Triggered Validation:** (`validate_overfitting=True` by default)
- Priority targets (facility variables)
- Suspiciously high scores (>0.9)
- Detected overfitting (gap > 0.10)

**Runs:**
- Nested cross-validation
- Learning curves
- Permutation tests (if not quick_validation)
- Stability selection (if not quick_validation)

**Output:**
- `04_analysis/plots/machine_learning/overfitting_validation/{target}/`
  - `learning_curves.html` - Interactive plot
  - `stability_selection.csv` - Feature stability table

---

### CatBoost (feature_selection/core.py)

**Built-in Protection:**
- Overfitting detector (`od_type='Iter'`)
- Early stopping (50 rounds)
- Use best model (not final)

**Cross-Validation:**
- GroupKFold if batch info available
- Tracks train/val gap per fold
- Reports final train/test gap

**Logs:**
- Per-fold overfitting warnings
- Final gap assessment
- Best iteration statistics

---

## Configuration Options

### Quick Validation Mode

For faster (but less thorough) validation:

```python
run_machine_learning_analysis(
    ...,
    validate_overfitting=True,
    quick_validation=True  # Reduces iterations
)
```

**Changes:**
- Nested CV: 3 outer folds instead of 5
- Learning curves: 3 CV folds instead of 5
- Permutation test: 50 iterations instead of 100
- Stability selection: 25 bootstraps instead of 50

---

### Disable Validation

To skip validation (not recommended for production):

```python
run_machine_learning_analysis(
    ...,
    validate_overfitting=False
)
```

---

## Interpreting Results

### Good Model (No Overfitting)
```
✓ OOB-Test gap: 0.045
✓ Nested CV gap: 0.038
✓ Learning curve gap: 0.052
✓ Permutation p-value: 0.0001
✓ Stable features: 22 / 50
→ Model generalizes well, features are real
```

### Moderate Overfitting
```
⚠️  OOB-Test gap: 0.123
⚠️  Nested CV gap: 0.108
⚠️  Learning curve gap: 0.115
✓ Permutation p-value: 0.003
⚠️  Stable features: 8 / 50
→ Model overfits somewhat, but features are likely real
→ Consider: More regularization, feature selection
```

### High Overfitting
```
⚠️  OOB-Test gap: 0.287
⚠️  Nested CV gap: 0.245
⚠️  Learning curve gap: 0.301
✗ Permutation p-value: 0.089 (NOT SIGNIFICANT)
✗ Stable features: 2 / 50
→ Model severely overfits, features may be spurious
→ Action required: Increase regularization, reduce complexity, collect more data
```

---

## Nuclear Contamination Validation

**Original MCC: 0.973** - Requires validation!

**Expected Validation:**
1. Run nested CV → Should be 0.85-0.95 if real
2. Check learning curves → Train/val should converge
3. Permutation test → p < 0.001 expected
4. Stability selection → Aquicella, Cellulomonas should be stable

**If validation passes:**
- ✓ MCC 0.973 is real biological signal
- ✓ Nuclear contamination has strong microbial signature
- ✓ Features (Aquicella, etc.) are true biomarkers

**If validation fails:**
- ✗ MCC 0.973 is inflated by overfitting
- ✗ May be memorizing batch or study-specific patterns
- ✗ Need more regularization or cross-study validation

---

## Recommendations

### For All Models
1. **Always check OOB vs Test gap** - First line of defense
2. **Run validation for important targets** - Priority variables
3. **Review stability selection** - Are top features consistent?
4. **Check permutation p-values** - Is model better than random?

### For Production
1. **Enable full validation** - Use `validate_overfitting=True, quick_validation=False`
2. **Archive validation outputs** - Keep HTML plots and CSVs
3. **Document decisions** - Log why high-performing models are trustworthy
4. **Cross-study validation** - Test on independent datasets when possible

### For High-Stakes Predictions
1. **Nested CV is mandatory** - Unbiased estimate
2. **Permutation test must be significant** - p < 0.01
3. **Stability selection ≥50% stable features** - Robust biomarkers
4. **External validation** - Test on completely independent data

---

## Session Improvements

**#25: Comprehensive Overfitting Prevention**

**Added:**
1. Model complexity constraints (RandomForest)
2. CatBoost overfitting detector integration
3. Nested cross-validation module
4. Learning curve diagnostics
5. Permutation testing framework
6. Stability selection analysis
7. Automated validation triggers
8. Comprehensive logging and warnings

**Files Modified:**
- `machine_learning.py` - RandomForest validation integration
- `models/feature_selection/core.py` - CatBoost overfitting detection
- `models/overfitting_prevention.py` - NEW validation module (600+ lines)

**Impact:**
- Models now have multiple overfitting checkpoints
- Automated warnings for suspicious performance
- Publication-quality validation methods
- Reduced risk of false discoveries

---

## References

- **Nested CV:** Varma & Simon (2006) "Bias in error estimation when using cross-validation for model selection"
- **Stability Selection:** Meinshausen & Bühlmann (2010) "Stability selection"
- **Permutation Tests:** Ojala & Garriga (2010) "Permutation tests for studying classifier performance"
- **Learning Curves:** Perlich et al. (2003) "Tree induction vs. logistic regression: A learning-curve analysis"
- **GroupKFold:** Scikit-learn documentation on cross-validation strategies

---

## Next Steps

1. **Run full validation** on nuclear contamination model
2. **Archive results** for reproducibility
3. **Update publication methods** with validation details
4. **Consider external validation** on independent nuclear facility datasets
5. **Monitor production models** with these metrics over time
