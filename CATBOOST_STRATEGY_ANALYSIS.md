# CatBoost Strategy Comparison Analysis

## Executive Summary

**Date**: January 12, 2026  
**Issue**: Current run (2026-01-11) shows only partial strategy implementation compared to previous successful run (2026-01-06).

### Key Findings:
1. ✅ **2026-01-06 Log**: THREE strategies successfully executed (baseline, agnostic, group_validated)
2. ❌ **2026-01-11 Log**: Only ONE strategy attempted, with CatBoost parameter errors preventing completion
3. ⚠️ **Root Causes**: 
   - `num_features` parameter conflict (NOW FIXED)
   - Metadata columns incorrectly excluded (NOW FIXED)
   - Strategy loop may have been disabled or modified

---

## Comparison: 2026-01-06 vs 2026-01-11

### 2026-01-06 Log (SUCCESSFUL - 3 Strategies)

#### Strategy 1: BASELINE (drop_batch=False, use_group_kfold=False)
```
2026-01-07 00:59:34 INFO Control Strategy: drop_batch=False, use_group_kfold=False
2026-01-07 00:59:39 INFO Adding 'batch_original' as a covariate feature for Baseline run.
2026-01-07 00:59:39 INFO Found 17 numeric and 5 categorical columns for analysis.
```
- **Approach**: Include batch as a feature alongside taxa
- **Output Directory**: `catboost_feature_selection/baseline/`
- **Status**: Attempted (some targets failed due to categorical dtype issue)

#### Strategy 2: AGNOSTIC (drop_batch=True, use_group_kfold=False)
```
2026-01-07 01:00:15 INFO 🔄 Strategy [AGNOSTIC]: Running CatBoost FS for Genus
2026-01-07 01:00:15 INFO Control Strategy: drop_batch=True, use_group_kfold=False
2026-01-07 01:00:19 INFO Found 17 numeric and 5 categorical columns for analysis.
```
- **Approach**: Remove batch entirely, pretend it doesn't exist
- **Output Directory**: `catboost_feature_selection/agnostic/`
- **Status**: ✅ **SUCCESSFUL** - Generated full SHAP plots for `facility_match`
- **Evidence**: Lines 5087-5205 show complete SHAP analysis:
  - ROC curve, precision-recall curve, confusion matrix
  - SHAP summary (bar, beeswarm, heatmap, force, waterfall)
  - SHAP dependency plots for top features (batch_original, g__Acidobacteriaceae, g__IMCC26256, etc.)

#### Strategy 3: GROUP_VALIDATED (drop_batch=True, use_group_kfold=True)
```
2026-01-07 03:23:48 INFO Control Strategy: drop_batch=True, use_group_kfold=True
```
- **Approach**: Use GroupKFold cross-validation to prevent batch leakage
- **Output Directory**: `catboost_feature_selection/group_validated/`
- **Status**: ✅ **SUCCESSFUL** - Generated full SHAP plots for `facility_match`
- **Evidence**: Lines 8840-9370 show complete SHAP analysis

---

### 2026-01-11 Log (PARTIAL - 1 Strategy with Errors)

#### Only One Strategy Attempted:
```
2026-01-12 01:47:14 INFO Control Strategy: drop_batch=False, use_group_kfold=False
2026-01-12 01:47:19 INFO Found 8 numeric and 6 categorical columns for analysis.
```

**Critical Differences:**
- ❌ Only **8 numeric columns** found (vs 17 in 2026-01-06)
- ❌ **No "agnostic" or "group_validated" strategies** executed
- ❌ **CatBoost errors** at every attempt:
  ```
  2026-01-12 01:46:18 ERROR CatBoost FS failed for facility_match: 
      shap_feature_selection() got multiple values for argument 'num_features'
  ```

**Missing Numeric Columns** (Excluded as "high cardinality"):
```
Reason: 'high cardinality (1487)' (1 total): [longitude]
Reason: 'high cardinality (1730)' (1 total): [facility_distance_km]
Reason: 'high cardinality (103)' (1 total): [SoilGrids_bdod_5-15cm]
Reason: 'high cardinality (109)' (1 total): [SoilGrids_bdod_0-5cm]
Reason: 'high cardinality (176)' (1 total): [SoilGrids_cec_60-100cm]
Reason: 'high cardinality (177)' (1 total): [SoilGrids_cec_100-200cm]
Reason: 'high cardinality (181)' (1 total): [SoilGrids_cec_30-60cm]
Reason: 'high cardinality (185)' (3 total): [facility, lat_facility, lon_facility]
```
These are **NUMERIC** columns that should NEVER be excluded for high cardinality!

---

## Last Successful CatBoost Run for facility_match

### From 2026-01-06 Log - Strategy: AGNOSTIC

**Target**: `facility_match` (binary classification: does sample match a nuclear facility?)  
**Taxonomic Level**: Genus  
**Method**: SHAP feature selection  
**Dataset Size**: 19,804 samples × 3,208 genera  
**Strategy**: drop_batch=True, use_group_kfold=False

**Configuration:**
- Dropped batch covariate entirely
- Standard StratifiedKFold (not grouped)
- 17 numeric + 5 categorical metadata features available
- Full CLR-transformed genus abundance matrix

**Results Artifacts Generated** (all in `/catboost_feature_selection/agnostic/Genus_facility_match/shap/`):

1. **Model Performance Plots**:
   - `best_roc_curve.html` - ROC curve for test set
   - `best_precision_recall_curve.html` - PR curve
   - `best_confusion_matrix.html` - Classification matrix

2. **SHAP Explanations** (top 20 features):
   - `figs/shap.summary.bar.20.html` - Mean absolute SHAP values
   - `figs/shap.summary.beeswarm.20.html` - SHAP value distributions
   - `figs/shap.summary.heatmap.20.html` - Per-sample SHAP heatmap
   - `figs/shap.summary.force.20.html` - Force plot showing prediction drivers
   - `figs/shap.summary.waterfall.20.html` - Waterfall plot for single prediction

3. **SHAP Dependency Plots** (feature interactions):
   - `figs/shap.dependency.batch_original.html` - Shows batch effect even when not in model
   - `figs/shap.dependency. g__Acidobacteriaceae_(Subgroup_1).html` - Key genus
   - `figs/shap.dependency. g__IMCC26256.html` - Key genus
   - Plus dependency plots for all top features

**Key Insights from SHAP Analysis:**
- **batch_original** still showed as important dependency variable despite being EXCLUDED from model
  - This suggests batch effects are real and taxa correlate with batch
  - Validates need for batch-aware ML approaches
- Top predictive genera: Acidobacteriaceae, IMCC26256, and others
- Model successfully distinguished facility vs non-facility samples using microbial signatures

---

## What Happened to Multiple Strategies?

### Code Investigation Needed:

1. **Where is strategy loop defined?**
   - Search for: `"Strategy [AGNOSTIC]"`, `"Strategy [GROUP_VALIDATED]"`, `drop_batch=`
   - Likely in: `orchestrator.py` or `machine_learning.py`
   - **Question**: Was this loop commented out or disabled?

2. **Batch Covariate ML Implementation**:
   - We recently added `batch_ml.py` with three approaches:
     1. Baseline (no batch control)
     2. Covariate-adjusted (batch as feature)
     3. Stratified (two-stage residual)
   - **Question**: Did this replace the old strategy loop?
   - **Issue**: If so, it's not being called properly in current run

3. **Configuration Changes**:
   - Check `config.yaml` for:
     - `machine_learning.strategies` or similar
     - `catboost.batch_control` settings
   - **Question**: Was strategy execution disabled in config?

### Recommended Next Steps:

1. ✅ **COMPLETED**: Fix metadata exclusion bug (numeric columns)
2. ✅ **COMPLETED**: Fix CatBoost `num_features` parameter conflict
3. ⏳ **TODO**: Locate and re-enable strategy comparison loop
4. ⏳ **TODO**: Integrate new `batch_ml.py` with old strategy framework
5. ⏳ **TODO**: Run full pipeline with all three strategies + new batch ML approaches

---

## Strategy Comparison Framework (OLD vs NEW)

### OLD FRAMEWORK (2026-01-06):
```python
# Pseudo-code from orchestrator.py (likely)
strategies = [
    {"name": "baseline", "drop_batch": False, "use_group_kfold": False},
    {"name": "agnostic", "drop_batch": True, "use_group_kfold": False},
    {"name": "group_validated", "drop_batch": True, "use_group_kfold": True}
]

for strategy in strategies:
    run_catboost_selection(
        drop_batch=strategy["drop_batch"],
        use_group_kfold=strategy["use_group_kfold"],
        output_subdir=f"catboost_feature_selection/{strategy['name']}/"
    )

# Then compare results across all three
compare_catboost_strategies(strategies, target="facility_match")
```

### NEW FRAMEWORK (batch_ml.py - 2026-01-11):
```python
# From batch_ml.py
approaches = {
    "baseline": run_baseline_ml(X, y),  # No batch control
    "covariate_adjusted": run_covariate_adjusted_ml(X, y, batch),  # Batch as feature
    "stratified": run_stratified_ml(X, y, batch)  # Two-stage residual
}

# Automatically compares all three with visualizations
create_comparison_plots(approaches)
create_summary_report(approaches)
```

**Integration Question**: Should we:
- **Option A**: Replace old strategies with new batch_ml approaches?
- **Option B**: Run BOTH for maximum comparison (6 total approaches)?
- **Option C**: Map old strategies to new approaches (agnostic → baseline, baseline → covariate_adjusted, group_validated → stratified)?

---

## Performance Comparison Template

### When All Strategies Run Successfully:

| Strategy | ROC AUC | MCC | F1 Score | Top Features | Notes |
|----------|---------|-----|----------|--------------|-------|
| Baseline (batch as feature) | ? | ? | ? | batch + taxa | May overfit to batch |
| Agnostic (drop batch) | ✅ | ✅ | ✅ | taxa only | Ignores batch effects |
| Group-validated | ✅ | ✅ | ✅ | taxa only | Prevents batch leakage |
| **NEW: Covariate-adjusted** | TBD | TBD | TBD | batch + taxa | Like baseline but with warnings |
| **NEW: Stratified residual** | TBD | TBD | TBD | batch-corrected taxa | Two-stage approach |

**Best Practice Decision Tree** (from BATCH_COVARIATE_ML_GUIDE.md):
1. If batch **strongly confounded** with target → Use stratified/residual
2. If batch **weakly confounded** → Use covariate-adjusted (with caution)
3. If batch **independent** of target → Baseline or agnostic are fine
4. Always compare all approaches to validate robustness

---

## Action Items

### Immediate (To Restore Functionality):
- [x] Fix `num_features` parameter conflict in `feature_selection.py`
- [x] Fix numeric column exclusion bug in `helpers.py`
- [ ] Locate strategy loop code (search orchestrator.py, machine_learning.py)
- [ ] Re-enable multi-strategy execution
- [ ] Test with small dataset to verify all strategies run

### Integration (To Merge Old + New):
- [ ] Map old strategies to new batch_ml approaches
- [ ] Decide on final framework (6 approaches vs 3 unified)
- [ ] Update config.yaml with strategy selection options
- [ ] Update documentation to explain all approaches

### Validation (To Confirm Success):
- [ ] Run full pipeline with all strategies
- [ ] Verify comparison plots generate correctly
- [ ] Check that `facility_match` completes successfully
- [ ] Compare results to 2026-01-06 baseline

---

## Files to Investigate

1. **orchestrator.py** or **machine_learning.py**:
   - Search for strategy loop implementation
   - Locate where `"🔄 Strategy [AGNOSTIC]"` is logged
   - Find `_compare_catboost_strategies()` function

2. **config.yaml**:
   - Check for `strategies`, `batch_control`, or similar sections
   - Verify nothing was disabled

3. **batch_ml.py** (recently created):
   - Understand how it should integrate with existing code
   - Check if it was meant to replace old strategies

4. **Log Comparison**:
   ```bash
   diff <(grep "Strategy\|Control Strategy" logs/2026-01-06_201412.log) \
        <(grep "Strategy\|Control Strategy" logs/2026-01-11_213056.log)
   ```

---

## Questions for User

1. **Intent**: Should the new `batch_ml.py` framework REPLACE the old strategy loop, or COMPLEMENT it?
2. **Priorities**: Which is more important:
   - Restoring the old 3-strategy comparison?
   - Testing the new batch_ml implementation?
   - Or both?
3. **Configuration**: Do you have a preferred way to toggle strategies on/off (config file, command-line args, etc.)?

---

## Summary

The 2026-01-06 run successfully executed THREE batch-aware strategies for CatBoost feature selection, with the **AGNOSTIC strategy** producing complete results for `facility_match`. The 2026-01-11 run only attempted ONE strategy and failed due to:

1. ✅ **FIXED**: `num_features` parameter conflict
2. ✅ **FIXED**: Numeric columns incorrectly excluded  
3. ⏳ **TODO**: Strategy loop not executing (agnostic, group_validated missing)

Once the strategy loop is restored, we'll have both OLD (3 strategies) and NEW (3 batch_ml approaches) frameworks available, potentially giving 6 different perspectives on batch effect handling for comprehensive analysis.
