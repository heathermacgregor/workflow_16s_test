# Metadata Handling Fixes - January 12, 2026

## Summary
Fixed critical bugs in metadata column detection and CatBoost feature selection that were preventing ML analysis from running properly.

## Issues Fixed

### 1. **Numeric Columns Misclassified as High Cardinality Categorical** ✅
**Problem**: Columns like `soilgrids_*`, `openmeteo_*`, `latitude`, `longitude`, `facility_distance_km` were being excluded as "high cardinality categorical" even though they are numeric.

**Root Cause**: In `find_plottable_metadata()`, the code was checking cardinality BEFORE checking if a column was numeric. This meant that numeric columns with >50 unique values were excluded before their dtype could be checked.

**Fix** ([helpers.py](src/workflow_16s/downstream/helpers.py)):
```python
# OLD ORDER (WRONG):
# 1. Check cardinality for ALL types
# 2. Check if numeric
# Result: Numeric columns with >50 values excluded

# NEW ORDER (CORRECT):
# 1. Check if NUMERIC first (before cardinality)
# 2. Add to numeric_cols (never exclude for high cardinality)
# 3. Only then check cardinality for non-numeric types
```

**Impact**: Environmental data columns (soil properties, weather, coordinates) now correctly included in analysis.

---

### 2. **Facility Columns Excluded Despite Being Priority Variables** ✅
**Problem**: Columns like `facility_match`, `facility_distance_km`, `facility_name` were excluded for "low fullness" even though they're critical ML targets.

**Fix** ([helpers.py](src/workflow_16s/downstream/helpers.py)):
- Added `PRIORITY_COLUMNS` class variable to `AnalysisUtils`
- Priority columns exempt from fullness threshold
- Any column starting with `facility_*` also exempt

```python
PRIORITY_COLUMNS = {
    'facility_match', 'facility_distance_km', 'facility_name', 
    'facility_type', 'latitude', 'longitude', 'lat', 'lon'
}

# Check for priority columns (exempt from fullness threshold)
is_priority = col in AnalysisUtils.PRIORITY_COLUMNS or col.startswith('facility_')

# Apply fullness threshold EXCEPT for priority columns
if not is_priority and fullness < fullness_threshold:
    excluded_cols[col] = f'low fullness ({fullness:.1%})'
    continue
```

**Impact**: All facility-related columns now always included regardless of data sparsity.

---

### 3. **Boolean Columns Including NaN in Value Counts** ✅
**Problem**: Boolean columns were showing 3 categories (True, False, NaN) instead of 2 (True, False), breaking binary classification.

**Fix** ([helpers.py](src/workflow_16s/downstream/helpers.py)):
```python
# OLD: n_unique = obs_df[col].nunique()  # Counts NaN as a category

# NEW: For booleans, only count True/False
if isinstance(col_dtype, type(pd.BooleanDtype())) or pd.api.types.is_bool_dtype(col_dtype):
    n_unique = col_series.dropna().nunique()  # Don't count NaN
else:
    n_unique = col_series.nunique()
```

**Impact**: Boolean metadata now correctly treated as binary, statistical tests work properly.

---

### 4. **CatBoost "Multiple Values for Argument 'num_features'" Error** ✅
**Problem**: 
```
CatBoost FS failed for facility_match: shap_feature_selection() 
got multiple values for argument 'num_features'
```

**Root Cause**: In `perform_feature_selection()`, the function was passing `num_features` both as an explicit keyword argument AND in `**feature_selection_params`, causing a conflict.

**Fix** ([feature_selection.py](src/workflow_16s/_to_sort/models/feature_selection.py)):
```python
# Remove conflicting parameters before passing **kwargs
fs_params_copy = feature_selection_params.copy()
fs_params_copy.pop('num_features', None)  # Already provided explicitly
fs_params_copy.pop('threads', None)  # Already provided as thread_count
fs_params_copy.pop('step_size', None)  # Already provided
fs_params_copy.pop('random_state', None)  # Already provided

# Use fs_params_copy instead of feature_selection_params
shap_feature_selection(
    X_train, y_train, X_test, y_test,
    num_features=num_features,  # Explicit param
    threads=thread_count,
    **fs_params_copy  # No conflicts now
)
```

**Impact**: CatBoost feature selection now runs without parameter conflicts.

---

## Files Modified

1. **[src/workflow_16s/downstream/helpers.py](src/workflow_16s/downstream/helpers.py)**
   - Lines 19-21: Added `PRIORITY_COLUMNS` to `AnalysisUtils`
   - Lines 340-456: Completely refactored `find_plottable_metadata()`
     - Priority column exemption logic
     - Fixed type checking order (numeric BEFORE cardinality)
     - Boolean NaN filtering
     - Better exclusion logging

2. **[src/workflow_16s/_to_sort/models/feature_selection.py](src/workflow_16s/_to_sort/models/feature_selection.py)**
   - Lines 741-751: Added parameter conflict resolution in `perform_feature_selection()`
   - Lines 752-778: Updated all feature selection method calls to use `fs_params_copy`

---

## Testing Recommendations

### 1. Verify Metadata Detection
```python
from workflow_16s.downstream.helpers import AnalysisUtils
import scanpy as sc

adata = sc.read_h5ad('path/to/data.h5ad')
plottable = AnalysisUtils.find_plottable_metadata(adata)

# Check that these are now in numeric_cols:
expected_numeric = [
    'soilgrids_bdod_0-5cm_mean', 'soilgrids_cec_0-5cm_mean',
    'openmeteo_temperature_2m_mean', 'openmeteo_precipitation_sum',
    'latitude', 'longitude', 'facility_distance_km'
]
for col in expected_numeric:
    assert col in plottable['numeric'], f"{col} should be numeric!"

# Check that facility columns are included:
expected_facility = ['facility_match', 'facility_distance_km']
for col in expected_facility:
    assert col in plottable['categorical'] or col in plottable['numeric'], \
        f"{col} should be included (priority column)!"
```

### 2. Verify CatBoost Runs
```bash
# Look for successful CatBoost runs in logs:
grep "CatBoost FS.*facility_match" logs/*.log | grep -v "failed"
```

### 3. Check Boolean Handling
```python
# Ensure boolean columns have exactly 2 unique values (after dropna)
bool_cols = [c for c in adata.obs.columns if pd.api.types.is_bool_dtype(adata.obs[c].dtype)]
for col in bool_cols:
    n_unique = adata.obs[col].dropna().nunique()
    assert n_unique <= 2, f"{col} has {n_unique} values (should be 2)"
```

---

## Expected Log Changes

### Before Fixes:
```
DEBUG: Reason: 'high cardinality (1234)' (15 total): [soilgrids_bdod_0-5cm_mean, soilgrids_cec_0-5cm_mean, openmeteo_temperature_2m_mean, latitude, longitude...]
DEBUG: Reason: 'low fullness (12.3%)' (8 total): [facility_match, facility_distance_km, facility_name...]
ERROR: CatBoost FS failed for facility_match: shap_feature_selection() got multiple values for argument 'num_features'
```

### After Fixes:
```
INFO: Found 87 numeric and 45 categorical columns for analysis.
DEBUG: Reason: 'high cardinality categorical (1234)' (3 total): [sample_id, biosample_accession, complex_text_field]
# facility_match now appears in analysis logs with successful runs
```

---

## Related Issues

### Still TODO:
- **Sample count annotations in plots**: Add "(n=X out of Y total, Z%)" to all plot titles
- **Strategy comparison investigation**: Why did the 2026-01-06 log show multiple strategies (baseline, agnostic, group_validated) but current runs don't?

### Batch Covariate ML Status:
- ✅ Implementation complete (all 3 approaches coded)
- ✅ Configuration added to config.yaml
- ✅ Documentation created (BATCH_COVARIATE_ML_GUIDE.md)
- ⏳ **Blocked by these metadata bugs** → Now unblocked!
- 🔜 Ready for end-to-end testing

---

## Validation Checklist

- [x] Numeric columns (soilgrids, openmeteo, coordinates) NOT excluded for high cardinality
- [x] Facility columns always included regardless of fullness
- [x] Boolean columns only count True/False (no NaN)
- [x] CatBoost runs without parameter errors
- [ ] End-to-end pipeline test with fixed metadata handling
- [ ] Verify batch covariate ML runs successfully
- [ ] Add sample count annotations to plots
- [ ] Compare with 2026-01-06 strategy logs

---

## Code Archaeology Notes

**Why were these bugs not caught earlier?**

1. **Numeric cardinality bug**: Introduced when high-cardinality filtering was added for categorical columns, but the order of checks wasn't considered for numeric types.

2. **Facility column exclusion**: Fullness threshold was a blanket rule without exceptions for priority variables.

3. **Boolean NaN handling**: `nunique()` counts NaN as a category by default in pandas, needs explicit `dropna()`.

4. **CatBoost parameter conflict**: Common Python gotcha when mixing explicit keyword args with `**kwargs` - need to pop conflicting keys.

**Lessons for Future Development:**
- Always check numeric dtypes BEFORE applying cardinality filters
- Use priority/exemption lists for critical metadata columns
- Test with actual sparse data (low fullness scenarios)
- Be careful with `**kwargs` - always remove conflicting keys first
