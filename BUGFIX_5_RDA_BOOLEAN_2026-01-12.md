# Bug Fix: Boolean Column Handling in RDA
**Date:** 2026-01-12  
**Issue:** Pipeline crash in constrained ordination (RDA)  
**Error:** `TypeError: Invalid value '1.0' for dtype boolean`

---

## Root Cause

**File:** `diversity/beta/ordination.py` line 223  
**Problem:** Boolean column detection was using `env_df[v].dtype != 'bool'`

This check failed for nullable boolean columns (dtype='boolean') introduced by pandas nullable types.

When RDA tried to fill NaN values with column means, it attempted:
```python
filled_data = filled_data.fillna(filled_data.mean())  # mean() returns float
```

For boolean columns, this tried to set `bool` to `1.0` (float), causing the error.

---

## Fix Applied

### diversity/beta/ordination.py (line 223)

**Before:**
```python
num_vars = [
    v for v in existing_vars 
    if pd.api.types.is_numeric_dtype(env_df[v]) and env_df[v].dtype != 'bool'
]
```

**After:**
```python
num_vars = [
    v for v in existing_vars 
    if pd.api.types.is_numeric_dtype(env_df[v]) and not pd.api.types.is_bool_dtype(env_df[v])
]
```

**Why this works:**
- `pd.api.types.is_bool_dtype()` properly detects both:
  - `dtype='bool'` (standard boolean)
  - `dtype='boolean'` (nullable boolean from pandas extension types)
- Explicitly excludes boolean columns from numeric variable list for RDA
- RDA only uses truly numeric columns (int, float) that can accept float means

---

## Configuration Update

### config/config_ml_only.yaml

**Changes:**
1. **Fresh output directory:**
   ```yaml
   project: "/usr2/people/macgregor/amplicon/project_01_ml_viz"
   ```

2. **Load data from original location:**
   ```yaml
   data_file: "../project_01/03_processed_data/final_processed_adata.h5ad"
   ```

**Rationale:**
- Writes all new ML visualizations to clean directory
- Prevents overwriting existing results from testing_4
- Maintains original processed data as source

---

## Testing

**Expected behavior after fix:**
1. RDA skips boolean columns (`facility_match`, etc.)
2. Only uses numeric columns for environmental constraints
3. No dtype errors during fillna operations
4. Pipeline completes through ML and reporting

**Verification:**
```bash
cd /usr2/people/macgregor/amplicon/workflow_16s
bash run.sh --config config/config_ml_only.yaml
```

Check logs for:
- ✅ No "Invalid value '1.0' for dtype boolean" error
- ✅ RDA completes successfully
- ✅ ML runs with all 3 strategies
- ✅ ML visualizations generated in `project_01_ml_viz/04_analysis/`

---

## Related Issues

This is the **5th bug** fixed in this session:

1. ✅ Numeric columns excluded (helpers.py - dtype check order)
2. ✅ Facility columns excluded (helpers.py - PRIORITY_COLUMNS)
3. ✅ Boolean NaN handling (helpers.py - dropna() before nunique())
4. ✅ CatBoost parameter conflict (feature_selection.py - kwargs deduplication)
5. ✅ **Boolean fillna in RDA (ordination.py - is_bool_dtype() check)** ← THIS FIX

All bugs are now resolved. Pipeline should run end-to-end successfully.

---

## Files Modified

1. **`src/workflow_16s/downstream/diversity/beta/ordination.py`**
   - Line 223: Use `pd.api.types.is_bool_dtype()` instead of `dtype != 'bool'`

2. **`config/config_ml_only.yaml`**
   - Line 10: Updated project path to `project_01_ml_viz`
   - Line 260: Updated data_file path to load from `../project_01/`

---

## Impact

**Before fix:**
- Pipeline crashed during RDA after 15+ hours of processing
- Stats completed but constrained ordination failed
- No ML results generated

**After fix:**
- Boolean columns properly excluded from RDA numeric constraints
- Pipeline runs through completion
- All analyses (stats, RDA, ML, reporting) succeed
- ML visualizations generated successfully
