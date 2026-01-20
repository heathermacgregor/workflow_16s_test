# Bugfix #6: Data Loading from Existing File (2026-01-12)

## Issue
The `config_ml_only.yaml` configuration included `load_existing_data: True` and `data_file: "../project_01/03_processed_data/final_processed_adata.h5ad"`, but the `run_fast_load()` function did not respect these settings.

**Result**: Pipeline would fail because it tries to load from empty `project_01_ml_viz/03_processed_data/` instead of using the existing processed data from `project_01/`.

## Root Cause
`run_fast_load()` in [ingestion.py](src/workflow_16s/downstream/steps/ingestion.py) immediately tries to glob for `*.h5ad` files in `workflow.data_dir` without checking:
1. `downstream.load_existing_data` config option
2. `downstream.data_file` path

```python
# OLD CODE (Lines 358-378):
def run_fast_load(workflow):
    workflow.logger.info("1. Modular Ingestion: Loading and filtering files...")
    h5ad_files = list(workflow.data_dir.glob("*.h5ad"))
    if not h5ad_files: return
    # ... rest of concatenation logic
```

## Fix
Added config-aware data loading logic at the start of `run_fast_load()`:

```python
# NEW CODE:
def run_fast_load(workflow):
    # Check if we should load existing processed data instead
    downstream_config = workflow.config.get("downstream", {})
    load_existing = downstream_config.get("load_existing_data", False)
    data_file = downstream_config.get("data_file", None)
    
    if load_existing and data_file:
        workflow.logger.info("1. Loading existing processed data from config...")
        # Resolve relative path
        data_path = Path(data_file)
        if not data_path.is_absolute():
            config_dir = Path(workflow.config.get("paths", {}).get("base", "."))
            data_path = (config_dir / data_file).resolve()
        
        if not data_path.exists():
            workflow.logger.error(f"Configured data file not found: {data_path}")
            workflow.logger.info("Falling back to standard data loading...")
        else:
            workflow.logger.info(f"Loading from: {data_path}")
            workflow.adata = sc.read_h5ad(data_path)
            workflow.logger.info(f"✅ Loaded existing data: {workflow.adata.n_obs} samples × {workflow.adata.n_vars} features")
            
            # Verify taxonomy columns
            tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
            missing_tax = [lvl for lvl in tax_levels if lvl not in workflow.adata.var.columns]
            if missing_tax:
                workflow.logger.warning(f"Missing taxonomy columns: {missing_tax}")
            else:
                workflow.logger.info(f"✅ All taxonomy columns present")
            return  # Skip standard loading
    
    # Standard loading from multiple h5ad files
    workflow.logger.info("1. Modular Ingestion: Loading and filtering files (PARALLEL + CACHED)...")
    h5ad_files = list(workflow.data_dir.glob("*.h5ad"))
    if not h5ad_files:
        workflow.logger.warning(f"No h5ad files found in {workflow.data_dir}")
        return
    # ... rest of concatenation logic
```

## Testing
```bash
cd /usr2/people/macgregor/amplicon/workflow_16s
bash run.sh --config config/config_ml_only.yaml
```

Expected behavior:
1. Reads config: `load_existing_data: True`, `data_file: "../project_01/03_processed_data/final_processed_adata.h5ad"`
2. Resolves path: `/usr2/people/macgregor/amplicon/project_01/03_processed_data/final_processed_adata.h5ad`
3. Loads existing data directly (skip concatenation)
4. Logs: "✅ Loaded existing data: 19900 samples × 541386 features"
5. Proceeds to ML analysis

## Impact
- **Critical**: Without this fix, config_ml_only.yaml cannot run
- **Performance**: Saves 22+ minutes by skipping concatenation
- **Reliability**: Ensures ML-only configs work as designed

## Related
- Bug #1-3: Metadata filtering issues (helpers.py)
- Bug #4: CatBoost parameter conflict (feature_selection.py)
- Bug #5: RDA boolean dtype (ordination.py)

## Files Changed
- `src/workflow_16s/downstream/steps/ingestion.py` (lines 358-411)
