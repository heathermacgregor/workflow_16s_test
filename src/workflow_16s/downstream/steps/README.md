# Downstream Analysis Steps Module

This module provides a modular, step-based architecture for the 16S amplicon downstream analysis workflow.

## Architecture

The workflow is broken down into 5 main steps that can be executed independently or as part of the complete pipeline:

### 1. **Ingestion** (`ingestion.py`)
- `run_fast_load(workflow)`: Loads and concatenates multiple h5ad files
- `run_filter_empty(workflow)`: Filters samples with missing critical metadata
- `find_conda_env_by_substring(name, logger)`: Helper to locate conda environments

### 2. **Preprocessing** (`preprocessing.py`)
- `run_preprocessing_pipeline(workflow)`: Handles QC, filtering, and functional prediction
  - Quality control metrics
  - Depth and prevalence filtering
  - FAPROTAX functional annotation
  - FASTA and abundance table export
  - Phylogenetic tree reconstruction
  - PICRUSt2 pathway prediction

### 3. **Data Backfill** (`backfill.py`)
- `run_data_backfill(workflow)`: Enriches metadata from external APIs
  - Arkin LLM-driven environmental agents
  - Nuclear Fuel Cycle (NFC) facility matching via GIS
  - Environmental data collection (SoilGrids, Meteostat, etc.)

### 4. **Analysis** (`analysis.py`)
- `run_analysis_suite(workflow)`: Executes comprehensive ecological analyses
  - Community State Typing (CST)
  - Alpha diversity metrics
  - Beta diversity and ordination
  - Taxa-metadata statistical associations
  - Constrained ordination (RDA/CCA)
  - Co-occurrence network analysis
  - Machine learning biomarker discovery (CatBoost)
  - Comparative batch correction strategies

### 5. **Synthesis** (`synthesis.py`)
- `run_results_synthesis(workflow)`: Aggregates and reports findings
  - Cross-validates ML and statistical results
  - Generates master biomarker summary
  - Creates performance comparison plots
  - Exports executive reports

## Usage

### Complete Workflow

```python
from pathlib import Path
from workflow_16s.downstream.analysis import DownstreamWorkflow

workflow = DownstreamWorkflow(
    data_dir=Path("data/processed"),
    output_dir=Path("results/downstream"),
    config=config_dict,
    n_cpus=16
)

workflow.execute()  # Runs all 5 steps
```

### Individual Steps

```python
from workflow_16s.downstream.steps import (
    run_fast_load,
    run_preprocessing_pipeline,
    run_analysis_suite
)

# Run only specific steps
run_fast_load(workflow)
run_preprocessing_pipeline(workflow)
run_analysis_suite(workflow)
```

### Legacy Compatibility

The `run_downstream()` function provides backward compatibility:

```python
from workflow_16s.downstream.analysis import run_downstream

analyzer = run_downstream(config, project_dir, existing_subsets)
```

## Workflow State

The `DownstreamWorkflow` class maintains state across steps:

### Required Attributes
- `data_dir`: Path to input h5ad files
- `output_dir`: Path for results and plots
- `config`: Configuration dictionary
- `logger`: Logging instance
- `n_cpus`: Number of parallel processes

### Data Attributes
- `adata`: Main AnnData object (populated by ingestion)
- `faprotax_db`: FAPROTAX functional database
- `nfc_facilities_df`: Nuclear facilities GIS data
- `picrust2_conda_env`: PICRUSt2 environment name

### Analysis Metadata
- `priority_categorical`: Key categorical variables for analysis
- `priority_numeric`: Key numeric variables for analysis
- `cst_col`: Community state type column name (from CST analysis)


### Output Directories
- `plot_dir_alpha`, `plot_dir_beta`, `plot_dir_stats`, etc.
- `catboost_output_dir`: CatBoost feature selection results
- `picrust2_output_dir`: PICRUSt2 functional predictions

#### ML Output Directory Structure (v2.1+)

**To prevent overwriting results from repeated or context-specific ML runs (e.g., pre-QC, post-QC), all CatBoost/ML outputs are now saved under a unique run context directory.**

**Structure:**

```
catboost_output_dir/
  <run_context>/           # e.g., pre_qc, post_qc, or custom string
    <strategy>/          # baseline, agnostic, group_validated
      Genus_<target>/  # e.g., Genus_facility_match
        results_summary.json
        shap_report_details.csv
        ...
```

**Example:**

```
04_analysis/catboost_feature_selection/
  pre_qc/
    baseline/Genus_facility_match/results_summary.json
    agnostic/Genus_facility_match/results_summary.json
    ...
  post_qc/
    baseline/Genus_facility_match/results_summary.json
    ...
```

**How run context is set:**
- The workflow will use `workflow.run_context` if present, otherwise falls back to `workflow.qc_state` or 'default_run'.
- You can set this attribute on the workflow object to control output separation.

**Why:**
- This ensures that results from different workflow states (e.g., before/after QC, different parameterizations) are never overwritten and are easy to compare.

## Configuration Schema

The `config` dictionary supports these keys:

```python
config = {
    'downstream': {
        'data_dir': 'path/to/h5ad',
        'output_dir': 'path/to/results',
        'n_cpus': 16,
        'min_depth': 1000,
        'min_prevalence': 0.05
    },
    'arkin': {
        'enabled': False
    },
    'nfc': {
        'enabled': True
    },
    'environmental_data': {
        'enabled': False
    }
}
```

## Extending the Module

To add a new analysis step:

1. Create a new function in the appropriate module (or create a new module)
2. Follow the pattern: `def run_my_analysis(workflow):`
3. Access data via `workflow.adata`
4. Save results to `workflow.output_dir`
5. Update `workflow` attributes as needed
6. Export the function in `__init__.py`
7. Call it in `DownstreamWorkflow.execute()` in [analysis.py](../analysis.py)

## Testing

Each step can be tested independently by mocking the workflow object:

```python
from unittest.mock import Mock
from workflow_16s.downstream.steps import run_analysis_suite

workflow = Mock()
workflow.adata = test_adata
workflow.output_dir = Path("test_output")
workflow.config = test_config

run_analysis_suite(workflow)
```

## Migration from Legacy Code

The legacy `execute_legacy()` method is preserved for backward compatibility. To migrate:

1. Update workflow instantiation to include `config` parameter
2. Replace `workflow.execute_legacy()` with `workflow.execute()`
3. Ensure config dict has required keys for enabled features

## Performance Considerations

- **Parallel Processing**: Most CPU-intensive steps support multiprocessing via `workflow.n_cpus`
- **Memory**: Large datasets (>100k features) may require 32GB+ RAM
- **Caching**: Intermediate results are saved to disk to enable resumption
- **Conda Environments**: PICRUSt2 requires a separate conda environment

## Dependencies

- Core: `anndata`, `scanpy`, `pandas`, `numpy`
- Diversity: `scikit-bio`, `scipy`
- ML: `scikit-learn`, `catboost`, `shap`
- Visualization: `plotly`, `matplotlib`, `seaborn`
- External: `picrust2` (conda env), FAPROTAX database
