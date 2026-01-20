#!/usr/bin/env python
"""
ML-only validation run with comprehensive overfitting prevention.
"""

import logging
from pathlib import Path
import scanpy as sc
from workflow_16s.downstream.machine_learning import run_machine_learning_analysis, run_catboost_selection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_validation')

# Load the processed data
logger.info("Loading processed data...")
data_file = Path("/usr2/people/macgregor/amplicon/project_01/03_processed_data/adata_processed.h5ad")
if not data_file.exists():
    logger.error(f"Data file not found: {data_file}")
    exit(1)

adata = sc.read_h5ad(data_file)
logger.info(f"Loaded data: {adata.n_obs} samples, {adata.n_vars} features")

# Setup output directories
output_dir = Path("/usr2/people/macgregor/amplicon/project_01/04_analysis")
plot_dir_ml = output_dir / "plots" / "machine_learning"
plot_dir_ml.mkdir(parents=True, exist_ok=True)

catboost_dir = output_dir / "catboost_feature_selection"
catboost_dir.mkdir(parents=True, exist_ok=True)

# Priority targets (facility-related for validation)
priority_vars = [
    "facility_match", 
    "facility_distance_km", 
    "nuclear_contamination", 
    "facility_type", 
    "facility_id", 
    "reactor_type"
]

logger.info("\n" + "="*70)
logger.info("RUNNING ML VALIDATION WITH OVERFITTING PREVENTION")
logger.info("="*70 + "\n")

# Run RandomForest with full validation
logger.info("Starting RandomForest analysis with overfitting validation...")
run_machine_learning_analysis(
    adata=adata,
    plot_dir_ml=plot_dir_ml,
    level='Genus',
    priority_targets=priority_vars,
    validate_overfitting=True,
    quick_validation=False  # Full validation
)

# Run CatBoost with overfitting detection
logger.info("\nStarting CatBoost feature selection with overfitting detection...")
run_catboost_selection(
    adata=adata,
    catboost_output_dir=catboost_dir,
    level='Genus',
    priority_targets=priority_vars,
    use_group_kfold=True,
    batch_col='batch_original'
)

logger.info("\n" + "="*70)
logger.info("ML VALIDATION COMPLETE")
logger.info("="*70)
logger.info(f"\nResults saved to: {output_dir}")
logger.info(f"  - RandomForest plots: {plot_dir_ml}")
logger.info(f"  - Overfitting validation: {plot_dir_ml / 'overfitting_validation'}")
logger.info(f"  - CatBoost results: {catboost_dir}")
