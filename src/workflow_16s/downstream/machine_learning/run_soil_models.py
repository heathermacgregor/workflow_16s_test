# src/workflow_16s/downstream/machine_learning/run_soil_models.py

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.machine_learning.main import run_catboost_selection
from workflow_16s.downstream.utils import AnalysisUtils


def run_soil_prediction_suite(
    adata: Any, 
    output_dir: Path, 
    level: str = 'Genus'
):
    """
    Trains regression models to predict SoilGrids variables from the microbiome.
    This validates that the microbiome data contains meaningful environmental signal.
    """
    logger = get_logger("workflow_16s")
    logger.info(" 🌍 Starting SoilGrids Prediction Suite...")
    
    # 1. Identify SoilGrids Targets
    #     We look for columns starting with 'SoilGrids_' that are numeric
    soil_targets = [col for col in adata.obs.columns if 'SoilGrids' in col]
    numeric_soil_targets = []
    
    for col in soil_targets:
        # Check if column is actually numeric
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
            # Check sparsity (must have at least 100 non-null values to train)
            if adata.obs[col].notna().sum() > 100:
                numeric_soil_targets.append(col)
                
    if not numeric_soil_targets:
        logger.warning(" ❌ No valid numeric SoilGrids targets found (requires >100 samples).")
        return

    logger.info(f" ✅ Found {len(numeric_soil_targets)} valid SoilGrids targets: {numeric_soil_targets[:5]}...")
    
    # 2. Define Output Directory
    soil_output_dir = output_dir / "soil_predictions"
    soil_output_dir.mkdir(exist_ok=True, parents=True)
    
    # 3. Run CatBoost Regression for each target
    # We use 'agnostic' strategy only (Microbiome -> Soil Property)
    # We disable 'baseline' because we don't want to predict Soil using Metadata (trivial)
    
    strategies = ['agnostic', 'lopocv'] 
    
    # Re-use the main feature selection function
    # It automatically detects regression tasks if the target is continuous
    run_catboost_selection(
        adata=adata,
        catboost_output_dir=soil_output_dir,
        level=level,
        priority_targets=numeric_soil_targets,
        strict_targets=True, # Only run these exact columns
        strategies=strategies,
        test_size=0.2,
        num_features=30, # Top 30 drivers of soil chemistry
        n_cpus=16
    )
    
    logger.info(f"✅ SoilGrids Prediction Suite Complete. Results in {soil_output_dir}")