# src/workflow_16s/downstream/machine_learning/workflows/soil_prediction.py

import pandas as pd

from pathlib import Path
from typing import Any, Union
from sklearn.cluster import KMeans

from workflow_16s.utils.logger import get_logger
from .feature_selection import run_catboost_selection


def run_soil_prediction_suite(
    adata: Any, 
    output_dir: Union[str, Path], 
    level: str = 'Genus',
    min_samples: int = 100,
    spatial_cv: bool = True,
    n_spatial_clusters: int = 10
) -> None:
    """
    Trains regression models to predict SoilGrids variables with Spatial Cross-Validation.
    
    Spatial CV prevents the model from overfitting to local geographical signatures by 
    ensuring that samples from the same region are never split between training and testing.
    
    Parameters
    ----------
    adata : Any
        Anndata object containing microbial counts and metadata.
    output_dir : Union[str, Path]
        Directory to save results and artifacts.
    level : str
        Taxonomic rank to aggregate features at.
    min_samples : int
        Minimum samples required to attempt SoilGrids prediction.
    spatial_cv : bool
        Whether to use Spatial Cross-Validation.
    n_spatial_clusters : int
        Number of spatial clusters to form for CV grouping. 
        
    Returns
    -------
    None
    """
    logger = get_logger("workflow_16s")
    logger.info(" 🌍 Starting SoilGrids Prediction Suite with Spatial CV...")
    
    out_path = Path(output_dir)
    
    # 1. Coordinate Validation for Spatial CV
    lat_col, lon_col = 'latitude', 'longitude'
    has_coords = lat_col in adata.obs.columns and lon_col in adata.obs.columns
    
    cv_groups_name = None
    if spatial_cv and has_coords:
        logger.info(f"📍 Generating {n_spatial_clusters} spatial clusters for CV grouping...")
        coords = adata.obs[[lat_col, lon_col]].dropna()
        if len(coords) > min_samples:
            # Cluster coordinates to create geographic blocks
            kmeans = KMeans(n_clusters=n_spatial_clusters, random_state=42, n_init=10)
            adata.obs['spatial_cluster'] = -1  # Default for missing coords
            adata.obs.loc[coords.index, 'spatial_cluster'] = kmeans.fit_predict(coords).astype(str)
            cv_groups_name = 'spatial_cluster'
            logger.info("✓ Spatial clusters successfully assigned to metadata.")
        else:
            logger.warning(" ⚠️ Insufficient coordinate data for Spatial CV. Falling back to Project-based CV.")

    # 2. Identify and Clean SoilGrids Targets
    soil_candidates = [col for col in adata.obs.columns if 'SoilGrids' in col]
    numeric_soil_targets = []
    
    for col in soil_candidates:
        try:
            series_numeric = pd.to_numeric(adata.obs[col], errors='coerce')
            if series_numeric.notna().sum() >= min_samples:
                adata.obs[col] = series_numeric 
                numeric_soil_targets.append(col)
        except Exception: continue

    if not numeric_soil_targets:
        logger.warning(f"❌ No valid numeric SoilGrids targets found.")
        return

    # 3. Execution via Workflow Engine
    soil_output_dir = out_path / "soil_predictions"
    soil_output_dir.mkdir(exist_ok=True, parents=True)
    
    # We use 'group_validated' or 'lopocv' depending on whether we use spatial or project groups
    strategy = 'group_validated' if cv_groups_name else 'lopocv'
    batch_col = cv_groups_name if cv_groups_name else 'study_accession'

    run_catboost_selection(
        adata=adata,
        catboost_output_dir=soil_output_dir,
        level=level,
        priority_targets=numeric_soil_targets,
        strict_targets=True,
        strategies=[strategy],
        test_size=0.2,
        num_features=30,
        n_cpus=16,
        method='shap',
        batch_col=batch_col  
    )
    
    logger.info(f" ✅ SoilGrids Prediction Suite Complete. Results in {soil_output_dir}")