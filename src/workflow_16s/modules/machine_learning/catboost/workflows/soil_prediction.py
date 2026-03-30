# workflow_16s/modules/machine_learning/catboost/workflows/soil_prediction.py

from pathlib import Path
from typing import Any, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

from workflow_16s.utils.logger import get_logger, with_logger
logger = get_logger("workflow_16s")
from .feature_selection import run_catboost_selection

@with_logger 
def run_soil_prediction_suite(
    adata: ad.AnnData, output_dir: Union[str, Path], level: str = 'Genus',
    min_samples: int = 100, spatial_cv: bool = True, n_spatial_clusters: int = 10
) -> None:
    """
    Trains regression models to predict SoilGrids variables with Spatial Cross-Validation.
    """
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
            
            # 💡 FIX: Convert degrees to radians for the Haversine formula
            coords_rad = np.radians(coords)
            
            # 💡 FIX: Replace KMeans (Euclidean) with AgglomerativeClustering (Haversine)
            # Note: 'ward' linkage only works with Euclidean, so we use 'average' linkage
            clusterer = AgglomerativeClustering(
                n_clusters=n_spatial_clusters, 
                metric='euclidean', 
                linkage='average'
            )
            
            adata.obs['spatial_cluster'] = -1  # Default for missing coords
            adata.obs.loc[coords.index, 'spatial_cluster'] = clusterer.fit_predict(coords_rad).astype(str)
            cv_groups_name = 'spatial_cluster'
            logger.info("✓ Spatial (Haversine) clusters successfully assigned to metadata.")
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
        adata=adata, catboost_output_dir=soil_output_dir, level=level,
        priority_targets=numeric_soil_targets, strict_targets=True,
        strategies=[strategy], test_size=0.2, num_features=30,
        n_cpus=16, method='shap', batch_col=batch_col  
    )
    
    logger.info(f" ✅ SoilGrids Prediction Suite Complete. Results in {soil_output_dir}")
    
