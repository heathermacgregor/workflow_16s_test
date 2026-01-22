"""
Functional Prediction Module.

Tools for inferring metabolic functions from taxonomy (FAPROTAX) 
or gene content (PICRUSt2).
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import anndata as ad
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from workflow_16s.utils.faprotax import FaprotaxDB
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

def predict_functions_faprotax(adata: ad.AnnData, faprotax_db: FaprotaxDB) -> ad.AnnData:
    """
    Predicts functional groups for taxa in an AnnData object using FAPROTAX.
    """
    logger.info("--- Predicting functions using FAPROTAX ---")
    if 'Taxon' not in adata.var.columns:
        logger.error("FAPROTAX requires 'Taxon' column in adata.var. Run parse_taxonomy first.")
        return adata

    # 1. Get the FULL list of all taxonomy strings
    full_taxa_list = adata.var['Taxon'].astype(str).tolist()
    logger.info(f"Predicting functions for {len(full_taxa_list)} total features...")

    # 2. Call the batch function
    try:
        all_function_lists = faprotax_db.predict_functions_batch(full_taxa_list)
    except Exception as e:
        logger.error(f"FAPROTAX batch prediction failed: {e}")
        return adata

    logger.info("Batch prediction complete. Binarizing results...")

    # 3. Binarize results
    mlb = MultiLabelBinarizer()
    function_matrix = mlb.fit_transform(all_function_lists)
    function_names = mlb.classes_
    n_found = len(function_names)

    if n_found == 0:
        logger.warning("FAPROTAX prediction returned no functional groups.")
        return adata

    logger.info(f"Found {n_found} unique FAPROTAX functions. Adding to adata.var...")

    # 4. Create boolean DataFrame
    func_df = pd.DataFrame(
        data=function_matrix, 
        index=adata.var_names, 
        columns=[f"faprotax:{name}" for name in function_names]
    ).astype(bool)

    # 5. Update adata.var
    old_cols = [c for c in adata.var.columns if c.startswith("faprotax:")]
    if old_cols:
        adata.var.drop(columns=old_cols, inplace=True)

    adata.var = pd.concat([adata.var, func_df], axis=1)
    logger.info(f"Added {n_found} FAPROTAX function columns.")

    return adata
