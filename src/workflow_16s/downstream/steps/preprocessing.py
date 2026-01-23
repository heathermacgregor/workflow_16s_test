# ==================================================================================== #
#                           downstream/steps/preprocessing.py
# ==================================================================================== #

import math
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import cpu_count

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, issparse
import joblib
from joblib import Parallel, delayed

from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.progress import get_progress_bar

logger = get_logger("workflow_16s")

from workflow_16s.downstream.utils import (
    qc_metrics, export_fasta, clean_metadata, parse_taxonomy,
    filter_samples_and_features, filter_low_depth_and_prevalence
)


# --- Main Entry Point ---

def run_preprocessing_pipeline(workflow):
    """Main entry point for the preprocessing pipeline."""
    logger.info("=== Starting Preprocessing Pipeline ===")
    
    if workflow.adata is None:
        logger.error("No AnnData object found in workflow. Skipping preprocessing.")
        return

    workflow.adata = clean_metadata(workflow.adata, workflow.config)
    workflow.adata = parse_taxonomy(workflow.adata)
    workflow.adata = filter_samples_and_features(workflow.adata, workflow.config)
    workflow.adata = filter_low_depth_and_prevalence(workflow.adata, workflow.config)
    
    if workflow.adata.n_obs == 0:
        logger.error("All samples removed during preprocessing.")
        workflow.adata = None
        return

    qc_metrics(workflow.adata, workflow.output_dir)
    export_fasta(workflow.adata, workflow.config, workflow.output_dir)
    
    # Ensure raw_counts layer exists for downstream steps 
    if 'raw_counts' not in workflow.adata.layers:
        logger.info("Initializing 'raw_counts' layer from X (assuming raw inputs)...")
        workflow.adata.layers['raw_counts'] = workflow.adata.X.copy()
    
    logger.info("=== Preprocessing Pipeline Complete ===")