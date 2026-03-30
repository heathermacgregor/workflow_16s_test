# workflow_16s/downstream/steps/preprocessing.py
""""""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from workflow_16s.utils.io.anndata import (
    export_fasta
)

from workflow_16s.downstream.utils.adata_biology import (
    clean_metadata,
    filter_low_depth_and_prevalence,
    filter_samples_and_features,
    parse_taxonomy, 
)
from workflow_16s.downstream.visualization.qc import qc_metrics
from workflow_16s.downstream.utils.helpers import AnalysisUtils
def run_preprocessing_pipeline(workflow):
    """Main entry point for the preprocessing pipeline."""
    workflow.logger.info("=== Starting Preprocessing Pipeline ===")
    
    if workflow.adata is None:
        workflow.logger.error("No AnnData object found in workflow. Skipping preprocessing.")
        return

    workflow.adata = clean_metadata(workflow.adata, workflow.config)
    
    # Parse taxonomy if not already present, otherwise skip to avoid 
    # overwriting existing taxonomy columns
    genus_exists = 'Genus' in workflow.adata.var.columns
    genus_has_data = genus_exists and not workflow.adata.var['Genus'].isnull().all()
    
    if genus_has_data:
        workflow.logger.info("Taxonomy columns found (Genus with data). Skipping redundant parse_taxonomy step.")
    else:
        workflow.logger.info("Genus column missing or empty - parsing taxonomy...")
        workflow.adata = parse_taxonomy(workflow.adata)
        
    # Filtering steps: remove samples and features based on config thresholds
    workflow.adata = filter_samples_and_features(workflow.adata, workflow.config)
    workflow.adata = filter_low_depth_and_prevalence(workflow.adata, workflow.config)
    
    if workflow.adata is None or workflow.adata.n_obs == 0:
        workflow.logger.error("All samples removed during preprocessing.")
        workflow.adata = None
        return

    # Ensure raw_counts layer exists for downstream steps 
    if 'raw_counts' not in workflow.adata.layers:
        workflow.logger.info("Initializing 'raw_counts' layer from X (assuming raw inputs)...")
        if issparse(workflow.adata.X):
            workflow.adata.layers['raw_counts'] = csr_matrix(workflow.adata.X)
        else:
            workflow.adata.layers['raw_counts'] = csr_matrix(np.array(workflow.adata.X))
    
    # Apply rCLR transformation (optional, configurable)
    apply_rclr = getattr(workflow.config.downstream, 'apply_rclr_transformation', True)
    if apply_rclr:
        workflow.logger.info("Applying Robust CLR (rCLR) Transformation...")
        transformed = AnalysisUtils.rclr_transform(workflow.adata)
        if transformed is not None:
            # rclr_transform returns sparse matrix to avoid memory explosion on large matrices
            workflow.adata.X = transformed
            workflow.logger.info(f"rCLR transformation complete. Matrix shape: {workflow.adata.X.shape}")
        else:
            workflow.logger.error("rCLR transformation failed.")
    else:
        workflow.logger.info("rCLR transformation disabled (apply_rclr_transformation=False in config)")


    # Generate QC metrics
    qc_metrics(workflow.adata, workflow.output_dir)

    # Export FASTA for downstream steps
    export_fasta(workflow.adata, workflow.config, workflow.output_dir)

    workflow.logger.info("=== Preprocessing Pipeline Complete ===")
