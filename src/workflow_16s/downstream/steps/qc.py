"""Quality Control step for comprehensive data validation.

This module integrates the comprehensive QC system into the downstream workflow,
including metadata validation, sample identity checking, and contamination detection.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger('workflow_16s')


def run_comprehensive_qc(workflow):
    """
    Run comprehensive quality control before analysis.
    
    This function integrates the new QC modules:
    - MetadataValidator: Remove redundancy, validate ranges, harmonize units
    - ENVOOntology: Semantic categorization of samples
    - SampleIdentityValidator: Cross-validate claimed vs. observed
    - Enhanced contamination detection: Works without negative controls
    
    Parameters
    ----------
    workflow : DownstreamWorkflow
        The workflow instance with loaded data
    
    Returns
    -------
    None
        Updates workflow.adata in place, adds QC results to .obs
    """
    if not hasattr(workflow, '_qc') or workflow._qc is None:
        logger.debug("QC module not initialized or disabled")
        return
    
    if workflow.adata is None:
        logger.warning("No data loaded, skipping QC")
        return
    
    qc_config = workflow._qc_config
    output_dir = workflow._qc_output_dir
    
    logger.info("Running comprehensive quality control...")
    
    try:
        # 1. Metadata Validation
        metadata_config = qc_config.get('metadata_validation', {})
        if metadata_config.get('enabled', True):
            logger.info("  1/3 Validating metadata...")
            
            metadata_results = workflow._qc.run_metadata_qc(
                workflow.adata.obs,
                output_dir=output_dir / 'metadata'
            )
            
            if metadata_results and 'cleaned_metadata' in metadata_results:
                # Update with cleaned metadata
                cleaned_meta = metadata_results['cleaned_metadata']
                
                # Keep only samples that are in both
                common_samples = workflow.adata.obs.index.intersection(cleaned_meta.index)
                workflow.adata = workflow.adata[common_samples, :].copy()
                
                # Update metadata
                for col in cleaned_meta.columns:
                    if col not in workflow.adata.obs.columns or col.startswith('env_category_'):
                        workflow.adata.obs[col] = cleaned_meta.loc[common_samples, col]
                    elif metadata_config.get('remove_redundant', True):
                        # Replace with cleaned version
                        workflow.adata.obs[col] = cleaned_meta.loc[common_samples, col]
                
                n_removed_cols = metadata_results.get('n_removed_columns', 0)
                logger.info(f"    ✓ Metadata validated: removed {n_removed_cols} redundant columns")
                
                # Check for semantic categories
                if 'env_category_type' in workflow.adata.obs.columns:
                    cat_counts = workflow.adata.obs['env_category_type'].value_counts()
                    logger.info(f"    ✓ ENVO categorization: {len(cat_counts)} environment types")
        
        # 2. Sample Identity Validation  
        sample_config = qc_config.get('sample_validation', {})
        if sample_config.get('enabled', True):
            logger.info("  2/3 Validating sample identities...")
            
            sample_results = workflow._qc.run_sample_validation(
                workflow.adata,
                output_dir=output_dir / 'samples'
            )
            
            if sample_results and 'validation_df' in sample_results:
                validation_df = sample_results['validation_df']
                
                # Add QC flags to adata.obs
                for col in ['qc_env_match', 'qc_human_contamination', 'qc_overall_flag']:
                    if col in validation_df.columns:
                        workflow.adata.obs[col] = validation_df[col]
                
                # Summary statistics
                if 'qc_overall_flag' in workflow.adata.obs.columns:
                    flag_counts = workflow.adata.obs['qc_overall_flag'].value_counts()
                    n_pass = flag_counts.get('PASS', 0)
                    n_warn = flag_counts.get('WARNING', 0)
                    n_fail = flag_counts.get('FAIL', 0)
                    logger.info(
                        f"    ✓ Sample validation: "
                        f"{n_pass} PASS, {n_warn} WARNING, {n_fail} FAIL"
                    )
                    
                    # Handle failed samples based on config
                    fail_action = qc_config.get('flagging', {}).get('fail_action', 'flag')
                    if fail_action == 'remove' and n_fail > 0:
                        workflow.adata = workflow.adata[
                            workflow.adata.obs['qc_overall_flag'] != 'FAIL', :
                        ].copy()
                        logger.info(f"    ✓ Removed {n_fail} failed samples")
        
        # 3. Contamination Detection
        contam_config = qc_config.get('contamination_detection', {})
        if contam_config.get('enabled', True):
            logger.info("  3/3 Detecting contamination...")
            
            contam_results = workflow._qc.run_contamination_detection(
                workflow.adata,
                method=contam_config.get('method', 'combined'),
                threshold=contam_config.get('threshold', 0.5),
                output_dir=output_dir / 'contamination'
            )
            
            if contam_results and 'contamination_scores' in contam_results:
                scores_df = contam_results['contamination_scores']
                
                # Store scores in adata.var
                if 'contamination_score' in scores_df.columns:
                    workflow.adata.var['contamination_score'] = scores_df['contamination_score']
                    workflow.adata.var['is_contaminant'] = scores_df.get(
                        'is_contaminant', 
                        scores_df['contamination_score'] > contam_config.get('threshold', 0.5)
                    )
                
                # Summary
                n_contam = workflow.adata.var['is_contaminant'].sum()
                logger.info(f"    ✓ Contamination detection: {n_contam} contaminants identified")
                
                # Remove contaminants if configured
                if contam_config.get('remove_contaminants', False) and n_contam > 0:
                    workflow.adata = workflow.adata[
                        :, ~workflow.adata.var['is_contaminant']
                    ].copy()
                    logger.info(f"    ✓ Removed {n_contam} contaminated features")
        
        # Generate comprehensive HTML report
        output_config = qc_config.get('output', {})
        if output_config.get('generate_html', True):
            try:
                qc_results = workflow._qc.run_all(
                    workflow.adata,
                    output_dir=output_dir,
                    fastq_files=None,  # FASTQ QC disabled for now
                    primers=None,
                    remove_contaminants=contam_config.get('remove_contaminants', False)
                )
                
                # Use cleaned data from comprehensive QC if available
                if qc_results and 'cleaned_data' in qc_results:
                    workflow.adata = qc_results['cleaned_data']
                    logger.info("  ✓ Applied all QC filters, data cleaned")
                
            except Exception as e:
                logger.warning(f"HTML report generation failed: {e}")
        
        logger.info("✓ Comprehensive QC complete")
        
    except Exception as e:
        logger.error(f"QC workflow failed: {e}")
        logger.warning("Continuing without QC")


def run_semantic_filtering(workflow, category: str, min_confidence: float = 0.5):
    """
    Filter samples by semantic category using ENVO ontology.
    
    Example usage:
        # Get all soil samples (regardless of exact string)
        run_semantic_filtering(workflow, 'soil')
        
        # Get all aquatic samples (marine + freshwater)
        run_semantic_filtering(workflow, 'aquatic')
    
    Parameters
    ----------
    workflow : DownstreamWorkflow
        The workflow instance with QC-annotated data
    category : str
        Category to filter by (e.g., 'soil', 'marine', 'freshwater', 'wastewater')
    min_confidence : float, optional
        Minimum ENVO categorization confidence (0-1)
    
    Returns
    -------
    None
        Updates workflow.adata in place
    """
    if workflow.adata is None:
        return
    
    # Check if ENVO categories are available
    if 'env_category_type' not in workflow.adata.obs.columns:
        logger.warning(
            "ENVO categories not found. Run metadata QC first to enable semantic filtering."
        )
        return
    
    # Filter by category
    if 'env_category_confidence' in workflow.adata.obs.columns:
        mask = (
            (workflow.adata.obs['env_category_type'] == category) &
            (workflow.adata.obs['env_category_confidence'] >= min_confidence)
        )
    else:
        mask = workflow.adata.obs['env_category_type'] == category
    
    n_before = workflow.adata.n_obs
    workflow.adata = workflow.adata[mask, :].copy()
    n_after = workflow.adata.n_obs
    
    logger.info(
        f"Filtered to {category} samples: {n_before} → {n_after} samples "
        f"({n_after/n_before:.1%} retained)"
    )
