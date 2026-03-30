"""
Orchestration entry point for Pagel's lambda phylogenetic signal analysis.

This module shows how to integrate Pagel's lambda analysis into the downstream pipeline.
It's separate from ML and can run independently on functional annotations.

Usage:
    from workflow_16s.downstream.steps.phylogenetic_signal_step import run_phylogenetic_signal_step
    
    result = run_phylogenetic_signal_step(
        config=config,
        adata=adata_processed,
        output_base=Path("results")
    )
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.pagels_lambda import (
    PagelsLambdaAnalyzer,
    run_pagels_lambda_pipeline
)

logger = get_logger("workflow_16s")


def extract_functional_data(
    adata,
    taxonomy_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract functional traits and OTU abundance from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Processed AnnData object with functional annotations in .var
    taxonomy_df : pd.DataFrame, optional
        Taxonomy information (for annotation)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (function_matrix: OTU × Function, otu_table: samples × OTU)
    """
    # Extract OTU abundance table (convert to dense if sparse)
    if hasattr(adata.X, 'toarray'):
        otu_table = pd.DataFrame(
            adata.X.toarray().T,
            index=adata.var_names,
            columns=adata.obs_names
        )
    else:
        otu_table = pd.DataFrame(
            adata.X.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
    
    # Extract functional traits from .var columns
    # Look for known functional trait columns (user can customize)
    functional_cols = [
        col for col in adata.var.columns
        if any(pattern in col.lower() for pattern in [
            'function', 'trait', 'gene', 'pathway', 'metabolic',
            'uranium', 'arsenic', 'metal', 'reduction', 'metabolism',
            'efflux', 'reduction', 'sulfur', 'nitrate', 'biofilm'
        ])
    ]
    
    if functional_cols:
        function_matrix = adata.var[functional_cols].copy()
        # Convert to binary if needed
        function_matrix = (function_matrix > 0).astype(int) if function_matrix.dtype != int \
            else function_matrix
    else:
        logger.warning(
            "No functional trait columns found in .var. "
            "Expected columns with patterns like: 'function', 'trait', 'uranium_reduction', etc."
        )
        function_matrix = pd.DataFrame(index=adata.var_names)
    
    logger.debug(f"Extracted functional matrix: {function_matrix.shape[0]} OTUs × {function_matrix.shape[1]} traits")
    logger.debug(f"OTU abundance table: {otu_table.shape[0]} OTUs × {otu_table.shape[1]} samples")
    
    return function_matrix, otu_table


def run_phylogenetic_signal_step(
    config: Dict,
    adata,
    output_base: Optional[Path] = None,
    taxonomy_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    End-to-end Pagel's lambda phylogenetic signal analysis step.
    
    Integrates into downstream workflow as independent optional step.
    **Not part of ML pipeline** - runs after functional annotation, before visualization.
    
    Parameters
    ----------
    config : Dict
        Pipeline configuration (must have 'downstream.phylogenetic_signal' key)
    adata : AnnData
        Processed AnnData object with optional functional annotations
    output_base : Path, optional
        Base output directory (default: config["paths"]["project"] / output_dir)
    taxonomy_df : pd.DataFrame, optional
        Taxonomy information for annotation
    
    Returns
    -------
    Dict
        Results dict with:
        - 'analyzer': PagelsLambdaAnalyzer with results
        - 'results': Dict of PagelsLambdaResult objects
        - 'output_dir': Path to results directory
        - 'status': 'success' or 'skipped'
    
    Example
    -------
    >>> result = run_phylogenetic_signal_step(config, adata, output_base=Path("results"))
    >>> if result['status'] == 'success':
    ...     print(f"Lambda values saved to {result['output_dir']}")
    """
    
    # Check if enabled in config
    ps_config = config.get('downstream', {}).get('phylogenetic_signal', {})
    
    if not ps_config.get('enabled', False):
        logger.info("Phylogenetic signal analysis disabled in config")
        return {
            'status': 'skipped',
            'reason': 'disabled in config'
        }
    
    # Determine output directory
    if output_base is None:
        project_dir = Path(config.get('paths', {}).get('project', './'))
        output_dir = project_dir / ps_config.get('output_dir', '03_phylogenetic_signal')
    else:
        output_dir = Path(output_base) / ps_config.get('output_dir', '03_phylogenetic_signal')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("PHYLOGENETIC SIGNAL (PAGEL'S LAMBDA) ANALYSIS")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}\n")
    
    try:
        # Extract functional data from AnnData
        function_matrix, otu_table = extract_functional_data(adata, taxonomy_df)
        
        if function_matrix.empty or function_matrix.shape[1] == 0:
            logger.warning("No functional traits to analyze. Skipping phylogenetic signal analysis.")
            return {
                'status': 'skipped',
                'reason': 'no functional traits found'
            }
        
        # Filter to traits of interest (if specified in config)
        traits_to_analyze = ps_config.get('traits_to_analyze', [])
        if traits_to_analyze:
            available_traits = [t for t in traits_to_analyze if t in function_matrix.columns]
            if available_traits:
                function_matrix = function_matrix[available_traits]
                logger.info(f"Analyzing {len(available_traits)} specified traits: {available_traits}")
            else:
                logger.warning(
                    f"Specified traits {traits_to_analyze} not found. "
                    f"Available: {list(function_matrix.columns)}"
                )
        
        # Initialize analyzer
        dpi = ps_config.get('plot_dpi', 300)
        analyzer = PagelsLambdaAnalyzer(output_dir, dpi=dpi)
        
        # Run analysis
        results = analyzer.analyze(function_matrix, otu_table, taxonomy_df)
        
        if not results:
            logger.warning("Phylogenetic signal analysis produced no results.")
            return {
                'status': 'failed',
                'reason': 'no results'
            }
        
        # Generate visualizations (if enabled)
        if ps_config.get('visualization_enabled', True):
            plot_formats = ps_config.get('plot_format', ['html', 'png'])
            
            logger.info("\nGenerating visualizations...")
            
            # Forest plot of lambda estimates
            fig_forest = analyzer.plot_lambda_distribution()
            if fig_forest:
                if 'html' in plot_formats:
                    fig_forest.write_html(output_dir / "pagels_lambda_forest.html")
                if 'png' in plot_formats:
                    try:
                        fig_forest.write_image(
                            output_dir / "pagels_lambda_forest.png",
                            width=1000,
                            height=max(400, 40*len(results)),
                            scale=2
                        )
                    except Exception as e:
                        logger.warning(f"PNG export failed (install kaleido): {e}")
                logger.info("✓ Forest plot saved")
            
            # Scatter plot (λ vs p-value)
            fig_scatter = analyzer.plot_lambda_scatter()
            if fig_scatter:
                if 'html' in plot_formats:
                    fig_scatter.write_html(output_dir / "pagels_lambda_scatter.html")
                if 'png' in plot_formats:
                    try:
                        fig_scatter.write_image(output_dir / "pagels_lambda_scatter.png", scale=2)
                    except Exception as e:
                        logger.warning(f"PNG export failed: {e}")
                logger.info("✓ Scatter plot saved")
            
            # Interpretation summary pie chart
            fig_summary = analyzer.plot_interpretation_summary()
            if fig_summary:
                if 'html' in plot_formats:
                    fig_summary.write_html(output_dir / "pagels_lambda_summary.html")
                if 'png' in plot_formats:
                    try:
                        fig_summary.write_image(output_dir / "pagels_lambda_summary.png", scale=2)
                    except Exception as e:
                        logger.warning(f"PNG export failed: {e}")
                logger.info("✓ Summary plot saved")
        
        # Save results table
        analyzer.save_results_table()
        
        # Generate markdown report (if enabled)
        if ps_config.get('generate_report', True):
            report = analyzer.generate_report()
            report_path = output_dir / "phylogenetic_signal_report.md"
            report_path.write_text(report)
            logger.info(f"✓ Report saved: {report_path}")
        
        logger.info("\n✓ Phylogenetic signal analysis complete!")
        logger.info(f"Results: {output_dir}\n")
        
        return {
            'status': 'success',
            'analyzer': analyzer,
            'results': results,
            'output_dir': output_dir,
            'num_traits': len(results),
            'num_conserved': sum(1 for r in results.values() if not np.isnan(r.lambda_estimate) and r.lambda_estimate > 0.6),
            'num_adaptive': sum(1 for r in results.values() if not np.isnan(r.lambda_estimate) and r.lambda_estimate < 0.3),
        }
    
    except Exception as e:
        logger.error(f"Phylogenetic signal analysis failed: {e}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e)
        }
