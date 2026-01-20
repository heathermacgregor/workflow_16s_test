import logging
from pathlib import Path
from typing import Dict
import anndata as ad
import pandas as pd
import skbio.diversity
from workflow_16s.logger import get_logger
from workflow_16s.downstream.diversity.alpha_diversity import analyze_alpha_correlations
from workflow_16s.visualization.alpha_diversity import plot_alpha_correlations

logger = get_logger()

class AlphaDiversity:
    """Calculates alpha diversity and correlations, storing results in the AnnData object."""
    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.alpha_config = self.config.get('alpha_diversity', {})
    
    def run(self, output_dir: Path) -> ad.AnnData:
        """Executes the alpha diversity analysis."""
        if not self.alpha_config.get('enabled', True): 
            return self.adata
        
        logger.info("STEP 3: Calculating alpha diversity...")
        metrics = self.alpha_config.get('metrics', ['shannon', 'sobs'])
        
        counts_matrix = self.adata.layers.get('counts')
        if counts_matrix is None:
            logger.error("Raw 'counts' layer not found. Skipping alpha diversity.")
            return self.adata

        counts_array = counts_matrix.toarray() if hasattr(counts_matrix, 'toarray') else counts_matrix
        
        sample_ids = self.adata.obs_names
        calculated_metrics = []
        for metric in metrics:
            try:
                div_series = skbio.diversity.alpha_diversity(metric, counts_array, ids=sample_ids)
                self.adata.obs[metric] = div_series
                calculated_metrics.append(metric)
            except Exception as e:
                logger.warning(f"Could not calculate alpha diversity metric '{metric}': {e}")
        
        corr_config = self.alpha_config.get("correlation_analysis", {})
        if corr_config.get('enabled', True) and calculated_metrics:
            logger.info("STEP 3: Running alpha diversity correlation analysis...")
            alpha_df = self.adata.obs[calculated_metrics]
            corr_results = analyze_alpha_correlations(alpha_df, self.adata.obs)
            self.adata.uns['alpha_diversity_correlations'] = corr_results
            
            plot_output_dir = output_dir / 'alpha_diversity'
            plot_output_dir.mkdir(exist_ok=True, parents=True)
            
            plot_alpha_correlations(corr_results, output_dir=plot_output_dir)

        return self.adata