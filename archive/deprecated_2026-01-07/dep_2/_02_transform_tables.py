import logging
from typing import Dict
import anndata as ad
import scanpy as sc
import numpy as np
from workflow_16s.logger import get_logger

logger = get_logger()

class DataProcessor:
    """Performs in-place data transformations on an AnnData object using Scanpy."""
    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata

    def run(self) -> ad.AnnData:
        """Applies a sequence of transformations, storing results in layers."""
        logger.info("STEP 2: Applying data transformations...")

        if 'counts' not in self.adata.layers:
            self.adata.layers['counts'] = self.adata.X.copy()
        
        self._filter()
        self._normalize()
        self._clr_transform()
        self._presence_absence()
        
        return self.adata

    def _filter(self):
        """Filters low-prevalence features and low-depth samples."""
        conf = self.config.get('features', {}).get('filter', {})
        if not conf.get('enabled', True): return

        min_prevalence = conf.get('min_prevalence_samples', 3)
        min_features = conf.get('min_features_per_sample', 100)
        
        original_shape = self.adata.shape
        
        sc.pp.filter_genes(self.adata, min_cells=min_prevalence)
        sc.pp.filter_cells(self.adata, min_genes=min_features)
        
        self.adata.layers['counts'] = self.adata.X.copy()
        
        logger.info(f"Filtered data from {original_shape} to {self.adata.shape} (samples, features).")

    def _normalize(self):
        """Performs total-sum scaling (TSS) normalization using Scanpy."""
        conf = self.config.get('features', {}).get('normalize', {})
        if not conf.get('enabled', True): return
        
        target_sum = conf.get('target_sum', 1e6)
        
        temp_adata = ad.AnnData(X=self.adata.layers['counts'].copy())
        sc.pp.normalize_total(temp_adata, target_sum=target_sum)
        self.adata.layers['normalized'] = temp_adata.X.copy()

    def _clr_transform(self):
        """Performs Centered Log-Ratio (CLR) transformation."""
        conf = self.config.get('features', {}).get('clr_transform', {})
        if not conf.get('enabled', True): return

        counts = self.adata.layers['counts']
        counts_dense = counts.toarray() if hasattr(counts, 'toarray') else counts.copy()
        counts_dense[counts_dense == 0] = 1
        
        gmean = np.exp(np.mean(np.log(counts_dense), axis=1))
        clr_data = np.log(counts_dense / gmean[:, np.newaxis])
        self.adata.layers['clr'] = clr_data

    def _presence_absence(self):
        """Creates a binary presence/absence layer."""
        conf = self.config.get('features', {}).get('presence_absence', {})
        if not conf.get('enabled', True): return
        
        counts = self.adata.layers['counts']
        pa_data = (counts > 0).astype(int)
        self.adata.layers['presence_absence'] = pa_data