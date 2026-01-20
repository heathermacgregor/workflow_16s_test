import logging
from pathlib import Path
from typing import Dict
import anndata as ad
import pandas as pd
from workflow_16s.downstream.statistics import advanced_analyses
from workflow_16s.logger import get_logger

logger = get_logger()

class AdvancedAnalyses:
    """Performs advanced analyses like core microbiome and network analysis."""
    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.advanced_config = self.config.get('advanced_analyses', {})

    def run(self, output_dir: Path) -> ad.AnnData:
        if not self.advanced_config.get('enabled', True): return self.adata
        logger.info("STEP 6: Running advanced analyses...")
        self.adata.uns['advanced_analyses'] = {}
        if self.advanced_config.get('core_microbiome', {}).get('enabled', True): self._run_core_microbiome()
        if self.advanced_config.get('network_analysis', {}).get('enabled', False): self._run_network_analysis()
        return self.adata

    def _run_core_microbiome(self):
        """Calculates the core microbiome for specified groups."""
        conf = self.advanced_config.get('core_microbiome', {})
        primary_cols = conf.get('group_columns', [])
        auto_cols = self.adata.uns.get('analysis_columns', {}).get("group_comparison", []) if conf.get('analyze_all_valid_columns', True) else []
        group_columns = sorted(list(set(primary_cols + auto_cols)))

        for group_col in group_columns:
            if group_col not in self.adata.obs.columns:
                logger.warning(f"Core microbiome: Skipping group column '{group_col}' as it's not in metadata.")
                continue
            logger.info(f"Calculating core microbiome for groups in '{group_col}'...")
            
            if 'presence_absence' not in self.adata.layers:
                logger.error("Core microbiome requires 'presence_absence' layer. Skipping.")
                continue

            pa_matrix = self.adata.layers['presence_absence']
            pa_array = pa_matrix.toarray() if hasattr(pa_matrix, 'toarray') else pa_matrix
            presence_df = pd.DataFrame(pa_array, index=self.adata.obs_names, columns=self.adata.var_names)
            presence_df[group_col] = self.adata.obs[group_col]
            
            prevalence_threshold = conf.get('prevalence_threshold', 0.80)
            core_members_by_group = {}

            for group_name, group_data in presence_df.groupby(group_col):
                prevalence = group_data.drop(columns=[group_col]).mean()
                core_features = prevalence[prevalence >= prevalence_threshold].index.tolist()
                # CORRECTED: Cast the group_name to a string to ensure it's serializable
                core_members_by_group[str(group_name)] = core_features

            core_results = self.adata.uns.setdefault('advanced_analyses', {}).setdefault('core_microbiome', {})
            core_results[group_col] = core_members_by_group

    def _run_network_analysis(self):
        """Performs microbial network analysis based on correlation."""
        conf = self.advanced_config.get('network_analysis', {})
        for layer in conf.get('layers', []):
            if layer not in self.adata.layers: continue
            
            data_matrix = self.adata.layers[layer]
            data_array = data_matrix.toarray() if hasattr(data_matrix, 'toarray') else data_matrix
            df = pd.DataFrame(data_array, index=self.adata.obs_names, columns=self.adata.var_names)

            corr_matrix, edges_df = advanced_analyses.microbial_network_analysis(df, method=conf.get('method', 'spearman'), threshold=conf.get('threshold', 0.3))
            network_stats = advanced_analyses.calculate_network_statistics(edges_df)
            
            net_results = self.adata.uns.setdefault('advanced_analyses', {}).setdefault('network_analysis', {})
            net_results[layer] = {'edges': edges_df, 'statistics': network_stats}