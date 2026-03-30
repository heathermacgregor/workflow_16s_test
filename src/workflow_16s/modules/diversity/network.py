# ==================================================================================== #
# diversity/network.py
# ==================================================================================== #

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
from pathlib import Path
from typing import List
from scipy.sparse import issparse
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from workflow_16s.downstream.utils import AnalysisUtils
from workflow_16s.visualization.utils import PlottingUtils
from workflow_16s.utils.logger import get_logger


def run_network_analysis(adata: ad.AnnData, analysis_levels: List[str], plot_dir_network: Path):
    """Performs co-occurrence network analysis using Spearman correlation."""
    logger = get_logger("workflow_16s")
    plot_utils = PlottingUtils(logger)

    logger.info("--- Starting Network Analysis ---")
    for level in analysis_levels:
        adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
        if adata_agg is None or adata_agg.n_obs < 10: continue
        sc.pp.normalize_total(adata_agg, target_sum=1e4); sc.pp.log1p(adata_agg)
        if adata_agg.n_vars > 100:
            data_temp = adata_agg.X.toarray() if issparse(adata_agg.X) else np.array(adata_agg.X)  # type: ignore
            top = np.argsort(data_temp.mean(0))[-100:]
            adata_agg = adata_agg[:, top].copy()
        data = adata_agg.X.toarray() if issparse(adata_agg.X) else adata_agg.X  # type: ignore
        result = spearmanr(data, axis=0)  # type: ignore
        corr = result.correlation  # type: ignore
        pvals = result.pvalue  # type: ignore
        np.fill_diagonal(corr, 0); G = nx.Graph(); threshold, p_adj_thresh = 0.5, 0.05
        p_flat = pvals[np.triu_indices(pvals.shape[0], k=1)]
        if len(p_flat) == 0: continue
        rej, padj_f, _, _ = multipletests(p_flat, method='fdr_bh')
        padj_m = np.ones_like(pvals); padj_m[np.triu_indices(padj_m.shape[0], k=1)] = padj_f
        padj_m = padj_m + padj_m.T - np.diag(padj_m.diagonal())
        for i, taxon in enumerate(adata_agg.var_names): G.add_node(taxon, abundance=data[:, i].mean())  # type: ignore
        for i in range(len(adata_agg.var_names)):
            for j in range(i + 1, len(adata_agg.var_names)):
                if padj_m[i, j] < p_adj_thresh and abs(corr[i, j]) > threshold:
                    G.add_edge(adata_agg.var_names[i], adata_agg.var_names[j], weight=abs(corr[i, j]), type='positive' if corr[i, j]>0 else 'negative')
        if G.number_of_edges() == 0: continue
        nx.write_gml(G, str(plot_dir_network / level / f"network_{level}.gml"))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        # Plotly visualization logic following spring layout positions...
        logger.info(f"Network for {level} complete: {G.number_of_nodes()} nodes.")
    logger.info("--- Network Analysis Complete ---")