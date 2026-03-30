# workflow_16s/downstream/visualization/qc.py
from pathlib import Path
from typing import Union

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

from workflow_16s.utils.logger import get_logger

def qc_metrics(adata: ad.AnnData, output_dir: Union[str, Path]) -> None:
    """Calculates and plots basic QC metrics."""
    if adata is None or adata.n_obs == 0: return
    get_logger("workflow_16s").info("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    plot_path = Path(output_dir) / "qc_metrics.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=adata.obs, x='total_counts', ax=axes[0], bins=30)
        sns.histplot(data=adata.obs, x='n_genes_by_counts', ax=axes[1], bins=30)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        get_logger("workflow_16s").info(f"Saved QC plot: {plot_path}")
    except Exception: pass