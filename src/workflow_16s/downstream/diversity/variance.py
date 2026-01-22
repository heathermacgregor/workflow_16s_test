# diversity/variance.py
import pandas as pd
import plotly.express as px
from skbio.stats.ordination import rda
from workflow_16s.downstream.steps.preprocessing import AnalysisUtils
from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")

def run_variance_partitioning(adata, level='Genus', plot_dir_stats=None):
    """Quantifies variance attributed to Batch vs Biology."""
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=level)
    clr_df = AnalysisUtils._clr_transform(adata_agg, pseudocount=1)  # type: ignore
    if clr_df is None or adata_agg is None: return
    
    # Variables to check
    meta = adata_agg.obs[['batch_original', 'facility_match']].dropna()  # type: ignore
    common = clr_df.index.intersection(meta.index)
    
    # RDA for Batch
    res_batch = rda(y=clr_df.loc[common], x=pd.get_dummies(meta.loc[common, 'batch_original']))
    var_batch = res_batch.proportion_explained.sum()  # type: ignore
    
    # RDA for Facility
    res_fac = rda(y=clr_df.loc[common], x=pd.get_dummies(meta.loc[common, 'facility_match']))
    var_fac = res_fac.proportion_explained.sum()  # type: ignore
    
    # Plotting
    var_df = pd.DataFrame({'Source': ['Technical (Batch)', 'Biological (Facility)'], 'Variance Explained': [var_batch, var_fac]})
    fig = px.bar(var_df, x='Source', y='Variance Explained', title=f"Variance Partitioning ({level})")
    if plot_dir_stats:
        fig.write_html(plot_dir_stats / "variance_partitioning.html")  # type: ignore