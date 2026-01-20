# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Dict, List

# Third-Party Imports
import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Local Imports 
from workflow_16s.constants import DEFAULT_HEIGHT, DEFAULT_WIDTH
from workflow_16s.visualization import DataPrep, apply_common_layout, save_fig 

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ============================= PLOTTING FUNCTIONS =================================== #

def create_alpha_diversity_boxplot(
    alpha_df: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    metric: str,
    output_dir: Path,
    test_type: str = "nonparametric"
) -> go.Figure:
    """Creates a boxplot for an alpha diversity metric with statistical annotations.
    
    Args:
        alpha_df (pd.DataFrame): DataFrame with alpha diversity metrics (samples x metrics).
        metadata (pd.DataFrame): Metadata DataFrame (samples x metadata variables).     
        group_column (str): Column in metadata to group by.
        metric (str): The alpha diversity metric to plot.
        output_dir (Path): Directory to save the plot.
        test_type (str): Type of statistical test ("nonparametric" or "parametric

    Returns:
        go.Figure: The generated boxplot figure.    

    Raises:
        ValueError: If the specified metric is not in alpha_df or group_column not in metadata
    """
    try:
        # Use the robust DataPrep class for reliable data merging
        prep = DataPrep(alpha_df[[metric]], metadata)
        merged_df = prep.prepare(color_col=group_column, symbol_col=group_column)

        if merged_df.empty:
            logger.warning(f"No common data for metric '{metric}' after merging with metadata.")
            return go.Figure()

        fig = px.box(
            merged_df,
            x=group_column,
            y=metric,
            color=group_column,
            points='all',
            title=f"{metric.replace('_', ' ').title()} by {group_column}"
        )

        fig = apply_common_layout(fig, group_column, metric.replace('_', ' ').title(), f"{metric.replace('_', ' ').title()} Diversity")

        # Add statistical annotation
        groups = merged_df[group_column].dropna().unique()
        if len(groups) > 1:
            if len(groups) == 2:
                group1 = merged_df[merged_df[group_column] == groups[0]][metric]
                group2 = merged_df[merged_df[group_column] == groups[1]][metric]
                test_func = stats.mannwhitneyu if test_type == "nonparametric" else stats.ttest_ind
                stat, p_val = test_func(group1, group2)
                test_name = "Mann-Whitney U" if test_type == "nonparametric" else "T-test"
            else: # More than 2 groups
                group_data = [merged_df[merged_df[group_column] == g][metric].dropna() for g in groups]
                test_func = stats.kruskal if test_type == "nonparametric" else stats.f_oneway
                stat, p_val = test_func(*group_data)
                test_name = "Kruskal-Wallis" if test_type == "nonparametric" else "ANOVA"

            fig.add_annotation( # type: ignore
                x=0.5, y=1.05, xref="paper", yref="paper",
                text=f"{test_name} p={p_val:.3g}", showarrow=False,
                font=dict(size=16)
            )

        save_fig(fig, output_dir / f"alpha_boxplot_{metric}", formats=['html', 'json', 'png'])
        return fig

    except Exception as e:
        logger.error(f"Failed to create boxplot for {metric}: {e}")
        return go.Figure()


def create_alpha_diversity_stats_plot(
    stats_df: pd.DataFrame,
    output_dir: Path,
    effect_size_threshold: float = 0.5
) -> go.Figure:
    """Creates an interactive visualization for statistical results (p-values and effect sizes).
    
    Args:
        stats_df (pd.DataFrame): DataFrame with statistical results (metrics x stats).
        output_dir (Path): Directory to save the plot.
        effect_size_threshold (float): Threshold to highlight significant effect sizes. 
        
    Returns:
        go.Figure: The generated statistics summary figure."""
    df = stats_df.copy()
    df['-log10(p_value)'] = -np.log10(df['p_value'])
    df['significant'] = df['p_value'] < 0.05

    fig = px.bar(
        df,
        x='metric',
        y='-log10(p_value)',
        color='significant',
        color_discrete_map={True: '#EF553B', False: '#636EFA'},
        hover_data=['p_value', 'effect_size', 'statistic'],
        title="Alpha Diversity Statistical Summary"
    )
    fig = apply_common_layout(fig, "Diversity Metric", "-log10(p-value)", "Alpha Diversity Statistical Summary")
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", annotation_text="p=0.05",
                  annotation_position="bottom right", line_color="red")
    
    save_fig(fig, output_dir / "alpha_stats_summary", formats=['html', 'json', 'png'])
    return fig


def plot_alpha_correlations(
    corr_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    top_n: int = 10
) -> Dict[str, go.Figure]:
    """Visualizes top correlations for each alpha diversity metric.
    
    Args:
        corr_results (Dict[str, pd.DataFrame]): Dictionary with correlation results per metric.
        output_dir (Path): Directory to save the plots.
        top_n (int): Number of top correlations to display per metric.
        
    Returns:
        Dict[str, go.Figure]: Dictionary of generated correlation figures per metric.
    """
    figs = {}
    for metric, df in corr_results.items():
        if df.empty:
            continue

        df_top = df.head(top_n).copy()
        df_top['association_strength'] = df_top['spearman_rho'].fillna(df_top['eta_squared']).abs()
        df_top['significant'] = (df_top['spearman_p'].fillna(df_top['kruskal_p'])) < 0.05
        
        fig = px.bar(
            df_top,
            x='metadata_column',
            y='association_strength',
            color='type',
            hover_data=['spearman_rho', 'spearman_p', 'eta_squared', 'kruskal_p'],
            title=f"Top {top_n} Associations with {metric.replace('_', ' ').title()}"
        )
        fig = apply_common_layout(fig, "Metadata Variable", "Association Strength (|ρ| or η²)", f"Top {top_n} Associations with {metric.replace('_', ' ').title()}")
        figs[metric] = fig
        save_fig(fig, output_dir / f"alpha_correlation_{metric}", formats=['html', 'json', 'png'])

    return figs