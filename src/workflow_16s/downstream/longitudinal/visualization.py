import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import anndata as ad
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger('workflow_16s')

def plot_temporal_trajectories(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    feature: str,
    color_by: Optional[str] = None,
    output_path: Optional[Path] = None
) -> go.Figure:
    """Plot spaghetti plot of trajectories for a specific feature."""
    
    # Extract data
    df = pd.DataFrame({
        'time': adata.obs[time_col],
        'abundance': adata[:, feature].to_df()[feature],
        'subject': adata.obs[subject_col].astype(str),
        'group': adata.obs[color_by].astype(str) if color_by else 'All'
    })
    
    # Spaghetti lines (light)
    fig = px.line(
        df, x='time', y='abundance', color='group', line_group='subject',
        title=f"Trajectory: {feature}",
        template='plotly_white',
        labels={'abundance': 'Relative Abundance', 'time': time_col}
    )
    fig.update_traces(opacity=0.2, line=dict(width=1))
    
    # Mean trend lines (heavy)
    mean_df = df.groupby(['time', 'group'])['abundance'].mean().reset_index()
    
    # Add mean lines manually to ensure they are visible on top
    for group in mean_df['group'].unique():
        g_data = mean_df[mean_df['group'] == group]
        fig.add_trace(go.Scatter(
            x=g_data['time'], y=g_data['abundance'],
            mode='lines+markers',
            line=dict(width=4),
            name=f"{group} (Mean)"
        ))

    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Trajectory plot saved to {output_path}")
        
    return fig