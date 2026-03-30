"""
Module 4: Integrated Visualization

Creates publication-ready interactive visualizations combining:
- Phylogenetic relationships (taxonomy)
- Functional traits (metal-resistance genes)
- Spatial/environmental data (geolocation, metal proxies, soil chemistry)
- Ecotype assignments (from Module 2)

Outputs interactive Plotly figures + static PNG exports.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    output_dir: Path
    dpi: int = 300
    width: int = 1200
    height: int = 800
    color_scheme: str = 'viridis'  # plotly color scale
    show_annotations: bool = True
    export_png: bool = True
    export_html: bool = True


class PhyloFunctionVisualizer:
    """
    Creates phylogenetic trees colored by functional traits.
    
    Requires:
    - Phylogenetic tree (newick or Nexus format)
    - OTU abundance matrix
    - Trait presence/absence per OTU
    
    Outputs:
    - Interactive phylogenetic tree with trait coloring
    - Trait prevalence at each tree node
    - Gene content distribution across clades
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize phylo-function visualizer"""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_phylo_trait_tree(
        self,
        adata: Any,  # AnnData with taxonomy in .obs
        trait_matrix: pd.DataFrame,  # OTUs × traits
        trait_name: str,
        tree_data: Optional[Dict[str, Any]] = None,
        sample_filter: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Create interactive phylogenetic tree with trait coloring.
        
        Args:
            adata: AnnData object with OTU taxonomy
            trait_matrix: DataFrame with OTU presence/absence
            trait_name: Which trait to visualize
            tree_data: Optional phylogenetic tree structure
            sample_filter: Filter samples by metadata (e.g., {'environment': 'soil'})
        
        Returns:
            Plotly figure (interactive)
        """
        
        # Aggregate trait presence by taxonomy level
        if 'Phylum' in adata.var.columns:
            phylum_data = self._aggregate_by_taxon(
                adata,
                trait_matrix,
                trait_name,
                'Phylum'
            )
            
            # Create sunburst-style visualization
            fig = px.sunburst(
                phylum_data,
                names='taxon',
                parents='parent',
                values='trait_count',
                color='trait_prevalence',
                color_continuous_scale=self.config.color_scheme,
                title=f"Phylogenetic Distribution of {trait_name}",
                hover_data={'trait_count': ':.0f', 'trait_prevalence': ':.2%'}
            )
            
            fig.update_layout(
                width=self.config.width,
                height=self.config.height,
                font=dict(size=10),
            )
            
            return fig
        else:
            logger.warning("No taxonomy data found; cannot create phylo tree")
            return None
    
    def _aggregate_by_taxon(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        trait_name: str,
        taxon_level: str
    ) -> pd.DataFrame:
        """Aggregate trait presence by taxonomic level"""
        
        taxa = adata.var[taxon_level].unique()
        aggregated = []
        
        for taxon in taxa:
            otus_in_taxon = adata.var[adata.var[taxon_level] == taxon].index
            trait_presence = trait_matrix.loc[
                trait_matrix.index.isin(otus_in_taxon), trait_name
            ]
            
            aggregated.append({
                'taxon': taxon,
                'parent': '',  # Would be higher tax level
                'trait_count': trait_presence.sum(),
                'trait_prevalence': trait_presence.mean() if len(trait_presence) > 0 else 0
            })
        
        return pd.DataFrame(aggregated)


class SpatialTraitVisualizer:
    """
    Creates geographic visualizations of trait distribution.
    
    Maps samples to geographic locations and colors by:
    - Trait presence/absence
    - Functional diversity
    - Metal proxy scores
    - Ecotype assignment
    
    Requires: Latitude, Longitude in sample metadata
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize spatial visualizer"""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_spatial_trait_distribution(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        trait_name: str,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        size_col: Optional[str] = None
    ) -> go.Figure:
        """
        Create geographic map showing trait distribution.
        
        Args:
            adata: AnnData with sample coordinates
            trait_matrix: Samples × Traits matrix
            trait_name: Trait to visualize
            lat_col: Latitude column name
            lon_col: Longitude column name
            size_col: Optional column for marker size
        
        Returns:
            Plotly scatter_mapbox figure
        """
        
        # Extract coordinates and trait data
        if lat_col not in adata.obs.columns or lon_col not in adata.obs.columns:
            logger.warning(f"Columns {lat_col}, {lon_col} not found")
            return None
        
        map_data = pd.DataFrame({
            'latitude': adata.obs[lat_col],
            'longitude': adata.obs[lon_col],
            'sample_id': adata.obs_names,
            'trait_present': [
                trait_matrix.loc[sample_id, trait_name] if sample_id in trait_matrix.index else 0
                for sample_id in adata.obs_names
            ]
        })
        
        # Add size if specified
        if size_col and size_col in adata.obs.columns:
            map_data['size'] = adata.obs[size_col]
        else:
            map_data['size'] = 8
        
        fig = go.Figure(data=go.Scattermapbox(
            lat=map_data['latitude'],
            lon=map_data['longitude'],
            mode='markers',
            marker=dict(
                size=map_data['size'],
                color=map_data['trait_present'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title=f"{trait_name}<br>Presence",
                    thickness=15,
                    len=0.7
                ),
                line=dict(width=0.5, color='white')
            ),
            text=map_data['sample_id'],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{trait_name}: %{{marker.color}}<br>' +
                         'Lat: %{lat:.3f}<br>' +
                         'Lon: %{lon:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Geographic Distribution of {trait_name}",
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=map_data['latitude'].median(),
                    lon=map_data['longitude'].median()
                ),
                zoom=3
            ),
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def plot_trait_vs_metal_proxy(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        trait_name: str,
        metal_proxy_col: str,
        color_by: Optional[str] = None
    ) -> go.Figure:
        """
        Scatter plot: trait presence vs. metal proxy score.
        
        Shows correlation between metal enrichment and trait presence.
        """
        
        if metal_proxy_col not in adata.obs.columns:
            logger.warning(f"Column {metal_proxy_col} not found")
            return None
        
        plot_data = pd.DataFrame({
            'sample_id': adata.obs_names,
            'trait_present': [
                trait_matrix.loc[sample_id, trait_name] if sample_id in trait_matrix.index else 0
                for sample_id in adata.obs_names
            ],
            'metal_proxy': adata.obs[metal_proxy_col],
        })
        
        if color_by and color_by in adata.obs.columns:
            plot_data['color'] = adata.obs[color_by]
            color_col = 'color'
        else:
            color_col = 'trait_present'
        
        fig = px.scatter(
            plot_data,
            x='metal_proxy',
            y='trait_present',
            color=color_col,
            hover_data=['sample_id'],
            title=f"{trait_name} vs. Metal Proxy",
            labels={
                'metal_proxy': f'{metal_proxy_col} Score',
                'trait_present': f'{trait_name} (0=absent, 1=present)'
            }
        )
        
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            hovermode='closest',
            showlegend=True
        )
        
        # Add trend line
        z = np.polyfit(plot_data['metal_proxy'], plot_data['trait_present'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data['metal_proxy'].min(), plot_data['metal_proxy'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        return fig


class EcotypeFunctionVisualizer:
    """
    Visualizes ecotype assignments colored by functional traits.
    
    Creates:
    - Heatmap: ecotypes × traits (gene content by ecotype)
    - UMAP/PCA: samples colored by ecotype + trait overlay
    - Bar charts: trait prevalence by ecotype
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize ecotype visualizer"""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_ecotype_trait_heatmap(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        ecotype_col: str = 'ecotype'
    ) -> go.Figure:
        """
        Create heatmap: ecotypes × traits.
        
        Shows which traits are enriched in which ecotypes.
        """
        
        if ecotype_col not in adata.obs.columns:
            logger.warning(f"Column {ecotype_col} not found")
            return None
        
        # Aggregate trait presence by ecotype
        ecotypes = adata.obs[ecotype_col].unique()
        heatmap_data = []
        
        for ecotype in sorted(ecotypes):
            samples_in_ecotype = adata.obs[adata.obs[ecotype_col] == ecotype].index
            trait_counts = {}
            
            for trait in trait_matrix.columns:
                trait_presence = trait_matrix.loc[
                    trait_matrix.index.isin(samples_in_ecotype), trait
                ]
                trait_counts[trait] = trait_presence.mean()  # Prevalence
            
            heatmap_data.append(trait_counts)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=[f"Ecotype {e}" for e in sorted(ecotypes)])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale='YlOrRd',
            colorbar=dict(title="Trait<br>Prevalence"),
            hovertemplate='%{y}<br>%{x}<br>Prevalence: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Functional Trait Distribution Across Ecotypes",
            xaxis_title="Functional Trait",
            yaxis_title="Ecotype",
            width=self.config.width,
            height=self.config.height,
            xaxis={'side': 'bottom'}
        )
        
        return fig
    
    def plot_umap_trait_overlay(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        trait_name: str,
        embedding_key: str = 'X_umap'
    ) -> go.Figure:
        """
        Create UMAP plot with samples colored by trait presence.
        
        Requires UMAP embedding in adata.obsm
        """
        
        if embedding_key not in adata.obsm:
            logger.warning(f"Embedding {embedding_key} not found")
            return None
        
        embedding = adata.obsm[embedding_key]
        
        plot_data = pd.DataFrame({
            'umap_1': embedding[:, 0],
            'umap_2': embedding[:, 1],
            'sample_id': adata.obs_names,
            'trait_present': [
                trait_matrix.loc[sample_id, trait_name] if sample_id in trait_matrix.index else 0
                for sample_id in adata.obs_names
            ]
        })
        
        fig = px.scatter(
            plot_data,
            x='umap_1',
            y='umap_2',
            color='trait_present',
            hover_data=['sample_id'],
            color_continuous_scale='RdYlGn',
            title=f"UMAP: {trait_name} Distribution",
            labels={'umap_1': 'UMAP 1', 'umap_2': 'UMAP 2'}
        )
        
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            hovermode='closest'
        )
        
        return fig
    
    def plot_trait_prevalence_by_ecotype(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        ecotype_col: str = 'ecotype',
        top_n_traits: int = 10
    ) -> go.Figure:
        """
        Bar chart: trait prevalence grouped by ecotype.
        
        Shows top N traits ranked by prevalence.
        """
        
        if ecotype_col not in adata.obs.columns:
            logger.warning(f"Column {ecotype_col} not found")
            return None
        
        # Calculate mean trait prevalence across all samples
        overall_prevalence = trait_matrix.mean()
        top_traits = overall_prevalence.nlargest(top_n_traits).index
        
        # Calculate per-ecotype prevalence
        ecotypes = sorted(adata.obs[ecotype_col].unique())
        plot_data = []
        
        for ecotype in ecotypes:
            samples_in_ecotype = adata.obs[adata.obs[ecotype_col] == ecotype].index
            
            for trait in top_traits:
                trait_presence = trait_matrix.loc[
                    trait_matrix.index.isin(samples_in_ecotype), trait
                ]
                prevalence = trait_presence.mean()
                
                plot_data.append({
                    'Trait': trait,
                    'Ecotype': f'Ecotype {ecotype}',
                    'Prevalence': prevalence
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        fig = px.bar(
            plot_df,
            x='Trait',
            y='Prevalence',
            color='Ecotype',
            barmode='group',
            title=f"Top {top_n_traits} Traits by Ecotype",
            labels={'Prevalence': 'Mean Trait Prevalence'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_tickangle=-45
        )
        
        return fig


class DashboardBuilder:
    """
    Assembles individual visualizations into integrated dashboard.
    
    Creates multi-panel dashboard showing:
    - Phylogenetic trait distribution
    - Geographic trait distribution
    - Metal proxy correlations
    - Ecotype functional profiles
    - Summary statistics
    """
    
    def __init__(self, config: VisualizationConfig):
        """Initialize dashboard builder"""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_integrated_dashboard(
        self,
        adata: Any,
        trait_matrix: pd.DataFrame,
        metal_proxy_col: Optional[str] = None,
        ecotype_col: str = 'ecotype',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> go.Figure:
        """
        Create integrated multi-panel dashboard.
        
        Returns:
            Plotly figure with subplots
        """
        
        # Get top 3 traits for dashboard
        trait_prevalence = trait_matrix.mean()
        top_traits = trait_prevalence.nlargest(3).index.tolist()
        
        if len(top_traits) == 0:
            logger.error("No traits found in matrix")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Trait Prevalence: {trait}" for trait in top_traits[:2]
            ] + ["Trait Correlations"],
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # Panel 1-2: Top trait prevalence
        for idx, trait in enumerate(top_traits[:2], 1):
            trait_values = [
                trait_matrix.loc[sample_id, trait] if sample_id in trait_matrix.index else 0
                for sample_id in adata.obs_names
            ]
            
            fig.add_trace(
                go.Histogram(x=trait_values, nbinsx=2, name=trait),
                row=1, col=idx
            )
        
        # Panel 3: Trait correlations
        if len(top_traits) >= 2:
            trait1 = [
                trait_matrix.loc[s, top_traits[0]] if s in trait_matrix.index else 0
                for s in adata.obs_names
            ]
            trait2 = [
                trait_matrix.loc[s, top_traits[1]] if s in trait_matrix.index else 0
                for s in adata.obs_names
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=trait1, y=trait2,
                    mode='markers',
                    name=f"{top_traits[0]} vs {top_traits[1]}"
                ),
                row=2, col=1
            )
        
        # Panel 4: Metal proxy if available
        if metal_proxy_col and metal_proxy_col in adata.obs.columns:
            fig.add_trace(
                go.Scatter(
                    x=adata.obs[metal_proxy_col],
                    y=[
                        trait_matrix.loc[s, top_traits[0]] if s in trait_matrix.index else 0
                        for s in adata.obs_names
                    ],
                    mode='markers',
                    name=f'{metal_proxy_col} vs {top_traits[0]}'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Integrated Functional Biogeography Dashboard",
            showlegend=False,
            height=self.config.height * 2,
            width=self.config.width * 2
        )
        
        return fig
    
    def save_dashboard(
        self,
        fig: go.Figure,
        filename: str = "dashboard"
    ) -> None:
        """Save dashboard to HTML and optionally PNG"""
        
        if fig is None:
            logger.warning("No figure to save")
            return
        
        # Save HTML
        if self.config.export_html:
            html_path = self.config.output_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            logger.info(f"Saved HTML: {html_path}")
        
        # Save PNG (requires kaleido)
        if self.config.export_png:
            try:
                png_path = self.config.output_dir / f"{filename}.png"
                fig.write_image(
                    str(png_path),
                    width=fig.layout.width or 1200,
                    height=fig.layout.height or 800,
                    scale=2
                )
                logger.info(f"Saved PNG: {png_path}")
            except Exception as e:
                logger.warning(f"Could not save PNG (install kaleido): {e}")
