# ===================================== IMPORTS ====================================== #

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

import seaborn as sns
sns.set_style('whitegrid')

import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import colorcet as cc

# ================================== LOCAL IMPORTS =================================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

largecolorset = list(
    cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark
)

# Define the plot template
pio.templates["heather"] = go.layout.Template(
    layout={
        'title': {
            'font': {
                'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif', 
                'size': 30, 
                'color': '#000'
            }
        }, 
        'font': {
            'family': 'Helvetica Neue, Helvetica, Sans-serif', 
            'size': 16, 
            'color': '#000'
        }, 
        'paper_bgcolor': 'rgba(0, 0, 0, 0)', 
        'plot_bgcolor': '#fff', 
        'colorway': largecolorset, 
        'xaxis': {'showgrid': False}, 
        'yaxis': {'showgrid': False}
    }
)

# ==================================== FUNCTIONS ===================================== #


class FastQCPlots:
    def __init__(
        self, 
        results: Dict, 
        top_n_sequences: int = 10, 
        show_individual_samples: bool = False
    ):
        self.results = results
        self.top_n_sequences = top_n_sequences
        self.show_individual_samples = show_individual_samples
        self.figs = {}
        self._validate_results()

    def _validate_results(self):
        """Ensure required data structure and columns exist"""
        required_sections = {
            'quality_scores': ['sample', 'direction', 'base', 'mean_quality'],
            'adapter_content': ['sample', 'direction', 'position', 'adapter_percent'],
        }
        
        for section, cols in required_sections.items():
            if section in self.results:
                missing = [col for col in cols if col not in self.results[section].columns]
                if missing:
                    logger.warning(f"Missing columns {missing} in {section} data")
                    del self.results[section]

    def _safe_get_data(self, section: str) -> Optional[pd.DataFrame]:
        """Safely retrieve and preprocess data with type validation"""
        if section not in self.results:
            logger.warning(f"Missing data for section: {section}")
            return None
        
        df = self.results[section].copy()
        
        numeric_cols = ['mean_quality', 'adapter_percent', 'count', 'percentage', 
                        'percent', 'gc_percent', 'n_percent', 'N', 'length', 'quality']
        for col in numeric_cols:
            if col in df.columns:
                if not col == 'length' and not len(str(df[col].iloc[0]).split('-')) == 2:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(how='all')
    
    def _plot_individual_samples(self, ax, subset, x_col, y_col):
        """Helper method to plot individual sample lines"""
        if self.show_individual_samples:
            for sample in subset['sample'].unique():
                sample_data = subset[subset['sample'] == sample]
                ax.plot(sample_data[x_col], sample_data[y_col],
                        color='grey', alpha=0.2, linewidth=0.5,
                        label='_nolegend_')
                
    def plot_sequence_content(self):
        """Plot sequence content distribution with average and std deviation"""
        df = self._safe_get_data('sequence_content')
        if df is None or df.empty:
            return
        
        for base in ['G', 'A', 'T', 'C']:
            df[base] = pd.to_numeric(df[base], errors='coerce')
        df = df.dropna(subset=['G', 'A', 'T', 'C'])

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            for base in ['G', 'A', 'T', 'C']:
                self._plot_individual_samples(ax, subset, 'position', base)

            # Plot mean and std
            grouped = subset.groupby(
                'position', sort=False
            )[['G', 'A', 'T', 'C']].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()
            colors = {'G': 'green', 'A': 'blue', 'T': 'red', 'C': 'orange'}

            for base in ['G', 'A', 'T', 'C']:
                mean_col = (base, 'mean')
                std_col = (base, 'std')
                ax.plot(grouped['position'], grouped[mean_col], 
                        color=colors[base], linewidth=1.5, label=f'{base} Mean')
                if num_samples > 1:
                    ax.fill_between(grouped['position'],
                                    grouped[mean_col] - grouped[std_col],
                                    grouped[mean_col] + grouped[std_col],
                                    color=colors[base], alpha=0.3)

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - Sequence Content")
            ax.set_xlabel('Position')
            ax.set_ylabel('Percentage')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["sequence_content"] = fig

    def plot_adapter_content(self):
        """Plot adapter content with average and std deviation"""
        df = self._safe_get_data('adapter_content')
        if df is None or df.empty:
            return

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            self._plot_individual_samples(ax, subset, 'position', 'adapter_percent')

            # Plot mean and std
            grouped = subset.groupby(
                'position', sort=False
            )['adapter_percent'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()

            ax.plot(grouped['position'], grouped['mean'], 
                    color='blue', linewidth=1.5, label='Mean')
            if num_samples > 1:
                ax.fill_between(grouped['position'],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - Adapter Content")
            ax.set_xlabel('Position')
            ax.set_ylabel('Adapter Percentage')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
            ax.tick_params(axis='x', labelrotation=90)

        plt.tight_layout()
        self.figs["adapter_content"] = fig

    def plot_per_base_gc_content(self):
        """Plot GC content distribution with average and std deviation"""
        df = self._safe_get_data('per_base_gc_content')
        if df is None or df.empty:
            return

        df['total'] = df.groupby(['sample', 'direction'])['count'].transform('sum')
        df['percentage'] = (df['count'] / df['total']) * 100

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            self._plot_individual_samples(ax, subset, 'gc_percent', 'percentage')

            # Plot mean and std
            grouped = subset.groupby('gc_percent', sort=False)['percentage'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()

            ax.plot(grouped['gc_percent'], grouped['mean'], 
                    color='blue', linewidth=1.5, label='Mean')
            if num_samples > 1:
                ax.fill_between(grouped['gc_percent'],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - GC Content")
            ax.set_xlabel('GC Percentage')
            ax.set_ylabel('Read Percentage')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["per_base_gc_content"] = fig

    def plot_per_base_n_content(self):
        """Plot average N content percentage and standard deviation across positions"""
        df = self._safe_get_data('per_base_n_content')
        if df is None or df.empty or 'N' not in df.columns:
            return

        df['N'] = pd.to_numeric(df['N'], errors='coerce')
        df = df.dropna(subset=['N'])

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            self._plot_individual_samples(ax, subset, 'position', 'N')

            # Group by position to compute mean and standard deviation
            grouped = subset.groupby('position', sort=False)['N'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()

            # Plot mean line
            ax.plot(grouped['position'], grouped['mean'], color='blue', linewidth=1.5, label='Mean')

            # Plot standard deviation
            if num_samples > 1:
                ax.fill_between(
                    grouped['position'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='blue', alpha=0.3, label='Standard Deviation'
                )

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - N Content")
            ax.set_xlabel('Position')
            ax.set_ylabel('N Percentage')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        self.figs["per_base_n_content"] = fig

    def plot_duplication_levels(self):
        """Plot sequence duplication levels"""
        df = self._safe_get_data('duplication_levels')
        if df is None or df.empty:
            return

        df['duplication_level'] = pd.Categorical(
            df['duplication_level'], 
            categories=df['duplication_level'].unique(),  # Use original order instead of sorted
            ordered=True
        )

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            if self.show_individual_samples:
                for sample in subset['sample'].unique():
                    sample_data = subset[subset['sample'] == sample]
                    ax.plot(sample_data['duplication_level'].astype(str), 
                           sample_data['percent'],
                           color='grey', alpha=0.2, linewidth=0.5,
                           marker='o', markersize=2, label='_nolegend_')

            # Plot mean and std
            grouped = subset.groupby('duplication_level', sort=False)['percent'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()
            
            x = np.arange(len(grouped))
            ax.plot(x, grouped['mean'], color='blue', linewidth=1.5, 
                    marker='o', label='Mean')
            if num_samples > 1:
                ax.fill_between(x,
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['duplication_level'])
            ax.set_title(f"Direction {direction} - Duplication Levels")
            ax.set_xlabel('Duplication Level')
            ax.set_ylabel('Percentage')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["duplication_levels"] = fig

    def plot_length_distribution(self):
        """Plot read length distribution with average and std deviation"""
        df = self._safe_get_data('length_distribution')
        if df is None or df.empty:
            return

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            self._plot_individual_samples(ax, subset, 'length', 'count')

            # Plot mean and std
            grouped = subset.groupby('length', sort=False)['count'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()

            ax.plot(grouped['length'], grouped['mean'], 
                    color='blue', linewidth=1.5, label='Mean')
            if num_samples > 1:
                ax.fill_between(grouped['length'],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - Read Lengths")
            ax.set_xlabel('Length (bp)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["length_distribution"] = fig

    def plot_quality_scores(self):
        """Plot per-base quality scores with average and std deviation"""
        df = self._safe_get_data('quality_scores')
        if df is None or df.empty:
            return
        
        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()
        
        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            self._plot_individual_samples(ax, subset, 'base', 'mean_quality')

            # Plot mean and std
            grouped = subset.groupby('base', sort=False)['mean_quality'].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()

            ax.plot(grouped['base'], grouped['mean'], 
                    color='blue', linewidth=1.5, label='Mean')
            if num_samples > 1:
                ax.fill_between(grouped['base'],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - Quality Scores")
            ax.set_xlabel('Base')
            ax.set_ylabel('Quality Score')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
            ax.tick_params(axis='x', labelrotation=90)

        plt.tight_layout()
        self.figs["quality_scores"] = fig

    def plot_per_seq_quality_scores(self):
        """Plot per-sequence quality scores with average and std deviation"""
        df = self._safe_get_data('per_seq_quality_scores')
        if df is None or df.empty:
            return

        # Convert score to numeric and clean data
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.dropna(subset=['score'])

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Sort by numeric score first
            subset = subset.sort_values('score')

            # Plot individual samples (sorted)
            if self.show_individual_samples:
                for sample in subset['sample'].unique():
                    sample_data = subset[subset['sample'] == sample].sort_values('score')
                    ax.plot(
                        sample_data['score'], 
                        sample_data['count'],
                        color='grey', alpha=0.2, linewidth=0.5,
                        label='_nolegend_'
                    )

            # Group and aggregate sorted data
            grouped = subset.groupby('score', sort=True)['count'].agg(['mean', 'std']).reset_index()

            # Main plot with numerical x-axis
            ax.plot(grouped['score'], grouped['mean'], 
                    color='blue', linewidth=1.5, label='Mean')

            if len(subset['sample'].unique()) > 1:
                ax.fill_between(grouped['score'],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                color='blue', alpha=0.3, label='Std Dev')

            # Configure axis
            ax.set_title(f"Direction {direction} - Per-Sequence Quality Scores")
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Ensure proper tick spacing
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["per_seq_quality_scores"] = fig

    def plot_sequence_content(self):
        """Plot sequence content distribution with average and std deviation"""
        df = self._safe_get_data('sequence_content')
        if df is None or df.empty:
            return
        
        for base in ['G', 'A', 'T', 'C']:
            df[base] = pd.to_numeric(df[base], errors='coerce')
        df['sort_x'] = [int(str(x).split('-')[0]) for x in df['position']]
        df = df.dropna(subset=['G', 'A', 'T', 'C'])

        directions = sorted(df['direction'].unique())
        fig, axes = plt.subplots(1, len(directions), figsize=(15, 5), squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = df[df['direction'] == direction]
            if subset.empty:
                continue

            # Plot individual samples
            for base in ['G', 'A', 'T', 'C']:
                self._plot_individual_samples(ax, subset, 'position', base)

            # Plot mean and std
            grouped = subset.groupby('sort_x')[['G', 'A', 'T', 'C']].agg(['mean', 'std']).reset_index()
            num_samples = subset['sample'].nunique()
            colors = {'G': 'green', 'A': 'blue', 'T': 'red', 'C': 'orange'}

            for base in ['G', 'A', 'T', 'C']:
                mean_col = (base, 'mean')
                std_col = (base, 'std')
                ax.plot(grouped['sort_x'], grouped[mean_col], 
                        color=colors[base], linewidth=1.5, label=f'{base} Mean')
                if num_samples > 1:
                    ax.fill_between(grouped['sort_x'],
                                    grouped[mean_col] - grouped[std_col],
                                    grouped[mean_col] + grouped[std_col],
                                    color=colors[base], alpha=0.3)

            # Add legend entries
            if self.show_individual_samples:
                ax.plot([], [], color='grey', alpha=0.3, linewidth=1, label='Individual Samples')
                
            ax.set_title(f"Direction {direction} - Sequence Content")
            ax.set_xlabel('Position')
            ax.set_ylabel('Percentage')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
        self.figs["sequence_content"] = fig

    def plot_overrepresented_seqs(self):
        """Plot top overrepresented sequences aggregated across samples"""
        df = self._safe_get_data('overrepresented_seqs')
        if df is None or df.empty:
            return

        top_n = self.top_n_sequences

        # Aggregate sequences across samples
        aggregated_df = (
            df.groupby(['direction', 'seq'])
            .agg(
                total_percentage=('percentage', 'sum'),
                samples=('sample', lambda x: f"{len(set(x))} samples")  # Show count instead of names
            )
            .reset_index()
            .groupby('direction', group_keys=False)
            .apply(lambda g: g.nlargest(top_n, 'total_percentage'))
            .reset_index(drop=True)
        )

        directions = sorted(aggregated_df['direction'].unique())
        fig, axes = plt.subplots(len(directions), 1, 
                               figsize=(18, 6 * len(directions)),  # Increased width
                               squeeze=False)
        axes = axes.flatten()

        for ax, direction in zip(axes[:len(directions)], directions):
            subset = aggregated_df[aggregated_df['direction'] == direction]
            if subset.empty:
                continue

            # Create shorter labels
            subset['label'] = (
                subset['seq'].str[:15] + '...' +  # Truncate more aggressively
                ' (' + subset['samples'] + ')'
            )

            # Plot with smaller font
            ax.barh(subset['label'], subset['total_percentage'], alpha=0.7)
            ax.set_title(f"Direction {direction} - Top Sequences", pad=20)
            ax.set_xlabel('Percentage', labelpad=10)
            ax.grid(True, alpha=0.3)

            # Adjust font sizes
            ax.tick_params(axis='y', labelsize=9)  # Smaller y-axis labels
            ax.tick_params(axis='x', labelsize=10)

            # Increase margins
            ax.margins(y=0.1)
            plt.sca(ax)
            plt.subplots_adjust(left=0.5, right=0.95)  # More space for labels

        plt.tight_layout(pad=4.0)
        self.figs["overrepresented_seqs"] = fig

    def export_figures(self, export_dir: str, dpi: int = 300):
        """Save figures with error handling and directory creation"""
        os.makedirs(export_dir, exist_ok=True)
        sections = [
            "basic_stats", "quality_scores", "adapter_content", 
            "length_distribution", "sequence_content", 
            "overrepresented_seqs", "per_seq_quality_scores", 
            "per_base_gc_content", "per_seq_gc_content", 
            "per_base_n_content", "duplication_levels", 
            "kmer_content"
        ]
        
        for section in sections:
            try:
                if section not in list(self.results.keys()) or self.results[section].empty:
                    continue
                    
                method_name = f"plot_{section}"
                if not hasattr(self, method_name):
                    continue
                
                #logger.info(f"Generating {section.replace('_', ' ').title()} plot")
                plot_method = getattr(self, method_name)
                plot_method()
                
                if section not in list(self.figs.keys()):
                    continue
                    
                output_path = Path(export_dir, f"{section}.png")
                self.figs[section].savefig(output_path, dpi=dpi, bbox_inches='tight')
                plt.close(self.figs[section])
                
            except Exception as e:
                logger.error(f"Failed {section.replace('_', ' ').title()}: {str(e)}", exc_info=True)
              
