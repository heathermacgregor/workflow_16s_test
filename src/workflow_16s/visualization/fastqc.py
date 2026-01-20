# ===================================== IMPORTS ====================================== #
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ================================ MAIN PLOTTING CLASS ================================= #

class FastQCPlots:
    """Generates a suite of modern, interactive FastQC plots using Plotly.
    
    This class handles data preparation and uses helper methods to create
    consistent, reusable, and interactive visualizations for bioinformatics QC.

    Attributes:
        results (Dict[str, pd.DataFrame]): A dictionary where keys are FastQC
            section names and values are the corresponding DataFrames.
        show_individual_samples (bool): If True, plots individual sample data
            as faint, toggleable traces behind the aggregate plot.
    """
    def __init__(self, results: Dict[str, pd.DataFrame], show_individual_samples: bool = False):
        self.results = results
        self.show_individual_samples = show_individual_samples
        # Set a default Plotly template for consistent styling across all plots
        pio.templates.default = "plotly_white"

    def _safe_get_data(self, section: str) -> Optional[pd.DataFrame]:
        """Safely retrieves and pre-processes the data for a given section.
        
        This method checks for the existence of the data, creates a copy, and
        prepares columns that represent positions or ranges for correct numerical sorting.
        """
        if section not in self.results or self.results[section].empty:
            print(f"Warning: Missing or empty data for section: {section}")
            return None
        
        df = self.results[section].copy()
        
        # Centralize robust numeric sorting for any position-based plots
        for col in ['base', 'position', 'length']:
            if col in df.columns:
                # Extracts the starting number from ranges like '10-15' for proper sorting
                df[f'{col}_sort'] = df[col].astype(str).str.split('-').str[0].astype(int)
                df = df.sort_values(f'{col}_sort')
        return df

    def _create_summary_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                                  title: str, x_label: str, y_label: str) -> go.Figure:
        """Generic helper to create a summary line plot with mean and standard deviation.
        This is the workhorse for most of the standard FastQC line graphs.
        """
        fig = go.Figure()
        
        # Aggregate data to calculate mean and std dev for plotting
        sort_col = f'{x_col}_sort' if f'{x_col}_sort' in df else x_col
        agg_df = df.groupby(['direction', x_col]).agg(mean_val=(y_col, 'mean'),
                                                      std_val=(y_col, 'std'),
                                                      sort_key=(sort_col, 'first')).reset_index().sort_values('sort_key')
        
        colors = px.colors.qualitative.Plotly

        # Plot mean line and shaded standard deviation band for each direction (e.g., Fwd/Rev)
        for i, (direction, group) in enumerate(agg_df.groupby('direction')):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=group[x_col], y=group['mean_val'], name=f'{direction} Mean',
                                     mode='lines', line=dict(color=color, width=2.5)))
            fig.add_trace(go.Scatter(x=pd.concat([group[x_col], group[x_col][::-1]]),
                                     y=pd.concat([group['mean_val'] + group['std_val'], 
                                                  (group['mean_val'] - group['std_val'])[::-1]]),
                                     fill='toself', fillcolor=color, opacity=0.2,
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip", showlegend=False))
        
        # Optionally add traces for each individual sample
        if self.show_individual_samples:
            for sample, sample_df in df.groupby('sample'):
                for i, (direction, group) in enumerate(sample_df.groupby('direction')):
                    fig.add_trace(go.Scatter(x=group[x_col], y=group[y_col], 
                                             name=f'{sample} ({direction})',
                                             mode='lines', line=dict(color='grey', 
                                                                     width=0.5),
                                             opacity=0.5, visible='legendonly'))
        
        fig.update_layout(title_text=f"<b>{title}</b>", xaxis_title=x_label,
                          yaxis_title=y_label, legend_title_text="Trace",
                          title_x=0.5)
        return fig

    # --------------------------- Public Plotting Methods ---------------------------- #

    def plot_quality_scores(self) -> Optional[go.Figure]:
        """Generates the plot for Per-Base Sequence Quality scores."""
        df = self._safe_get_data('quality_scores')
        if df is None: return None
        
        return self._create_summary_line_plot(df=df, x_col='base', y_col='mean_quality',
                                              title="Per-Base Sequence Quality",
                                              x_label="Position in Read (bp)", 
                                              y_label="Phred Quality Score")

    def plot_adapter_content(self) -> Optional[go.Figure]:
        """Generates the plot for Adapter Content."""
        df = self._safe_get_data('adapter_content')
        if df is None: return None

        return self._create_summary_line_plot(df=df, x_col='position', 
                                              y_col='adapter_percent',
                                              title="Adapter Content", 
                                              x_label="Position in Read (bp)", 
                                              y_label="Adapter Content (%)")

    def plot_per_base_n_content(self) -> Optional[go.Figure]:
        """Generates the plot for Per-Base 'N' Content."""
        df = self._safe_get_data('per_base_n_content')
        if df is None: return None
        
        return self._create_summary_line_plot(df=df, x_col='position', 
                                              y_col='N',
                                              title="Per-Base 'N' Content",
                                              x_label="Position in Read (bp)",
                                              y_label="'N' Content (%)")

    def plot_sequence_length_distribution(self) -> Optional[go.Figure]:
        """Generates the plot for Sequence Length Distribution."""
        df = self._safe_get_data('length_distribution')
        if df is None: return None
        
        return self._create_summary_line_plot(df=df, x_col='length', y_col='count',
                                              title="Sequence Length Distribution",
                                              x_label="Sequence Length (bp)",
                                              y_label="Count")

    def plot_overrepresented_sequences(self, top_n: int = 15) -> Optional[go.Figure]:
        """Generates a bar chart for the most overrepresented sequences."""
        df = self._safe_get_data('overrepresented_seqs')
        if df is None: return None

        # Aggregate sequences across all samples, summing their percentages
        agg_df = df.groupby('seq')['percentage'].sum().nlargest(top_n).reset_index()
        
        # Create a user-friendly label with a truncated sequence
        agg_df['label'] = agg_df['seq'].str.slice(0, 30) + '...'
        
        fig = px.bar(agg_df, x='percentage', y='label', orientation='h',
                     title=f"<b>Top {top_n} Overrepresented Sequences (Aggregated)</b>",
                     labels={'label': 'Sequence', 
                             'percentage': 'Total Percentage Across All Samples'},
                     hover_data={'label': False, 'seq': True})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
        return fig

    # ------------------------------ Exporting Logic ------------------------------- #

    def export_figures(self, export_dir: str):
        """Generates and exports all available plots as interactive HTML files.
        This method dynamically finds and calls all public 'plot_*' methods.
        """
        output_path = Path(export_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        print(f"Exporting interactive plots to: {output_path.resolve()}")
        
        # Dynamically find all 'plot_*' methods in the class to run them
        plot_methods = [m for m in dir(self) if m.startswith('plot_')]
        
        for method_name in plot_methods:
            section_name = method_name.replace('plot_', '')
            print(f"  -> Generating plot for: {section_name}...")
            try:
                # Call the plot method to get the figure object
                plot_method = getattr(self, method_name)
                fig = plot_method()
                
                if fig:
                    file_path = output_path / f"{section_name}.html"
                    fig.write_html(str(file_path))
            except Exception as e:
                print(f"Failed to generate plot for {section_name}: {e}")
        print("Done.")

# ================================== EXAMPLE USAGE =================================== #

def generate_dummy_data() -> Dict[str, pd.DataFrame]:
    """Creates a dictionary of dummy DataFrames to simulate real FastQC results."""
    data = {}
    samples = ['SampleA', 'SampleB', 'SampleC']
    directions = ['Fwd', 'Rev']
    
    # --- Quality Scores ---
    qs_list = []
    for sample in samples:
        for direction in directions:
            for i in range(1, 151):
                quality = 35 - (i / 15) + np.random.uniform(-1, 1)
                qs_list.append({'sample': sample, 'direction': direction, 
                                'base': str(i), 'mean_quality': quality})
    data['quality_scores'] = pd.DataFrame(qs_list)
    
    # --- Adapter Content ---
    adapter_list = []
    for sample in samples:
        for direction in directions:
            for i in range(1, 151):
                adapter = max(0, (i - 130) / 2 + np.random.uniform(-0.5, 0.5))
                adapter_list.append({'sample': sample, 'direction': direction, 
                                     'position': str(i), 'adapter_percent': adapter})
    data['adapter_content'] = pd.DataFrame(adapter_list)

    # --- N Content ---
    n_list = []
    for sample in samples:
        for direction in directions:
            for i in range(1, 151):
                n_content = np.random.uniform(0, 0.2)
                n_list.append({'sample': sample, 'direction': direction, 
                               'position': str(i), 'N': n_content})
    data['per_base_n_content'] = pd.DataFrame(n_list)

    # --- Length Distribution ---
    len_list = []
    for sample in samples:
        for direction in directions:
            for length in range(145, 152):
                count = (100000 - abs(length - 150)**3 * 50) + np.random.randint(-500, 500)
                len_list.append({'sample': sample, 'direction': direction, 
                                 'length': length, 'count': count})
    data['length_distribution'] = pd.DataFrame(len_list)
    
    # --- Overrepresented Seqs ---
    seqs = ['AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC', 'GATCGGAAGAGCACACGTCTGAACTCCAGTCAC'
            'AAGAGCACACGTCTGAACTCCAGTCAC', 'CATTGATCGGAAGAGCACACGTCTGAACTCCAGTCA']
    overrep_list = []
    for sample in samples:
        for i, seq in enumerate(np.random.choice(seqs, 3, replace=False)):
            percent = (5 - i) * 0.5 + np.random.uniform(-0.1, 0.1)
            overrep_list.append({'sample': sample, 'seq': seq, 
                                 'percentage': percent})
    data['overrepresented_seqs'] = pd.DataFrame(overrep_list)

    return data

if __name__ == "__main__":
    # 1. Generate some fake FastQC data
    dummy_results = generate_dummy_data()
    # 2. Instantiate the plotting class
    #    Set show_individual_samples=True to see the grey background lines
    qc_plots = FastQCPlots(results=dummy_results, show_individual_samples=True)
    
    # 3. Export all generated plots to a directory
    qc_plots.export_figures(export_dir="fastqc_interactive_plots")
    qc_plots.export_figures(export_dir="fastqc_interactive_plots")
    