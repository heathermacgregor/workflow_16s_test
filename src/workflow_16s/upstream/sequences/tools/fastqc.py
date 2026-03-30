# workflow_16s/upstream/sequences/fastqc.py

import gzip
import io
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.Seq import Seq
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from workflow_16s.utils.logger import get_logger, with_logger

from .constants import (
    DEFAULT_REGIONS, DEFAULT_PRIMER_REGIONS, DEFAULT_16S_PRIMERS
)

class FastQCWrapper:
    """A modern wrapper for the FastQC command-line tool. 
    
    Provides methods to run FastQC on FASTQ files and parse the resulting reports.
    
    Attributes:
        max_workers (int): Maximum number of parallel workers to use.
    Methods:
        run_and_parse(sample_files: Dict[str, List[Union[str, Path]]], output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]: Runs FastQC on the provided FASTQ files and returns parsed data as DataFrames.
        _run_fastqc(file_path: Path, output_dir: Path) -> Path: Runs FastQC on a single file and returns the path to the resulting ZIP file.
        _parse_zip_files(zip_paths: List[Tuple[str, Path]]) -> Dict[str, pd.DataFrame]: Parses multiple FastQC ZIP files and returns a dictionary of DataFrames.
        _parse_fastqc_data(file_handle, sample: str, direction: str, data_agg: dict): Parses the content of a FastQC data file and aggregates the results.
    """
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or os.cpu_count() or 1
        logger = get_logger("workflow_16s")
        logger.info("Initialized FastQCWrapper.")

    def run_and_parse(self, sample_files: Dict[str, List[Union[str, Path]]], output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        flat_files = [(s, Path(p)) for s, paths in sample_files.items() for p in paths]
        zip_paths = []
        
        with get_progress_bar() as progress:
            task = progress.add_task("[yellow]Running FastQC...", total=len(flat_files))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._run_fastqc, p, out_dir): s for s, p in flat_files}
                for future in as_completed(futures):
                    zip_paths.append((futures[future], future.result()))
                    progress.advance(task)
        
        # Parse the data
        parsed_data = self._parse_zip_files(zip_paths)
        
        # 🧹 CRITICAL STORAGE OPTIMIZATION: Delete the bulky ZIP and HTML files
        for _, zip_path in zip_paths:
            if zip_path.exists():
                zip_path.unlink() # Delete .zip
            html_path = zip_path.with_suffix('.html')
            # FastQC sometimes names the html slightly differently based on the fastq name
            html_path_alt = zip_path.parent / zip_path.name.replace('_fastqc.zip', '_fastqc.html')
            if html_path.exists(): html_path.unlink()
            if html_path_alt.exists(): html_path_alt.unlink()
                
        return parsed_data

    def _run_fastqc(self, file_path: Path, output_dir: Path) -> Path:
        command = ["fastqc", "-q", "-o", str(output_dir), str(file_path)]
        _run_command(command, f"FastQC for {file_path.name}")
        return output_dir / f"{file_path.stem}_fastqc.zip"
        
    def _parse_zip_files(self, zip_paths: List[Tuple[str, Path]]) -> Dict[str, pd.DataFrame]:
        parsed_data = {'quality_scores': [], 'per_base_n_content': [], 'overrepresented_seqs': []}
        for sample_name, path in zip_paths:
            if not path.exists(): continue
            with zipfile.ZipFile(path) as z:
                data_file_list = [name for name in z.namelist() if name.endswith('fastqc_data.txt')]
                if not data_file_list:
                    logger.warning(f"Could not find 'fastqc_data.txt' in {path.name}")
                    continue
                data_file = data_file_list[0]
                with z.open(data_file) as f:
                    direction = "R2" if "_R2_" in Path(data_file).stem else "R1"
                    self._parse_fastqc_data(io.TextIOWrapper(f, encoding='utf-8'), sample_name, direction, parsed_data)
        return {key: pd.DataFrame(value) for key, value in parsed_data.items() if value}

    def _parse_fastqc_data(self, file_handle, sample, direction, data_agg):
        current_module = None
        for line in file_handle:
            if line.startswith('>>END_MODULE'): current_module = None; continue
            if line.startswith('>>'):
                current_module = line.strip().split('\t')[0][2:]
                logger.debug(f"Found module '{current_module}' in FastQC report for {sample}/{direction}")
                continue
            if line.startswith('#') or current_module is None: continue
            parts = line.strip().split('\t')
            try:
                if current_module == 'Per base sequence quality':
                    data_agg['quality_scores'].append({'sample': sample, 'direction': direction, 'base': parts[0], 'mean_quality': float(parts[1])})
                elif current_module == 'Per base N content':
                    data_agg['per_base_n_content'].append({'sample': sample, 'direction': direction, 'position': parts[0], 'N': float(parts[1])})
                elif current_module == 'Overrepresented sequences' and len(parts) >= 3 and parts[0] != 'No Hit':
                    data_agg['overrepresented_seqs'].append({'sample': sample, 'seq': parts[0], 'percentage': float(parts[2])})
            except (ValueError, IndexError):
                logger.warning(f"Could not parse line in module '{current_module}' for {sample}: {line.strip()}")
                continue

# ================================ PLOTTING CLASS ================================== #

class FastQCPlotter:
    """
    Generates a suite of modern, interactive plots from parsed FastQC data. 🎨
    """
    def __init__(self, parsed_data: Dict[str, pd.DataFrame], show_individual: bool = False):
        self.data = parsed_data
        self.show_individual = show_individual
        logger = get_logger("workflow_16s")
        logger.info("Initialized FastQCPlotter.")

    def _safe_get_data(self, section: str) -> Optional[pd.DataFrame]:
        if section not in self.data or self.data[section].empty:
            logger.warning(f"Plotting skipped: Missing or empty data for '{section}'.")
            return None
        df = self.data[section].copy()
        for col in ['base', 'position']:
            if col in df.columns:
                df[f'{col}_sort'] = df[col].astype(str).str.split('-').str[0].astype(int)
                df = df.sort_values(f'{col}_sort')
        return df

    def _create_summary_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str) -> go.Figure:
        fig = go.Figure()
        sort_col = f'{x_col}_sort' if f'{x_col}_sort' in df else x_col
        agg_df = df.groupby(['direction', x_col]).agg(mean_val=(y_col, 'mean'), std_val=(y_col, 'std'), sort_key=(sort_col, 'first')).reset_index().sort_values('sort_key')
        colors = px.colors.qualitative.Plotly
        for i, (direction, group) in enumerate(agg_df.groupby('direction')):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=group[x_col], y=group['mean_val'], name=f'{direction} Mean', mode='lines', line=dict(color=color, width=2.5)))
            fig.add_trace(go.Scatter(x=pd.concat([group[x_col], group[x_col][::-1]]), y=pd.concat([group['mean_val'] + group['std_val'], (group['mean_val'] - group['std_val'])[::-1]]), fill='toself', fillcolor=color, opacity=0.2, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        if self.show_individual:
            for _, group in df.groupby(['sample', 'direction']):
                fig.add_trace(go.Scatter(x=group[x_col], y=group[y_col], name=f"{group['sample'].iloc[0]} ({group['direction'].iloc[0]})", mode='lines', line=dict(color='grey', width=0.5), opacity=0.5, visible='legendonly'))
        fig.update_layout(title_text=f"<b>{title}</b>", xaxis_title=x_label, yaxis_title=y_label, legend_title_text="Trace", title_x=0.5)
        fig.update_layout(
            title_text=f"<b>{title}</b>",
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title_text="Trace",
            title_x=0.5,
            template="plotly_white"
        )
        return fig
    
    def plot_quality_scores(self) -> Optional[go.Figure]:
        df = self._safe_get_data('quality_scores')
        return self._create_summary_line_plot(df, 'base', 'mean_quality', "Per-Base Sequence Quality", "Position in Read (bp)", "Phred Quality Score") if df is not None else None

    def plot_per_base_n_content(self) -> Optional[go.Figure]:
        df = self._safe_get_data('per_base_n_content')
        return self._create_summary_line_plot(df, 'position', 'N', "Per-Base 'N' Content", "Position in Read (bp)", "'N' Content (%)") if df is not None else None

    # 🚀 FIXED: Separated this into its own proper method
    def plot_overrepresented_seqs(self) -> Optional[go.Figure]:
        df = self._safe_get_data('overrepresented_seqs')
        if df is None: return None
        
        agg_df = df.groupby('seq')['percentage'].sum().nlargest(15).reset_index()
        agg_df['label'] = agg_df['seq'].str.slice(0, 40) + '...'
        fig = px.bar(
            agg_df, x='percentage', y='label', orientation='h', 
            title=f"<b>Top 15 Overrepresented Sequences</b>", 
            labels={'label': 'Sequence', 'percentage': 'Total %'}, 
            hover_data={'label': False, 'seq': True}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5, template="plotly_white")
        return fig

    def export_all_plots(self, export_dir: str):
        output_path = Path(export_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"🎨 Exporting interactive plots to: {output_path.resolve()}")
        plot_methods = [f"plot_{key}" for key in self.data.keys()]
        for method_name in plot_methods:
            if hasattr(self, method_name):
                try:
                    fig = getattr(self, method_name)()
                    if fig:
                        fig.write_html(str(output_path / f"{method_name.replace('plot_', '')}.html"))
                except Exception as e:
                    logger.error(f"Failed to generate plot for {method_name}: {e}")
        print("✅ Plotting complete.")