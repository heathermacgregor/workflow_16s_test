# ===================================== IMPORTS ====================================== #
import gzip
import io
import json
import logging
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

# Assuming a progress bar utility like rich.progress is available
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_REGIONS = {
    'V1-V2': (100, 400),
    'V2-V3': (200, 500),
    'V3-V4': (350, 800),
    'V4': (515, 806),
    'V4-V5': (515, 950),
    'V5-V7': (830, 1190),
    'V6-V8': (900, 1400),
    'V7-V9': (1100, 1500)
}

DEFAULT_PRIMER_REGIONS = {
    "V1-V2": ("AGAGTTTGATCMTGGCTCAG", "TGCTGCCTCCCGTAGGAGT"),
    "V2-V3": ("ACTCCTACGGGAGGCAGCAG", "TTACCGCGGCTGCTGGCAC"),
    "V3-V4": ("CCTACGGGNGGCWGCAG", "GACTACHVGGGTATCTAATCC"),
    "V4": ("GTGCCAGCMGCCGCGGTAA", "GGACTACHVGGGTWTCTAAT"),
    "V4-V5": ("GTGYCAGCMGCCGCGGTAA", "CCGYCAATTYMTTTRAGTTT"),
    "V6-V8": ("AAACTYAAAKGAATTGACGG", "ACGGGCGGTGTGTACAAG")
}

DEFAULT_16S_PRIMERS = {
    "V1-V2": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "AGAGTTTGATCMTGGCTCAG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "TGCTGCCTCCCGTAGGAGT",
            "ref": None,
        },
    },
    "V2-V3": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "ACTCCTACGGGAGGCAGCAG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "TTACCGCGGCTGCTGGCAC",
            "ref": None,
        },
    },
    "V3-V4": {
        "fwd": {
            "name": "Bakt_341F",
            "full_name": "S-D-Bact-0341-b-S-17",
            "position": (341, 357),
            "seq": "CCTACGGGNGGCWGCAG",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
        "rev": {
            "name": "Bakt_805R",
            "full_name": "S-D-Bact-0785-a-A-21",
            "position": (785, 805),
            "seq": "GACTACHVGGGTATCTAATCC",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
    },
    "V4": {
        "fwd": {
            "name": "U515F",
            "full_name": "S-*-Univ-0515-a-S-19",
            "position": (515, 533),
            "seq": "GTGCCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
        "rev": {
            "name": "806R",
            "full_name": "S-D-Bact-0787-b-A-20",
            "position": (787, 808),
            "seq": "GGACTACHVGGGTWTCTAAT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
    },
    "V4-V5": {
        "fwd": {
            "name": "515F-Y",
            "full_name": None,
            "position": (515, 533),
            "seq": "GTGYCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
        "rev": {
            "name": "926R",
            "full_name": "S-D-Bact-0907-a-A-19",
            "position": (907, 926),
            "seq": "CCGYCAATTYMTTTRAGTTT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
    },
    "V6-V8": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "AAACTYAAAKGAATTGACGG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "ACGGGCGGTGTGTACAAG",
            "ref": None,
        },
    },
}

# =============================== UTILITY FUNCTIONS ================================ #

def get_progress_bar():
    """Returns a pre-configured Rich progress bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True
    )

def _run_command(command: List[str], description: str = "Command") -> Tuple[str, str]:
    """
    Runs a subprocess command with standardized error handling and logging.
    """
    logger.debug(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        return process.stdout, process.stderr
    except FileNotFoundError:
        raise RuntimeError(f"Error: The command '{command[0]}' was not found. Is it installed and in your PATH?")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"{description} failed with exit code {e.returncode}.\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Stderr: {e.stderr.strip()}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message) from e

# ============================= DATA PROCESSING CLASSES ============================== #

class SeqKitWrapper:
    """A modern wrapper for the SeqKit command-line tool. 
    Provides methods to analyze FASTQ files and summarize statistics.
    
    Attributes:
        max_workers (int): Maximum number of parallel workers to use.
        version (str): Detected version of SeqKit.
    Methods:
        analyze(sample_files: Dict[str, List[Union[str, Path]]]) -> pd.DataFrame: Analyzes the provided FASTQ files and returns a summary DataFrame.
        _process_file(file_path: Path) -> dict: Processes a single FASTQ file and returns its statistics as a dictionary.
    """
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or os.cpu_count() or 1
        self.version = self._get_version()
        logger.info(f"Initialized SeqKitWrapper (using SeqKit v{self.version}).")

    def _get_version(self) -> str:
        stdout, _ = _run_command(["seqkit", "version"], "SeqKit version check")
        return stdout.strip().split()[-1]

    def analyze(self, sample_files: Dict[str, List[Union[str, Path]]]) -> pd.DataFrame:
        flat_files = [(sample, Path(p)) for sample, paths in sample_files.items() for p in paths]
        all_stats = []
        with get_progress_bar() as progress:
            task = progress.add_task("[cyan]Running SeqKit...", total=len(flat_files))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_file, p): (s, p) for s, p in flat_files}
                for future in as_completed(futures):
                    sample_name, _ = futures[future]
                    stats_dict = future.result()
                    stats_dict['sample'] = sample_name
                    all_stats.append(stats_dict)
                    progress.advance(task)
        if not all_stats:
            logger.warning("No statistics were generated by SeqKit.")
            return pd.DataFrame()
        df = pd.DataFrame(all_stats)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        agg_functions = {col: 'sum' for col in ['num_seqs', 'sum_len', 'sum_gap']}
        agg_functions.update({col: 'min' for col in ['min_len']})
        agg_functions.update({col: 'max' for col in ['max_len']})
        valid_agg_functions = {k: v for k, v in agg_functions.items() if k in df.columns}
        sample_summary = df.groupby('sample').agg(valid_agg_functions).reset_index()
        if 'GC' in numeric_cols and 'sum_len' in sample_summary.columns:
            gc_weighted_avg = df.groupby('sample').apply(lambda x: np.average(x['GC'], weights=x['sum_len'])).reset_index(name='GC')
            sample_summary = pd.merge(sample_summary, gc_weighted_avg, on='sample')
        overall_summary = df.agg(valid_agg_functions)
        overall_summary['sample'] = 'OVERALL'
        if 'GC' in numeric_cols and 'sum_len' in df.columns and df['sum_len'].sum() > 0:
             overall_summary['GC'] = np.average(df['GC'], weights=df['sum_len'])
        final_df = pd.concat([sample_summary, pd.DataFrame([overall_summary])], ignore_index=True)
        if 'sum_len' in final_df.columns and 'num_seqs' in final_df.columns:
            final_df['avg_len'] = (final_df['sum_len'] / final_df['num_seqs'])
        return final_df.round(2)

    def _process_file(self, file_path: Path) -> dict:
        command = ["seqkit", "stats", "-T", "-a", str(file_path)]
        stdout, _ = _run_command(command, f"SeqKit stats for {file_path.name}")
        lines = stdout.strip().split('\n')
        if len(lines) < 2:
            raise ValueError(f"Unexpected output from SeqKit for {file_path.name}: {stdout}")
        headers = [h.replace('.', '').replace('%', '') for h in lines[0].split('\t')]
        values = lines[1].split('\t')
        stats = {}
        for h, v in zip(headers, values):
            try:
                stats[h] = float(v.replace(',', ''))
            except ValueError:
                stats[h] = v
        return stats

class CutAdaptWrapper:
    """A modern wrapper for the CutAdapt command-line tool. 
    Provides methods to trim adapters/primers from FASTQ files and summarize trimming statistics.
    Attributes:
        fwd_primer (str): Forward primer sequence.
        rev_primer (str): Reverse primer sequence.
        min_length (int): Minimum length of reads to keep after trimming.
        quality_cutoff (int): Quality score cutoff for trimming.
        cores_per_job (int): Number of CPU cores to allocate per CutAdapt job.
    Methods:
        trim(sample_files: Dict[str, List[Union[str, Path]]], output_dir: Union[str, Path], max_workers: int = None) -> Tuple[Dict[str, List[Path]], pd.DataFrame]: Trims the provided FASTQ files and returns paths to trimmed files and
        a summary DataFrame.
        _prepare_task(sample_name: str, paths: list, out_dir: Path) -> dict: Prepares a task dictionary for a sample.
        _build_command(task: dict) -> List[str]: Builds the CutAdapt command for a task.
        _process_sample(task: dict) -> Path: Processes a single sample using CutAdapt and returns the path to its JSON report.
        _parse_json_reports(json_paths: List[Path]) -> pd.DataFrame: Parses multiple CutAdapt JSON reports and returns a summary DataFrame.
    """
    def __init__(self, fwd_primer: str, rev_primer: str, min_length: int = 150, 
                 quality_cutoff: int = 20, cores_per_job: int = 4):
        self.fwd_primer = fwd_primer
        self.rev_primer = rev_primer
        self.min_length = min_length
        self.quality_cutoff = quality_cutoff
        self.cores_per_job = cores_per_job
        logger.info("Initialized CutAdaptWrapper.")

    def trim(self, sample_files: Dict[str, List[Union[str, Path]]], 
             output_dir: Union[str, Path], 
             max_workers: Optional[int] = None) -> Tuple[Dict[str, List[Path]], pd.DataFrame]:
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        max_workers = max_workers or os.cpu_count() or 1
        tasks = [self._prepare_task(s, p, out_dir) for s, p in sample_files.items()]
        json_paths = []
        with get_progress_bar() as progress:
            prog_task = progress.add_task("Trimming with CutAdapt...", total=len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_sample, task): task for task in tasks}
                for future in as_completed(futures):
                    json_paths.append(future.result())
                    progress.advance(prog_task)
        summary_df = self._parse_json_reports(json_paths)
        trimmed_paths = {task['sample_name']: task['output_paths'] for task in tasks}
        return trimmed_paths, summary_df

    def _prepare_task(self, sample_name: str, paths: list, out_dir: Path) -> dict:
        input_paths = [Path(p) for p in paths]
        output_paths = [out_dir / p.name for p in input_paths]
        json_path = out_dir / f"{sample_name}.cutadapt.json"
        return {"sample_name": sample_name, "input_paths": input_paths, "output_paths": output_paths, "json_path": json_path}

    def _build_command(self, task: dict) -> List[str]:
        cmd = [
            "cutadapt", f"--json={task['json_path']}", "--cores", str(self.cores_per_job),
            "-m", str(self.min_length), "-q", str(self.quality_cutoff),
            "--discard-untrimmed"
        ]
        if len(task['input_paths']) == 2: # Paired-end trimming
            cmd.extend(["-g", self.fwd_primer, "-G", self.rev_primer, 
                        "-o", str(task['output_paths'][0]), 
                        "-p", str(task['output_paths'][1])])
        else: # Single-end trimming
            cmd.extend(["-g", self.fwd_primer, "-o", str(task['output_paths'][0])])
        cmd.extend([str(p) for p in task['input_paths']])
        return cmd

    def _process_sample(self, task: dict) -> Path:
        command = self._build_command(task)
        _run_command(command, f"CutAdapt for {task['sample_name']}")
        return task['json_path']

    def _parse_json_reports(self, json_paths: List[Path]) -> pd.DataFrame:
        records = []
        for path in json_paths:
            if not path.exists(): continue
            with open(path) as f:
                data = json.load(f)
            records.append({'sample': Path(path).stem.replace('.cutadapt', ''), 
                            'reads_processed': data['read_counts']['input'], 
                            'reads_written': data['read_counts']['output']})
        if not records: return pd.DataFrame()
        df = pd.DataFrame(records)
        df['percent_reads_kept'] = (df['reads_written'] / df['reads_processed'] * 100).round(2)
        return df

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
        return self._parse_zip_files(zip_paths)

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

        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            title_x=0.5,
            template="plotly_white"
        )
        return fig
        if df is None: return None
        agg_df = df.groupby('seq')['percentage'].sum().nlargest(15).reset_index()
        agg_df['label'] = agg_df['seq'].str.slice(0, 40) + '...'
        fig = px.bar(agg_df, x='percentage', y='label', orientation='h', title=f"<b>Top 15 Overrepresented Sequences</b>", labels={'label': 'Sequence', 'percentage': 'Total %'}, hover_data={'label': False, 'seq': True})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
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

# ================================== EXAMPLE USAGE =================================== #

def create_dummy_paired_fastqs(r1_path: Path, r2_path: Path, n_reads: int = 100):
    """Creates realistic gzipped paired-end FASTQ files without artificial linkers."""
    primer_fwd = "CCTACGGGNGGCWGCAG"
    primer_rev = "GACTACHVGGGTATCTAATCC"
    
    with gzip.open(r1_path, 'wt', encoding='ascii') as f1, gzip.open(r2_path, 'wt', encoding='ascii') as f2:
        for i in range(n_reads):
            # Create a random inner sequence of realistic length
            random_seq_fwd = Seq("".join(np.random.choice(list("ATGC"), size=145)))
            random_seq_rev = random_seq_fwd.reverse_complement()
            
            # Create R1 read: primer + biological sequence
            seq1 = primer_fwd + str(random_seq_fwd)
            qual1 = "".join(np.random.choice(list("?@ABCDEFGHIJ"), size=len(seq1)))
            f1.write(f"@READ_{i}/1\n{seq1}\n+\n{qual1}\n")
            
            # Create R2 read: primer + biological sequence
            seq2 = primer_rev + str(random_seq_rev)
            qual2 = "".join(np.random.choice(list("?@ABCDEFGHIJ"), size=len(seq2)))
            f2.write(f"@READ_{i}/2\n{seq2}\n+\n{qual2}\n")

if __name__ == "__main__":
    # --- 1. Setup a dummy project directory and data ---
    project_dir = Path("bioinformatics_pipeline_output")
    if project_dir.exists(): shutil.rmtree(project_dir)
    raw_dir, trimmed_dir, fastqc_dir, plot_dir = [project_dir / d for d in ["01_raw", "02_trimmed", "03_fastqc", "04_plots"]]
    raw_dir.mkdir(parents=True)

    from typing import Dict, List, Union

    samples: Dict[str, List[Union[str, Path]]] = {
        "sample_A": [raw_dir / "sample_A_R1.fastq.gz", raw_dir / "sample_A_R2.fastq.gz"],
        "sample_B": [raw_dir / "sample_B_R1.fastq.gz", raw_dir / "sample_B_R2.fastq.gz"]
    }
    for s, paths in samples.items():
        create_dummy_paired_fastqs(r1_path=Path(paths[0]), r2_path=Path(paths[1]))
        
    logger.info(f"📁 Created dummy project in: {project_dir.resolve()}")

    # --- 2. Run the full pipeline ---
    logger.info("🔬 STEP 1: Running SeqKit for initial statistics...")
    initial_stats_df = SeqKitWrapper().analyze(samples)
    print("Initial Stats Summary:\n", initial_stats_df.to_string())

    logger.info("✂️  STEP 2: Running CutAdapt to trim reads...")
    trimmed_files, trim_summary_df = CutAdaptWrapper(
        fwd_primer="CCTACGGGNGGCWGCAG", 
        rev_primer="GACTACHVGGGTATCTAATCC",
        min_length=100
    ).trim(samples, trimmed_dir)
    print("Trimming Summary:\n", trim_summary_df.to_string())

    logger.info("📊 STEP 3: Running FastQC on trimmed reads...")
    trimmed_files_path = {k: [Path(p) for p in v] for k, v in trimmed_files.items()}
    parsed_fastqc_data = FastQCWrapper().run_and_parse(trimmed_files_path, fastqc_dir) # type: ignore

    # --- 3. Generate and export plots from the parsed data ---
    if parsed_fastqc_data:
        logger.info("🎨 STEP 4: Generating interactive plots from FastQC results...")
        plotter = FastQCPlotter(parsed_data=parsed_fastqc_data, show_individual=True)
        plotter.export_all_plots(export_dir=str(plot_dir))
    else:
        logger.warning("Skipping plotting step because no FastQC data was parsed.")
    
    logger.info(f"\n✅ Pipeline finished. Check results in '{project_dir.resolve()}'")