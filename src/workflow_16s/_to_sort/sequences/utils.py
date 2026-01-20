# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import gzip
import io
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from matplotlib.colors import LogNorm

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from workflow_16s.figures.fastqc.fastqc import FastQCPlots
from workflow_16s.utils.progress import get_progress_bar 

project_root = str(Path(__file__).resolve().parent.parent.parent) # Adjust .parent count
sys.path.append(project_root)
import workflow_16s.custom_tmp_config  

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= DEFAULT VALUES =================================== #

DEFAULT_RERUN_CUTADAPT = True

DEFAULT_FASTQC_PATH = 'fastqc'

DEFAULT_N_CORES_SEQKIT = 4 #os.cpu_count() or 1
DEFAULT_N_CORES_CUTADAPT = 16
DEFAULT_MAX_WORKERS = 1

DEFAULT_PROGRESS_TEXT_N = 65
DEFAULT_N = DEFAULT_PROGRESS_TEXT_N

DEFAULT_START_TRIM = 0
DEFAULT_END_TRIM = 0
DEFAULT_START_Q_CUTOFF = 30
DEFAULT_END_Q_CUTOFF = 15
DEFAULT_MIN_SEQ_LENGTH = 150

DEFAULT_TARGET_REGION = 'V4'

# ==================================== FUNCTIONS ===================================== #

def get_all_values(dictionary):
    values = []
    for value in dictionary.values():
        # Check if the value is a nested dictionary
        if isinstance(value, dict):  
            values.extend(get_all_values(value))  
        else:
            values.append(value)  
    return values


def import_seqs_fasta(fasta_path: Union[str, Path]):
    seqs = dict(zip((record.id for record in SeqIO.parse(fasta_path, "fasta")), 
                    (str(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))))
    return seqs

# ============================== FILE FORMAT CONVERSION ============================== #

def fastq_gz_to_fasta(fastq_file: Union[str, Path], n_sequences: int = 0) -> Path:
    """
    Converts a FASTQ.GZ file to a FASTA file.

    Args:
        fastq_file:  Path to the input FASTQ.GZ file.
        n_sequences: Maximum number of sequences to convert.

    Returns:
        fasta_file:  Path to the created FASTA file.
    """
    fastq_file = Path(fastq_file)
    fasta_file = fastq_file.with_suffix("").with_suffix(".fasta")
    
    try:
        with gzip.open(fastq_file, "rt") as fastq, open(fasta_file, "w") as fasta:
            seq_count = 0
            for i, line in enumerate(fastq):
                if i % 4 == 0:  # Sequence identifier
                    if not line.startswith("@"):
                        raise ValueError(
                            f"Invalid FASTQ format in file {fastq_file} "
                            f"at line {i + 1}"
                        )
                    fasta.write(">" + line[1:])  # Convert to FASTA header
                elif i % 4 == 1:  # Sequence
                    fasta.write(line)
                    seq_count += 1
                if n_sequences > 0 and seq_count >= n_sequences:
                    break
        return fasta_file
    except Exception as e:
        raise RuntimeError(f"Error converting {fastq_file} to FASTA: {e}")

# ===================================== SEQKIT ======================================= #

class SeqKit:
    """
    A class to handle parallel processing of FASTQ.GZ files using SeqKit 
    (https://bioinf.shenwei.me/seqkit/).
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, self.DEFAULT_N_CORES_SEQKIT // 2)
        self._seqkit_version = self._get_seqkit_version()

    def _get_seqkit_version(self) -> str:
        """Get SeqKit version for compatibility checks."""
        result = subprocess.run(
            ["seqkit", "version"], capture_output=True, text=True
        )
        return result.stdout.strip().split()[-1]

    def _run_seqkit(self, cmd: List[str]) -> str:
        """Execute SeqKit command with error handling."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SeqKit error: {result.stderr.strip()}")
        return result.stdout

    def analyze_samples(
        self, samples: Dict[str, List[Union[str, Path]]]
    ) -> Dict[str, Any]:
        """Analyze multiple FASTQ.GZ files with parallel processing."""
        file_list = self._flatten_samples(samples)
        agg_stats, overall = self._init_aggregators(samples)
        
        with get_progress_bar() as progress:
            desc = "Running SeqKit..."
            task = progress.add_task(
                f"[white]{desc:<{DEFAULT_N}}",
                total=len(file_list)
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_file, sample, fpath
                    ): (sample, fpath)
                    for sample, fpath in file_list
                }
                
                for future in as_completed(futures):
                    progress.advance(task)
                    sample, stats = future.result()
                    self._aggregate_stats(sample, stats, agg_stats, overall)
                    
            elapsed = progress.tasks[task].elapsed
        
        return self._finalize_results(agg_stats, overall, elapsed)

    def _process_file(self, sample: str, fpath: Path) -> tuple:
        """Process single file with SeqKit stats and length distribution."""
        fpath = Path(fpath)
        
        # Get basic statistics
        stats_output = self._run_seqkit([
            "seqkit", "stats", "-T", "--all", str(fpath)
        ])
        stats = self._parse_stats(stats_output)
        
        # Get length distribution
        lengths_output = self._run_seqkit([
            "seqkit", "fx2tab", "-n", "-l", str(fpath)
        ])
        length_counts = Counter(
            int(line.split("\t")[1]) 
            for line in lengths_output.strip().split("\n")
        )
        
        return (sample, {
            "total_seqs": stats["num_seqs"],
            "total_bases": stats["sum_len"],
            "min_length": stats["min_len"],
            "max_length": stats["max_len"],
            "length_counts": length_counts,
            "file": fpath.name
        })

    def _parse_stats(self, output: str) -> Dict[str, int]:
        """Parse SeqKit stats table output."""
        lines = output.strip().split('\n')
        headers = [h.lower().replace(' ', '_') for h in lines[0].split('\t')]
        values = [int(v) if v.isdigit() else v for v in lines[1].split('\t')]
        return dict(zip(headers, values))

    def _flatten_samples(self, samples: Dict) -> List[tuple]:
        return [(s, Path(p)) for s, ps in samples.items() for p in ps]

    def _init_aggregators(self, samples: Dict) -> tuple:
        agg_stats = defaultdict(lambda: {
            "total_seqs": 0,
            "total_bases": 0,
            "min_length": float('inf'),
            "max_length": 0,
            "length_counts": Counter(),
            "files": []
        })
        
        overall = {
            "total_samples": len(samples),
            "total_files": sum(len(ps) for ps in samples.values()),
            "total_seqs": 0,
            "total_bases": 0,
            "global_min": float('inf'),
            "global_max": 0,
            "length_distribution": Counter()
        }
        
        return agg_stats, overall

    def _aggregate_stats(
        self, sample: str, stats: Dict, agg: Dict, overall: Dict
    ) -> None:
        """Update aggregators with new file statistics."""
        # Sample-level aggregation
        agg[sample]["total_seqs"] += stats["total_seqs"]
        agg[sample]["total_bases"] += stats["total_bases"]
        agg[sample]["min_length"] = min(
            agg[sample]["min_length"], stats["min_length"]
        )
        agg[sample]["max_length"] = max(
            agg[sample]["max_length"], stats["max_length"]
        )
        agg[sample]["length_counts"].update(stats["length_counts"])
        agg[sample]["files"].append(stats["file"])
        
        # Overall aggregation
        overall["total_seqs"] += stats["total_seqs"]
        overall["total_bases"] += stats["total_bases"]
        overall["global_min"] = min(
            overall["global_min"], stats["min_length"]
        )
        overall["global_max"] = max(
            overall["global_max"], stats["max_length"]
        )
        overall["length_distribution"].update(stats["length_counts"])

    def _calculate_avg(self, bases: int, sequences: int) -> float:
        """Helper to safely calculate average length."""
        return bases / sequences if sequences else 0.0

    def _build_sample_metrics(self, data: dict) -> dict:
        """Construct metrics dictionary for a single sample."""
        return {
            "total_sequences": data["total_seqs"],
            "total_bases": data["total_bases"],
            "avg_length": self._calculate_avg(
                data["total_bases"], data["total_seqs"]
            ),
            "min_length": data["min_length"],
            "max_length": data["max_length"],
            "file_count": len(data["files"]),
            "files": data["files"],
            "top_lengths": data["length_counts"].most_common(5)
        }

    def _build_overall_metrics(self, data: dict) -> dict:
        """Construct metrics dictionary for overall results."""
        return {
            "total_samples": data["total_samples"],
            "total_files": data["total_files"],
            "total_sequences": data["total_seqs"],
            "total_bases": data["total_bases"],
            "avg_length": self._calculate_avg(
                data["total_bases"], data["total_seqs"]
            ),
            "min_length": data["global_min"],
            "max_length": data["global_max"],
            "most_common_lengths": data["length_distribution"].most_common(10)
        }

    def _build_timing_metrics(
        self, elapsed: float, files: int, sequences: int
    ) -> dict:
        """Construct processing time metrics dictionary."""
        return {
            "execution_seconds": elapsed,
            "files_per_second": files / elapsed if elapsed > 0 else 0,
            "sequences_per_second": sequences / elapsed if elapsed > 0 else 0
        }
        
    def _finalize_results(
        self, agg_stats: Dict, overall: Dict, elapsed: float
    ) -> Dict:
        """Calculate final metrics and structure results."""
        # Main construction logic
        return {
            "samples": {
                sample: self._build_sample_metrics(data) 
                for sample, data in agg_stats.items()
            },
            "overall": self._build_overall_metrics(overall),
            "proc_time": self._build_timing_metrics(
                elapsed, 
                overall["total_files"], 
                overall["total_seqs"]
            )
        }
        
# ===================================== CUTADAPT ===================================== #

class CutAdapt:
    """
    A class to handle parallel processing of FASTQ.GZ files using CutAdapt 
    (https://doi.org/10.14806/ej.17.1.200).
    
    Attributes:
        fastq_dir:
        trimmed_fastq_dir:
        primer_fwd:
        primer_rev:
        start_trim:
        end_trim:
        start_q_cutoff:
        end_q_cutoff:
        min_seq_length:
        cores:
        rerun:
        region:
    """
    
    def __init__(
        self,
        fastq_dir: Union[str, Path],
        trimmed_fastq_dir: Union[str, Path],
        primer_fwd: str,
        primer_rev: str,
        start_trim: int = DEFAULT_START_TRIM,
        end_trim: int = DEFAULT_END_TRIM,
        start_q_cutoff: int = DEFAULT_START_Q_CUTOFF,
        end_q_cutoff: int = DEFAULT_END_Q_CUTOFF,
        min_seq_length: int = DEFAULT_MIN_SEQ_LENGTH,
        cores: int = DEFAULT_N_CORES_CUTADAPT,
        rerun: bool = DEFAULT_RERUN_CUTADAPT,
        region: str = DEFAULT_TARGET_REGION,
    ):
        self.fastq_dir = Path(fastq_dir)
        self.trimmed_fastq_dir = Path(trimmed_fastq_dir)
        self.primer_fwd = primer_fwd
        self.primer_rev = primer_rev
        self.start_trim = start_trim
        self.end_trim = end_trim
        self.start_q_cutoff = start_q_cutoff
        self.end_q_cutoff = end_q_cutoff
        self.min_seq_length = min_seq_length
        self.cores = cores
        self.rerun = rerun
        self.region = region

    @staticmethod
    def get_stdout(cmd: list) -> str:
        """Runs a subprocess command and returns stdout, exits on error."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error in CutAdapt: {result.stderr}")
            raise RuntimeError(f"CutAdapt error: {result.stderr}")
        return result.stdout

    @staticmethod
    def run_cutadapt(parameters: Dict) -> Union[str, None]:
        """Runs CutAdapt with specified parameters."""
        try:
            command = ["conda", "run", "-n", "workflow_16s", 
                       "cutadapt", "--report=minimal"]
            if parameters["cores"] is not None:
                command.extend(["--cores", str(parameters["cores"])])
            if (
                parameters["start_q_cutoff"] is not None
                and parameters["end_q_cutoff"] is not None
            ):
                command.extend([
                    "-q",
                    f"{parameters['start_q_cutoff']},{parameters['end_q_cutoff']}",
                ])
            if parameters["min_seq_length"] != 0:
                command.extend(["-m", str(parameters["min_seq_length"])])
            if parameters["primer_fwd"] is not None:
                command.extend(["-b", parameters["primer_fwd"]])
                command.extend(
                    ["-a", 
                     str(Seq(parameters["primer_fwd"]).reverse_complement())]
                )
            if parameters["primer_rev"] is not None:
                command.extend(["-B", parameters["primer_rev"]])
                command.extend(
                    ["-A", 
                     str(Seq(parameters["primer_rev"]).reverse_complement())]
                )
            if parameters["start_trim"] is not None:
                command.extend(["-u", str(parameters["start_trim"])])
            if parameters["end_trim"] is not None:
                command.extend(["-U", str(parameters["end_trim"])])
            # Handle input/output files
            sample_paths = parameters["sample_fastq_paths"]
            trimmed_paths = parameters["trimmed_sample_fastq_paths"]
            json_path = parameters["sample_json_file_path"]

            if len(sample_paths) == 2:
                out1, out2 = trimmed_paths
                out1_path = Path(out1)
                out2_path = Path(out2)
                if any(not p.exists() for p in (out1_path, out2_path)) or parameters["rerun"]:
                    command.extend(["-o", str(out1), "-p", str(out2)])
                    command.extend(map(str, sample_paths))
                    command.append(f"--json={json_path}")
                    result = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    return CutAdapt.get_stdout(command)
                else:
                    logger.info("Output files exist, skipping trimming.")
                    return None
            elif len(sample_paths) == 1:
                out = trimmed_paths[0]
                out_path = Path(out)
                if not out_path.exists() or parameters["rerun"]:
                    command.extend(["-o", str(out)])
                    command.append(str(sample_paths[0]))
                    command.append(f"--json={json_path}")
                    result = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    return CutAdapt.get_stdout(command)
                else:
                    return None
            else:
                error_msg = f"Unexpected number of FASTQ files: {len(sample_paths)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error processing {parameters['sample']}: {e}")
            raise

    @staticmethod
    def process_sample(params: Dict) -> Tuple[str, Union[str, Exception]]:
        """Processes a single sample with error handling."""
        try:
            result = CutadaptPipeline.run_cutadapt(params)
            return (params['sample'], result)
        except Exception as e:
            logger.error(f"Error processing {params['sample']}: {e}", exc_info=True)
            return (params['sample'], e)

    @staticmethod
    def parse_metrics_to_dataframe(data_tuples: List[Tuple[str, str]]) -> pd.DataFrame:
        """Parses metrics strings into a DataFrame."""
        rows = []
        for sample_id, metrics_str in data_tuples:
            if not metrics_str:
                continue
            lines = metrics_str.strip().split('\n')
            if len(lines) < 2:
                continue
            headers = lines[0].split('\t')
            values = lines[1].split('\t')
            if len(headers) != len(values):
                continue
            row = {'ID': sample_id}
            row.update(zip(headers, values))
            rows.append(row)
        df = pd.DataFrame(rows)
        numeric_cols = [col for col in df.columns if col not in ['ID', 'status']]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df

    def run(
        self, fastq_paths: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """Executes the CutAdapt pipeline on provided FASTQ paths."""
        samples = list(fastq_paths.keys())
        logger.info(
            f"CutAdapt Parameters:\n"
            f"  [Forward Primer]          {self.primer_fwd}\n"
            f"  [Reverse Primer]          {self.primer_rev}\n"
            f"  [Trim Range]              {self.start_trim} - {self.end_trim}\n"
            f"  [Quality Cutoffs]         {self.start_q_cutoff} - {self.end_q_cutoff}\n"
            f"  [Min Sequence Length]     {self.min_seq_length}\n"
            f"  [Target Region]           {self.region}\n"
            f"  [Workers]                 {self.cores}"
        )

        all_params = []
        trimmed_fastq_paths = {}
        for sample in samples:
            sample_paths = [self.fastq_dir / Path(p).name for p in fastq_paths[sample]]
            trimmed_paths = [self.trimmed_fastq_dir / Path(p).name for p in sample_paths]
            trimmed_fastq_paths[sample] = [str(p) for p in trimmed_paths]
            all_params.append({
                'sample': sample,
                'sample_fastq_paths': [str(p) for p in sample_paths],
                'trimmed_sample_fastq_paths': [str(p) for p in trimmed_paths],
                'sample_json_file_path': str(
                    self.trimmed_fastq_dir / 
                    f"{Path(sample_paths[0]).stem.split('_')[0]}.cutadapt.json"
                ),
                'primer_fwd': self.primer_fwd,
                'primer_rev': self.primer_rev,
                'start_trim': self.start_trim,
                'end_trim': self.end_trim,
                'start_q_cutoff': self.start_q_cutoff,
                'end_q_cutoff': self.end_q_cutoff,
                'min_seq_length': self.min_seq_length,
                'cores': 4,  
                'rerun': self.rerun,
            })
        results = []
        with get_progress_bar() as prog:
            desc = "Running CutAdapt..."
            task = prog.add_task(
                f"[white]{desc:<{DEFAULT_N}}", 
                total=len(samples)
            )
            with ThreadPoolExecutor(max_workers=self.cores) as executor:
                futures = {
                    executor.submit(self.process_sample, params): params['sample'] 
                    for params in all_params
                }
                for future in as_completed(futures):
                    sample, result = future.result()
                    results.append((sample, result))
                    prog.update(task, advance=1)

            # Calculate total processing time
            elapsed = prog.tasks[task].time_elapsed

            proc_time = {
                "execution_seconds": elapsed,
                "files_per_second": len(samples) / elapsed if elapsed > 0 else 0
            }

        df = self.parse_metrics_to_dataframe(
            [(s, r) for s, r in results if isinstance(r, str)]
        )
        return trimmed_fastq_paths, df, proc_time


# ===================================== BASIC ====================================== #

class BasicStats:
    """"""
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, DEFAULT_N_CORES_SEQKIT // 2)

    def calculate_statistics(
        self, samples: Dict[str, List[Union[str, Path]]]
    ) -> Dict[str, Any]:
        """Main method to calculate sequencing statistics for provided samples."""
        file_list = self._flatten_samples(samples)
        sample_agg, overall_stats = self._initialize_structures(samples)
        
        with get_progress_bar() as prog:
            desc = "Analyzing runs..."
            task = prog.add_task(
                f"[white]{desc:<{DEFAULT_N}}", 
                total=len(file_list)
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_file, sample, fpath
                    ): (sample, fpath)
                    for sample, fpath in file_list
                }
                
                self._process_results(futures, prog, task, sample_agg)
        
            elapsed = prog.tasks[task].elapsed
        
        return self._finalize_stats(sample_agg, overall_stats, elapsed)

    def _flatten_samples(
        self, samples: Dict[str, List[Union[str, Path]]]
    ) -> list:
        """
        Convert nested sample structure to flat list of (sample, path) tuples.
        """
        return [
            (sample, fpath)
            for sample, paths in samples.items()
            for fpath in paths
        ]

    def _initialize_structures(self, samples):
        """Create initial data structures for aggregation."""
        sample_agg = defaultdict(lambda: {
            "total_sequences": 0,
            "total_bases": 0,
            "total_gc": 0,
            "min_length": float("inf"),
            "max_length": 0,
        })

        overall_stats = {
            "total_samples": len(samples),
            "total_sequences": 0,
            "total_bases": 0,
            "avg_length": None,
            "min_length": None,
            "max_length": None,
            "gc_percent": None,
        }

        return sample_agg, overall_stats

    def _process_results(self, futures, prog, task, sample_agg):
        """Process completed futures and update aggregations."""
        for future in as_completed(futures):
            prog.advance(task)
            sample, result = future.result()
            agg = sample_agg[sample]
            agg["total_sequences"] += result["total_sequences"]
            agg["total_bases"] += result["total_bases"]
            agg["total_gc"] += result["gc_count"]
            agg["min_length"] = min(
                agg["min_length"], result["min_length"]
            )
            agg["max_length"] = max(
                agg["max_length"], result["max_length"]
            )

    def _finalize_stats(self, sample_agg, overall_stats, elapsed):
        """Calculate final statistics and structure results."""
        sample_stats = {}
        for sample, agg in sample_agg.items():
            total_seq = agg["total_sequences"]
            stats = {
                "total_sequences": total_seq,
                "total_bases": agg["total_bases"],
                "avg_length": agg["total_bases"] / total_seq if total_seq else None,
                "min_length": agg["min_length"] if total_seq else None,
                "max_length": agg["max_length"] if total_seq else None,
                "gc_percent": (
                    agg["total_gc"] / agg["total_bases"] * 100
                ) if agg["total_bases"] else None,
            }
            sample_stats[sample] = stats
            overall_stats["total_sequences"] += total_seq
            overall_stats["total_bases"] += agg["total_bases"]

        if overall_stats["total_sequences"] > 0:
            overall_stats.update({
                "avg_length": overall_stats["total_bases"] / overall_stats["total_sequences"],
                "gc_percent": (sum(a["total_gc"] for a in sample_agg.values()) / 
                              overall_stats["total_bases"] * 100),
                "min_length": min(a["min_length"] for a in sample_agg.values()),
                "max_length": max(a["max_length"] for a in sample_agg.values()),
            })
            
        # Add timing metrics using progress bar data
        proc_time = {
            "execution_seconds": elapsed,
            "sequences_per_second": overall_stats['total_sequences'] / elapsed if elapsed > 0 else 0
        }

        return {"samples": sample_stats, "overall": overall_stats, "proc_time": proc_time}

    @staticmethod
    def process_file(sample: str, file_path: Union[str, Path]) -> tuple:
        """Process a single FASTQ file and return basic statistics."""
        total_sequences = total_bases = gc_count = 0
        min_length, max_length = float('inf'), 0

        with gzip.open(file_path, 'rt') as f:
            while True:
                if not f.readline():  # Header
                    break
                sequence = f.readline().strip()
                f.readline()  # Separator
                f.readline()  # Quality
                
                seq_len = len(sequence)
                total_sequences += 1
                total_bases += seq_len
                gc_count += sequence.upper().count('G') + sequence.upper().count('C')
                min_length = min(min_length, seq_len)
                max_length = max(max_length, seq_len)

        return (sample, {
            'total_sequences': total_sequences,
            'total_bases': total_bases,
            'gc_count': gc_count,
            'min_length': min_length,
            'max_length': max_length
        })    

# ====================================== FASTQC ====================================== #

class FastQC:
    """Complete FastQC analysis pipeline with integrated execution."""
    def __init__(
        self,
        fastq_paths: Dict[str, List[Path]],
        output_dir: Union[str, Path],
        fastqc_path: Union[str, Path] = DEFAULT_FASTQC_PATH,
        max_workers: int = DEFAULT_MAX_WORKERS,
        clean_intermediates: bool = True,
        encoding: str = 'utf-8'
    ):
        self.fastq_paths = {k: [Path(p) for p in v] for k, v in fastq_paths.items()}
        self.output_dir = Path(output_dir)
        self.fastqc_path = Path(fastqc_path)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)
        self.clean_intermediates = clean_intermediates
        self.encoding = encoding
        self.parsed_data = None
        self.figs = {}

        # Initialize directories
        self.results_dir = self.output_dir / 'fastqc_results'
        self.processed_dir = self.output_dir / 'processed_results'
        self._validate_environment()

    def _validate_environment(self):
        """Verify required dependencies and paths."""
        # Get Conda environment's bin path
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise RuntimeError("Conda environment not activated!")
        
        fastqc_bin = Path(conda_prefix) / "bin" / "fastqc"
        if not fastqc_bin.exists():
            raise FileNotFoundError(
                f"FastQC not found in Conda env: {fastqc_bin}"
            )
        try:
            subprocess.run(
                [str(fastqc_bin), '-v'],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FastQC validation failed: {e.stderr.decode().strip()}"
            )

    def run_pipeline(self) -> 'FastQCPipeline':
        """Execute complete analysis pipeline."""
        self._run_fastqc_analyses()
        self._parse_results()
        if self.clean_intermediates:
            self._cleanup_intermediates()
        self._organize_results()
        return self

    def _run_fastqc_analyses(self):
        """Execute FastQC analyses with parallel processing."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Get Conda environment's bin path
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise RuntimeError("Conda environment not activated!")
        
        fastqc_bin = Path(conda_prefix) / "bin" / "fastqc"
        if not fastqc_bin.exists():
            raise FileNotFoundError(
                f"FastQC not found in Conda env: {fastqc_bin}"
            )
        def process_file(sample: str, fastq_path: Path):
            
            cmd = [
                #"conda", "run", "-n", "workflow_16s", 
                #str(self.fastqc_path),
                str(fastqc_bin),
                '-o', str(self.results_dir),
                '-f', 'fastq',
                '-q',
                str(fastq_path)
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding=self.encoding
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed processing {sample}: {fastq_path.name}\n"
                    f"Error: {e.stderr.strip() or e.stdout.strip()}"
                )
                return False

        tasks = [
            (sample, path)
            for sample, paths in self.fastq_paths.items()
            for path in paths
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    process_file, sample, path
                ): (sample, path.name)
                for sample, path in tasks
            }
            with get_progress_bar() as prog:
                desc = "Running FastQC..."
                task = prog.add_task(
                    f"[white]{desc:<{DEFAULT_N}}",
                    total=len(futures)
                )
                
                for future in as_completed(futures):
                    sample, fname = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            f"Critical error processing {sample}/{fname}: {str(e)}"
                        )
                    prog.update(task, advance=1)

    def _parse_sample_name(self, zip_path: Path) -> Tuple[str, str]:
        """Parse sample name and direction from filename."""
        pattern = r'^(.*?)[_-](?:R?)([12])'
        match = re.search(pattern, zip_path.stem)
        return (match.groups() if match else (zip_path.stem, 'unknown'))

    def _parse_fastqc_data(
        self, fastqc_data_path: Union[str, Path, zipfile.ZipExtFile]
    ) -> Dict:
        """Parse FastQC data file with comprehensive section handling."""
        sections = {
            "basic_stats": {
                "start": ">>Basic Statistics",
                "data": [],
                "columns": ["measure", "value"]
            },
            "quality_scores": {
                "start": ">>Per base sequence quality", 
                "data": [], 
                "columns": ["base", "mean_quality"]
            },
            "adapter_content": {
                "start": ">>Adapter Content", 
                "data": [], 
                "columns": ["position", "adapter_percent"]
            },
            "length_distribution": {
                "start": ">>Sequence Length Distribution", 
                "data": [], 
                "columns": ["length", "count"]
            },
            "sequence_content": {
                "start": ">>Per base sequence content", 
                "data": [], 
                "columns": ["position", "G", "A", "T", "C"]
            },
            "overrepresented_seqs": {
                "start": ">>Overrepresented sequences", 
                "data": [], 
                "columns": ["seq", "count", "percentage", "source"]
            },
            "per_seq_quality_scores": {
                "start": ">>Per sequence quality scores", 
                "data": [], 
                "columns": ["score", "count"]
            },
            "per_base_gc_content": {
                "start": ">>Per base GC content", 
                "data": [], 
                "columns": ["base", "gc_percent"]
            },
            "per_seq_gc_content": {
                "start": ">>Per sequence GC content", 
                "data": [], 
                "columns": ["gc_percent", "count"]
            },
            "per_base_n_content": {
                "start": ">>Per base N content", 
                "data": [], 
                "columns": ["base", "n_percent"]
            },
            "duplication_levels": {
                "start": ">>Sequence Duplication Levels", 
                "data": [], 
                "columns": ["duplication_level", "percent"]
            },
            "kmer_content": {
                "start": ">>Kmer Content", 
                "data": [], 
                "columns": ["kmer", "count", "percent", "source"]
            },
        }
        
        current_section = None
        lines = []

        if isinstance(fastqc_data_path, (str, Path)):
            try:
                with open(fastqc_data_path, 'r', encoding=self.encoding) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(fastqc_data_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
        else:
            with TextIOWrapper(fastqc_data_path, encoding=self.encoding) as text_file:
                lines = text_file.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith(">>"):
                if line.startswith(">>END_MODULE"):
                    current_section = None
                else:
                    current_section = next(
                        (k 
                         for k, v in sections.items() if line.startswith(v["start"])),
                        None
                    )
                continue

            if current_section and not line.startswith('#'):
                section = sections[current_section]
                parts = line.split('\t')

                if current_section == "basic_stats":
                    if len(parts) == 2:
                        sections[current_section]["data"].append({
                            "measure": parts[0],
                            "value": parts[1]
                        })
                elif len(parts) >= len(section["columns"]):
                    parsed = {
                        col: parts[i] for i, col in enumerate(section["columns"])
                    }
                    section["data"].append(parsed)

        return {
            k: pd.DataFrame(v["data"]) for k, v in sections.items() if v["data"]
        }

    def _parse_results(self):
        """Parse and consolidate all FastQC results."""
        parsed_data = defaultdict(list)
        
        for zip_path in self.results_dir.glob("*.zip"):
            try:
                sample, direction = self._parse_sample_name(zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    data_file = next(
                        f for f in zf.namelist() if f.endswith('fastqc_data.txt')
                    )
                    with zf.open(data_file) as binary_file:
                        result = self._parse_fastqc_data(binary_file)
                        for section, df in result.items():
                            df['sample'] = sample
                            df['direction'] = direction
                            parsed_data[section].append(df)
            except Exception as e:
                logger.error(f"Failed parsing {zip_path.name}: {str(e)}")

        self.parsed_data = {
            section: pd.concat(dfs).reset_index(drop=True)
            for section, dfs in parsed_data.items()
        }

    def _cleanup_intermediates(self):
        """Clean up intermediate analysis files."""
        for pattern in ('*.zip', '*.html'):
            for f in self.results_dir.glob(pattern):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass 
        try:
            self.results_dir.rmdir()
        except OSError:
            logger.warning(f"Directory not empty: {self.results_dir}")

    def _organize_results(self):
        """Organize and save final results with plots."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for section, df in self.parsed_data.items():
            try:
                output_path = self.processed_dir / f"{section}.tsv"
                df.to_csv(output_path, sep='\t', index=False)
            except Exception as e:
                logger.error(f"Failed saving {section} data: {str(e)}")

        self._generate_plots()

    def _generate_plots(self):
        """Generate analysis visualizations."""
        try:
            plots = FastQCPlots(self.parsed_data)
            plot_dir = self.processed_dir / "plots"
            plots.export_figures(export_dir=plot_dir, dpi=300)
            self.figs.update(plots.figs)
        except ImportError:
            logger.warning("Plotting dependencies not available")
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")

    def get_results(self) -> Dict[str, pd.DataFrame]:
        """Access parsed results dataframes."""
        return self.parsed_data or {}

    def get_figures(self) -> Dict[str, plt.Figure]:
        """Access generated matplotlib figures."""
        return self.figs or {}
