# workflow_16s/upstream/sequences/cutadapt.py

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
        logger = get_logger("workflow_16s")
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
    
