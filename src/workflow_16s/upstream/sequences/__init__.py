# workflow_16s/upstream/sequences/__init__.py

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
from .tools import (
    SeqKitWrapper, CutAdaptWrapper, FastQCWrapper, FastQCPlotter,
    DEFAULT_REGIONS, DEFAULT_PRIMER_REGIONS, DEFAULT_16S_PRIMERS
)
from .primers import (
    build_primer_database_direct, get_primer_id_from_search, 
    get_primer_details, process_and_save_primer_data, 
    create_and_populate_db, query_primers, 
    import_and_save_database, query_primer_pairs, main
)

__all__ = [
    "SeqKitWrapper", "CutAdaptWrapper", "FastQCWrapper", "FastQCPlotter",
    "DEFAULT_REGIONS", "DEFAULT_PRIMER_REGIONS", "DEFAULT_16S_PRIMERS",
    "build_primer_database_direct", "get_primer_id_from_search", 
    "get_primer_details", "process_and_save_primer_data", 
    "create_and_populate_db", "query_primers", 
    "import_and_save_database", "query_primer_pairs", "main"
]

def get_progress_bar():
    """Returns a pre-configured Rich progress bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True
    )

@with_logger
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
    logger = get_logger("workflow_16s")
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