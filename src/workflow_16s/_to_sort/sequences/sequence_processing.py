# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-Party Imports
import pandas as pd

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import ena
from workflow_16s.sequences.utils import BasicStats, CutAdapt, FastQC, SeqKit

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N = 20
DEFAULT_MAX_WORKERS_ENA = 16
DEFAULT_MAX_WORKERS_SEQKIT = 8

# ==================================== FUNCTIONS ===================================== #

def process_sequences(
    cfg: Dict[str, Any],
    subset: Dict[str, Any],
    subset_dirs: Dict[str, Path],
    info: Any,
) -> Tuple[List[Path], pd.DataFrame]:
    run_fastqc = cfg.get("fastqc", {}).get("enabled", False)
    run_seqkit = cfg.get("seqkit", {}).get("enabled", False)
    run_cutadapt = cfg.get("cutadapt", {}).get("enabled", False)
    dataset_type = info.get('dataset_type', '').upper()

    if dataset_type == 'ENA':
        fetcher = ena.api.SequenceFetcher(
            fastq_dir=subset_dirs["raw_seqs"], 
            max_workers=cfg.get("max_workers", DEFAULT_MAX_WORKERS_ENA)
        )
    
        if not subset["sample_pooling"]:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(
                subset["metadata"].set_index("run_accession", drop=False)
            )
        else:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(
                subset["ena_runs"]
            )
            processor = ena.api.PooledSamplesProcessor(
                metadata_df=subset["metadata"],
                output_dir=subset_dirs["raw_seqs"] / 'sorted'
            )
            processor.process_all(subset_dirs["raw_seqs"])
            raw_seqs_paths = processor.sample_file_map    
    else:
        raise ValueError(f"Dataset type '{dataset_type}' not recognized. Expected 'ENA'.")
        
    if run_cutadapt:    
        seq_analyzer = BasicStats()
        raw_stats = seq_analyzer.calculate_statistics(raw_seqs_paths)
        raw_df = pd.DataFrame(
            [{"Metric": k, "Raw": v} for k, v in raw_stats["overall"].items()]
        )

    if run_fastqc:
        FastQC(
            fastq_paths=raw_seqs_paths, output_dir=subset_dirs["raw_seqs"]
        ).run_pipeline()

    if run_seqkit:
        raw_stats_seqkit = SeqKit(
            max_workers=DEFAULT_MAX_WORKERS_SEQKIT
        ).analyze_samples(raw_seqs_paths)
        stats = raw_stats_seqkit["overall"]
        report = (
            f"\n=== Summary ===\n"
            f"{'Total Samples'.ljust(DEFAULT_N)}: {stats['total_samples']}\n"
            f"{'Total Files'.ljust(DEFAULT_N)}: {stats['total_files']}\n"
            f"{'Total Sequences'.ljust(DEFAULT_N)}: {stats['total_sequences']:,}\n"
            f"{'Total Bases'.ljust(DEFAULT_N)}: {stats['total_bases']:,}\n\n"
            "=== Length Distribution ===\n"
            f"{'Average Length'.ljust(DEFAULT_N)}: {stats['avg_length']:.2f}\n"
            f"{'Minimum Length'.ljust(DEFAULT_N)}: {stats['min_length']}\n"
            f"{'Maximum Length'.ljust(DEFAULT_N)}: {stats['max_length']}\n\n"
            "=== Most Common Lengths ===\n"
            + "".join(
                f"{rank:>2}. {length:3} bp - {count:>9,} sequences\n"
                for rank, (length, count) in enumerate(
                    stats["most_common_lengths"], start=1
                )
            )
        )
        logger.info(report)
        
    processed_paths = raw_seqs_paths
    stats_df = pd.DataFrame()
    
    if run_cutadapt:
        trimmed_seqs_paths, *_ = CutAdapt(
            fastq_dir=subset_dirs["raw_seqs"],
            trimmed_fastq_dir=subset_dirs["trimmed_seqs"],
            primer_fwd=subset["pcr_primer_fwd_seq"],
            primer_rev=subset["pcr_primer_rev_seq"],
            start_trim=cfg["cutadapt"]["start_trim"],
            end_trim=cfg["cutadapt"]["end_trim"],
            start_q_cutoff=cfg["cutadapt"]["start_q_cutoff"],
            end_q_cutoff=cfg["cutadapt"]["end_q_cutoff"],
            min_seq_length=cfg["cutadapt"]["min_seq_length"],
            cores=cfg["cutadapt"]["n_cores"],
            rerun=True,
            region=subset["target_subfragment"],
        ).run(fastq_paths=raw_seqs_paths)
        processed_paths = trimmed_seqs_paths
        trimmed_stats = seq_analyzer.calculate_statistics(trimmed_seqs_paths)
        trimmed_df = pd.DataFrame(
            [{"Metric": k, "Trimmed": v} for k, v in trimmed_stats["overall"].items()]
        )
        stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
        stats_df["Percent Change"] = (
            (stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]
        ) * 100
        stats_df[["Raw", "Trimmed", "Percent Change"]] = stats_df[
            ["Raw", "Trimmed", "Percent Change"]
        ].applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        stats_df = stats_df.dropna(axis=1, how="all")

    if run_cutadapt and run_fastqc:
        FastQC(
            fastq_paths=processed_paths, 
            output_dir=subset_dirs["trimmed_seqs"]
        ).run_pipeline()

    if run_cutadapt and run_seqkit:
        SeqKit(max_workers=DEFAULT_MAX_WORKERS_SEQKIT).analyze_samples(processed_paths)

    return processed_paths, stats_df
  
