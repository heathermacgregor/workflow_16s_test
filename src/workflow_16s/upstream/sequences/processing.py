# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-Party Imports
import pandas as pd

# Local Imports
from workflow_16s.api.ena.sequences import PooledSamplesProcessor, SequenceFetcher
from workflow_16s.config_schema import AppConfig
from workflow_16s.upstream.sequences.utils import BasicStats, CutAdapt, FastQC, SeqKit
from workflow_16s.utils.dir_utils import SubSet, RawData

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def process_sequences(
    config: AppConfig, subset: Dict[str, Any], subset_dirs: SubSet, info: Any
) -> Tuple[Dict[str, List[str]] | List[Path], pd.DataFrame]:
    run_fastqc = config.sequences.quality_control.fastqc.enabled
    run_seqkit = config.sequences.quality_control.seqkit.enabled
    run_cutadapt = config.sequences.trim.cutadapt.enabled
    
    dataset_type = info.get('dataset_type', '').upper()

    if dataset_type == 'ENA':
        fetcher = SequenceFetcher(
            RawData(subset_dirs).raw_seqs, 10, 5, config.sequences.ena.max_concurrent
        )
    
        if not subset["sample_pooling"]:
            run_acc_meta = subset["metadata"].set_index("run_accession", drop=False)
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(run_acc_meta)
        else:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(subset["ena_runs"])
            processor = PooledSamplesProcessor(
                subset["metadata"], RawData(subset_dirs).raw_seqs / 'sorted'
            )
            processor.process_all(RawData(subset_dirs).raw_seqs)
            raw_seqs_paths = processor.sample_file_map    
    else:
        raise ValueError(f"Dataset type '{dataset_type}' not recognized. Expected 'ENA'.")
        
    raw_df = pd.DataFrame()
    if run_cutadapt: 
        seq_analyzer = BasicStats()
        # Ensure values are lists of Path objects for type compatibility
        stats_input = {
            k: [Path(p) for p in v] 
            for k, v in raw_seqs_paths.items()
        } if isinstance(raw_seqs_paths, dict) else raw_seqs_paths
        raw_stats = seq_analyzer.calculate_statistics(stats_input)  # type: ignore
        raw_df = pd.DataFrame([
            {"Metric": k, "Raw": v} 
            for k, v in raw_stats["overall"].items()
        ])

    if run_fastqc:
        fastqc_input = {
            k: [Path(p) for p in v] 
            for k, v in raw_seqs_paths.items()
        } if isinstance(raw_seqs_paths, dict) else raw_seqs_paths
        FastQC(fastqc_input, RawData(subset_dirs).raw_seqs).run_pipeline()

    if run_seqkit:
        seqkit_input = {k: [Path(p) for p in v] 
                        for k, v in raw_seqs_paths.items()} if isinstance(raw_seqs_paths, dict) else raw_seqs_paths
        raw_stats_seqkit = SeqKit(config.sequences.quality_control.seqkit.max_workers).analyze_samples(seqkit_input)  # type: ignore
        stats = raw_stats_seqkit["overall"]
        N = 20
        report = (
            f"\n=== Summary ===\n"
            f"{'Total Samples'.ljust(N)}: {stats['total_samples']}\n"
            f"{'Total Files'.ljust(N)}: {stats['total_files']}\n"
            f"{'Total Sequences'.ljust(N)}: {stats['total_sequences']:,}\n"
            f"{'Total Bases'.ljust(N)}: {stats['total_bases']:,}\n\n"
                "=== Length Distribution ===\n"
            f"{'Average Length'.ljust(N)}: {stats['avg_length']:.2f}\n"
            f"{'Minimum Length'.ljust(N)}: {stats['min_length']}\n"
            f"{'Maximum Length'.ljust(N)}: {stats['max_length']}\n\n"
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
        cutadapt_input = raw_seqs_paths
        cutadapt_config = config.sequences.trim.cutadapt
        trimmed_seqs_paths, *_ = CutAdapt(
            RawData(subset_dirs).raw_seqs, RawData(subset_dirs).trimmed_seqs,
            subset["pcr_primer_fwd_seq"], subset["pcr_primer_rev_seq"],
            cutadapt_config.start_trim, cutadapt_config.end_trim,
            cutadapt_config.start_q_cutoff, cutadapt_config.end_q_cutoff,
            cutadapt_config.min_seq_length, cutadapt_config.n_cores, True,
            subset["target_subfragment"]
        ).run(cutadapt_input)
        processed_paths = trimmed_seqs_paths
        seq_analyzer = BasicStats()
        stats_input = {k: [Path(p) for p in v] for k, v in trimmed_seqs_paths.items()}
        trimmed_stats = seq_analyzer.calculate_statistics(stats_input)  # type: ignore
        trimmed_df = pd.DataFrame([
            {"Metric": k, "Trimmed": v} 
            for k, v in trimmed_stats["overall"].items()
        ])
        stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
        stats_df["Percent Change"] = ((stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]) * 100
        stats_df[["Raw", "Trimmed", "Percent Change"]] = stats_df[["Raw", "Trimmed", "Percent Change"]].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        stats_df = stats_df.dropna(axis=1, how="all")

    if run_cutadapt and run_fastqc:
        fastqc_input = {k: [Path(p) for p in v] 
                        for k, v in processed_paths.items()} if isinstance(processed_paths, dict) else processed_paths
        FastQC(fastqc_input, RawData(subset_dirs).trimmed_seqs).run_pipeline()

    if run_cutadapt and run_seqkit:
        seqkit_input = {k: [Path(p) for p in v] 
                        for k, v in processed_paths.items()} if isinstance(processed_paths, dict) else processed_paths
        SeqKit(config.sequences.quality_control.seqkit.max_workers).analyze_samples(seqkit_input)  # type: ignore

    return processed_paths, stats_df