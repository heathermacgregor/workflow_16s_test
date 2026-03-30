# workflow_16s/upstream/sequences/processing.py

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from workflow_16s.api.ena.sequences import PooledSamplesProcessor, SequenceFetcher
from workflow_16s.config import AppConfig
from workflow_16s.utils.dir_utils import SubSet, RawData
from workflow_16s.utils.logger import get_logger

from workflow_16s.upstream.sequences.cutadapt import CutAdaptWrapper
from workflow_16s.upstream.sequences.fastqc import FastQCWrapper
from workflow_16s.upstream.sequences.seqkit import SeqKitWrapper


def process_sequences(
    config: AppConfig, subset: Dict[str, Any], subset_dirs: SubSet, info: Any
) -> Tuple[Dict[str, List[Path]], pd.DataFrame]:
    logger = get_logger("workflow_16s")
    run_fastqc = config.sequences.quality_control.fastqc.enabled
    run_seqkit = config.sequences.quality_control.seqkit.enabled
    run_cutadapt = config.sequences.trim.cutadapt.enabled
    
    dataset_type = info.get('dataset_type', '').upper()

    # --- 1. FETCH RAW DATA ---
    if dataset_type == 'ENA':
        fetcher = SequenceFetcher(
            fastq_dir=RawData(subset_dirs).raw_seqs, 
            retries=10, 
            initial_delay=5, 
            max_workers=config.sequences.ena.max_concurrent,
            progress_obj=None
        )
    
        if not subset["sample_pooling"]:
            run_acc_meta = subset["metadata"].set_index("run_accession", drop=False)
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(run_acc_meta)
        else:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(subset["ena_runs"])
            processor = PooledSamplesProcessor(
                metadata_df=subset["metadata"], 
                output_dir=RawData(subset_dirs).raw_seqs / 'sorted',
                progress_obj=None
            )
            processor.process_all(RawData(subset_dirs).raw_seqs)
            raw_seqs_paths = processor.sample_file_map    
    else:
        raise ValueError(f"Dataset type '{dataset_type}' not recognized. Expected 'ENA'.")
        
    # Standardize paths to Path objects
    raw_seqs_paths = {k: [Path(p) for p in v] for k, v in raw_seqs_paths.items()}
    raw_df = pd.DataFrame()

    # --- 2. RAW DATA QC (BasicStats replaced by SeqKit) ---
    if run_seqkit:
        # Run SeqKit ONCE to get all basic stats and length distributions (Fixes Double I/O)
        seqkit_wrapper = SeqKitWrapper(max_workers=config.sequences.quality_control.seqkit.max_workers)
        raw_seqkit_df = seqkit_wrapper.analyze(raw_seqs_paths)
        
        # Extract the OVERALL row to populate the legacy stats dictionary
        if not raw_seqkit_df.empty and 'OVERALL' in raw_seqkit_df['sample'].values:
            overall = raw_seqkit_df[raw_seqkit_df['sample'] == 'OVERALL'].iloc[0]
            raw_df = pd.DataFrame([
                {"Metric": "Total Sequences", "Raw": overall.get("num_seqs", 0)},
                {"Metric": "Total Bases", "Raw": overall.get("sum_len", 0)},
                {"Metric": "Average Length", "Raw": overall.get("avg_len", 0)},
                {"Metric": "Min Length", "Raw": overall.get("min_len", 0)},
                {"Metric": "Max Length", "Raw": overall.get("max_len", 0)}
            ])
            logger.info(f"\n=== Raw Data Summary ===\n"
                        f"Total Sequences: {overall.get('num_seqs', 0):,}\n"
                        f"Total Bases:     {overall.get('sum_len', 0):,}\n"
                        f"Average Length:  {overall.get('avg_len', 0):.2f} bp")

    if run_fastqc:
        FastQCWrapper(max_workers=config.sequences.quality_control.fastqc.max_workers).run_and_parse(
            sample_files=raw_seqs_paths, 
            output_dir=RawData(subset_dirs).raw_seqs / 'fastqc_results'
        )

    # --- 3. CUTADAPT TRIMMING ---
    processed_paths = raw_seqs_paths
    stats_df = pd.DataFrame()
    
    if run_cutadapt:
        cutadapt_config = config.sequences.trim.cutadapt
        
        # Initialize CutAdapt (Fixes Nested Parallelism by setting cores_per_job=1)
        wrapper = CutAdaptWrapper(
            fwd_primer=subset["pcr_primer_fwd_seq"],
            rev_primer=subset["pcr_primer_rev_seq"],
            min_length=cutadapt_config.min_seq_length,
            quality_cutoff=cutadapt_config.end_q_cutoff, 
            cores_per_job=1  # ⚠️ CRITICAL: Let ThreadPoolExecutor handle concurrency!
        )
        
        trimmed_seqs_paths, trim_summary_df = wrapper.trim(
            sample_files=raw_seqs_paths,
            output_dir=RawData(subset_dirs).trimmed_seqs,
            max_workers=cutadapt_config.n_cores  # Parallelize across files here
        )
        processed_paths = trimmed_seqs_paths

        # --- 4. TRIMMED DATA QC ---
        if run_seqkit:
            # We run SeqKit again on the trimmed data to populate the comparison DataFrame
            trimmed_seqkit_df = seqkit_wrapper.analyze(processed_paths)
            
            if not trimmed_seqkit_df.empty and 'OVERALL' in trimmed_seqkit_df['sample'].values:
                overall_trim = trimmed_seqkit_df[trimmed_seqkit_df['sample'] == 'OVERALL'].iloc[0]
                trimmed_df = pd.DataFrame([
                    {"Metric": "Total Sequences", "Trimmed": overall_trim.get("num_seqs", 0)},
                    {"Metric": "Total Bases", "Trimmed": overall_trim.get("sum_len", 0)},
                    {"Metric": "Average Length", "Trimmed": overall_trim.get("avg_len", 0)},
                    {"Metric": "Min Length", "Trimmed": overall_trim.get("min_len", 0)},
                    {"Metric": "Max Length", "Trimmed": overall_trim.get("max_len", 0)}
                ])
                
                # Merge Raw and Trimmed stats
                if not raw_df.empty:
                    stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
                    stats_df["Percent Change"] = ((stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]) * 100
                    stats_df[["Raw", "Trimmed", "Percent Change"]] = stats_df[["Raw", "Trimmed", "Percent Change"]].map(
                        lambda x: f"{x:.2f}" if isinstance(x, float) else x
                    )

        if run_fastqc:
            FastQCWrapper(max_workers=config.sequences.quality_control.fastqc.max_workers).run_and_parse(
                sample_files=processed_paths, 
                output_dir=RawData(subset_dirs).trimmed_seqs / 'fastqc_results'
            )

    return processed_paths, stats_df