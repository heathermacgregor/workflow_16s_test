#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QIIME 2 Self-Contained Amplicon Analysis Workflow

This script provides a multi-mode, command-line-driven pipeline for processing
16S rRNA gene amplicon sequencing data using QIIME 2.

MODES:
1. inspect:   Run with `--dada2-mode inspect` to generate quality plots and
              parameter estimates for DADA2, then stop.
2. auto:      Run with `--dada2-mode auto` to automatically estimate and use
              the optimal DADA2 parameters in a single, non-interactive run.
3. manual:    Run with `--dada2-mode manual` and provide `--dada2-params`
              to execute the full pipeline with user-specified parameters.
"""

# ===================================== IMPORTS ====================================== #

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from Bio.Seq import Seq
from qiime2 import Artifact, Metadata, Visualization # type: ignore
from qiime2.plugins import demux, taxa # type: ignore
from qiime2.plugins.cutadapt.methods import trim_paired, trim_single # type: ignore
from qiime2.plugins.dada2.methods import denoise_paired, denoise_single # type: ignore
from qiime2.plugins.feature_classifier.methods import classify_sklearn # type: ignore
from qiime2.plugins.phylogeny.pipelines import align_to_tree_mafft_fasttree # type: ignore

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==================================== CONSTANTS ===================================== #

DEFAULT_N_THREADS = max(1, (os.cpu_count() or 2) // 2)
DEFAULT_MIN_FREQUENCY = 1000
DEFAULT_COLLAPSE_LEVEL = 6  # Corresponds to Genus

# ================================ WORKFLOW CLASS ================================== #

class QIIMEWorkflow:
    """Manages and executes the QIIME 2 analysis pipeline."""

    def __init__(self, **kwargs):
        """Initializes the workflow from parsed command-line arguments."""
        self.params = kwargs
        self.qiime_dir = self.params['qiime_dir']
        self._setup_file_registry()

    def run(self) -> None:
        """Determines the run mode and starts the corresponding workflow."""
        self._check_inputs()
        self.qiime_dir.mkdir(parents=True, exist_ok=True)

        if self.params.get("hard_rerun", False):
            self._clean_qiime_dir(artifacts_only=True)
        
        mode = self.params['dada2_mode']
        if mode == 'inspect':
            self._run_inspection_workflow()
        else: # 'auto' or 'manual'
            self._run_execution_workflow(mode)

    def _run_inspection_workflow(self) -> None:
        """Runs the initial steps to generate DADA2 parameter inspection files."""
        logging.info("[INSPECT MODE] Starting DADA2 parameter inspection.")
        
        # Correct Order: Import -> Trim -> Filter -> Summarize -> Inspect
        seqs = self._import_sequences()
        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)
        
        filtered_seqs = self._filter_sequences(seqs)
        self._summarize_sequences(filtered_seqs, "03_filtered-summary")
        self._generate_inspection_files("03_filtered-summary")
        
        logging.info("[INSPECT MODE] Inspection complete. The workflow will now stop.")

    def _run_execution_workflow(self, mode: str) -> None:
        """Executes the full, non-interactive QIIME 2 pipeline."""
        logging.info(f"[{mode.upper()} MODE] Starting full QIIME 2 processing.")
        
        if self._skip("rep_seqs", "table", "taxonomy", "rooted_tree"):
            logging.info("All final outputs already exist. Skipping full execution.")
            return

        # Correct Order: Import -> Trim -> Filter -> Denoise
        seqs = self._import_sequences()
        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)

        filtered_seqs = self._filter_sequences(seqs)

        if mode == 'auto':
            # Estimate parameters from the filtered data summary
            self._summarize_sequences(filtered_seqs, "03_filtered-summary")
            dada2_params = self._estimate_dada2_params("03_filtered-summary")
            logging.info(f"Automatically determined DADA2 parameters: {dada2_params}")
        else: # manual
            dada2_params = self._get_locked_dada2_params()
            logging.info(f"Using manually specified DADA2 parameters: {dada2_params}")

        rep_seqs, table, _ = self._denoise_sequences(filtered_seqs, **dada2_params)
        
        taxonomy = self._classify_taxonomy(rep_seqs)
        self._build_phylogenetic_tree(rep_seqs)
        self._collapse_table(table, taxonomy)
        
        self._export_final_artifacts(table, taxonomy, rep_seqs, rooted_tree)
        
        logging.info(f"[{mode.upper()} MODE] QIIME 2 processing completed successfully.")

    # --- Core QIIME 2 Step Methods ---
    
    def _import_sequences(self) -> Artifact:
        if self._skip("seqs"): return Artifact.load(self.files['seqs'])
        logging.info("Importing sequences...")
        layout = self.params['library_layout']
        types = {
            "single": ("SampleData[SequencesWithQuality]", "SingleEndFastqManifestPhred33V2"),
            "paired": ("SampleData[PairedEndSequencesWithQuality]", "PairedEndFastqManifestPhred33V2"),
        }
        import_type, view_type = types[layout]
        seqs = Artifact.import_data(import_type, self.params['manifest_tsv'], view_type=view_type)
        seqs.save(str(self.files["seqs"]))
        return seqs

    def _summarize_sequences(self, seqs: Artifact, base_name: str) -> None:
        if self._skip(base_name): return
        logging.info(f"Generating summary visualization for {base_name}...")
        summary_viz = demux.visualizers.summarize(data=seqs).visualization
        summary_viz.save(str(self.files[base_name]))

    def _trim_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("trimmed_seqs"): return Artifact.load(self.files['trimmed_seqs'])
        logging.info("Trimming primer sequences with Cutadapt...")
        rev_primer_rc = str(Seq(self.params['rev_primer_seq']).reverse_complement()) if self.params.get('rev_primer_seq') else ''
        if self.params['library_layout'] == 'paired':
            trimmed = trim_paired(
                demultiplexed_sequences=seqs,
                front_f=[self.params['fwd_primer_seq']],
                front_r=[rev_primer_rc],
                cores=self.params['n_threads'],
            ).trimmed_sequences
        else:
            trimmed = trim_single(
                demultiplexed_sequences=seqs,
                front=[self.params['fwd_primer_seq']],
                cores=self.params['n_threads'],
            ).trimmed_sequences
        trimmed.save(str(self.files["trimmed_seqs"]))
        return trimmed
    
    def _filter_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("filtered_seqs"): return Artifact.load(self.files['filtered_seqs'])
        
        # Use summary of pre-filtered data to determine which samples to drop
        summary_to_use = self._get_summary_key()
        if not self.files[summary_to_use].exists():
            self._summarize_sequences(seqs, summary_to_use)
        
        logging.info(f"Filtering samples with fewer than {self.params['min_frequency']} reads...")
        viz = Visualization.load(self.files[summary_to_use])
        df = viz.view(pd.DataFrame)
        valid_samples = df[df['total-sequences'] >= self.params['min_frequency']].index
        
        metadata = Metadata(pd.DataFrame(index=valid_samples))
        filtered = demux.methods.filter_samples(demux=seqs, metadata=metadata).filtered_demux
        filtered.save(str(self.files["filtered_seqs"]))
        logging.info(f"Retained {len(valid_samples)} of {len(df)} samples after filtering.")
        return filtered

    def _denoise_sequences(self, seqs: Artifact, **dada2_params) -> Tuple[Artifact, Artifact, Artifact]:
        if self._skip("rep_seqs", "table", "stats"):
            return (Artifact.load(self.files["rep_seqs"]), Artifact.load(self.files["table"]), Artifact.load(self.files["stats"]))
        
        logging.info("Denoising sequences with DADA2...")
        p = self.params
        if p['library_layout'] == 'paired':
            results = denoise_paired(demultiplexed_seqs=seqs, chimera_method=p['chimera_method'], n_threads=p['n_threads'], **dada2_params)
        else:
            single_params = {'trunc_len': dada2_params['trunc_len_f'], 'trim_left': dada2_params['trim_left_f']}
            results = denoise_single(demultiplexed_seqs=seqs, chimera_method=p['chimera_method'], n_threads=p['n_threads'], **single_params)
        
        results.representative_sequences.save(str(self.files["rep_seqs"]))
        results.table.save(str(self.files["table"]))
        results.denoising_stats.save(str(self.files["stats"]))
        return results.representative_sequences, results.table, results.denoising_stats

    def _classify_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        if self._skip("taxonomy"): return Artifact.load(self.files["taxonomy"])
        logging.info("Classifying taxonomy...")
        classifier = Artifact.load(self.params['classifier_path'])
        taxonomy = classify_sklearn(reads=rep_seqs, classifier=classifier, confidence=self.params['confidence'], n_jobs=self.params['n_threads']).classification
        taxonomy.save(str(self.files["taxonomy"]))
        return taxonomy

    def _build_phylogenetic_tree(self, rep_seqs: Artifact) -> Artifact:
        if self._skip("rooted_tree"): return Artifact.load(self.files["rooted_tree"])
        logging.info("Building phylogenetic tree...")
        results = align_to_tree_mafft_fasttree(sequences=rep_seqs, n_threads=self.params['n_threads'])
        results.rooted_tree.save(str(self.files["rooted_tree"]))
        return results.rooted_tree

    def _collapse_table(self, table: Artifact, taxonomy: Artifact) -> Artifact:
        if self._skip("collapsed_table"): return Artifact.load(self.files["collapsed_table"])
        level = self.params['collapse_level']
        logging.info(f"Collapsing feature table to level {level}...")
        collapsed = taxa.actions.collapse(table=table, taxonomy=taxonomy, level=level).collapsed_table
        collapsed.save(str(self.files["collapsed_table"]))
        return collapsed

    # --- Helper and Utility Methods ---
    
    def _estimate_dada2_params(self, summary_key: str) -> Dict[str, int]:
        """Calculates and returns the estimated best DADA2 parameters with safety checks."""
        summary_qzv_path = self.files[summary_key]
        with zipfile.ZipFile(summary_qzv_path, 'r') as z:
            viz_root = [i for i in z.infolist() if i.is_dir()][0].filename
            fwd_qual_path = z.extract(f"{viz_root}data/forward-seven-number-summary.csv", path=self.qiime_dir)
            df_fwd = pd.read_csv(fwd_qual_path, index_col=0)
            is_paired = self.params['library_layout'] == 'paired'
            if is_paired:
                rev_qual_path = z.extract(f"{viz_root}data/reverse-seven-number-summary.csv", path=self.qiime_dir)
                df_rev = pd.read_csv(rev_qual_path, index_col=0)

        # Estimate parameters with safety checks
        est_trim_f = len(self.params['fwd_primer_seq']) if self.params.get("trim_sequences") else 0
        
        # Use median position where quality >= 25 (quality score cutoff)
        # If quality drops below 25 everywhere, use the median quality position instead
        good_qual_fwd = df_fwd[df_fwd['50%'] >= 25]
        if not good_qual_fwd.empty:
            # Use the last position with good median quality
            est_trunc_f = int(good_qual_fwd.index.max())
        else:
            # Fallback: use position where median quality drops below 30 (more lenient)
            ok_qual_fwd = df_fwd[df_fwd['50%'] >= 20]
            est_trunc_f = int(ok_qual_fwd.index.max()) if not ok_qual_fwd.empty else 0
        
        # Safety check: ensure truncation length is reasonable
        # If estimated < 75% of max read length, likely problematic data
        max_len_fwd = int(df_fwd.index.max())
        if est_trunc_f < max_len_fwd * 0.5:  # Less than 50% of reads remain
            logging.warning(f"Forward truncation ({est_trunc_f}) is very short compared to max length ({max_len_fwd}). "
                          f"This may indicate low quality sequences. Defaulting to 75% of max length.")
            est_trunc_f = int(max_len_fwd * 0.75)
        
        params = {'trim_left_f': est_trim_f, 'trunc_len_f': est_trunc_f}
        
        if is_paired:
            est_trim_r = len(self.params.get('rev_primer_seq', '')) if self.params.get("trim_sequences") else 0
            good_qual_rev = df_rev[df_rev['50%'] >= 25]
            if not good_qual_rev.empty:
                est_trunc_r = int(good_qual_rev.index.max())
            else:
                ok_qual_rev = df_rev[df_rev['50%'] >= 20]
                est_trunc_r = int(ok_qual_rev.index.max()) if not ok_qual_rev.empty else 0
            
            # Safety check for reverse reads
            max_len_rev = int(df_rev.index.max())
            if est_trunc_r < max_len_rev * 0.5:  # Less than 50% of reads remain
                logging.warning(f"Reverse truncation ({est_trunc_r}) is very short compared to max length ({max_len_rev}). "
                              f"This may indicate low quality sequences. Defaulting to 75% of max length.")
                est_trunc_r = int(max_len_rev * 0.75)
            
            params.update({'trim_left_r': est_trim_r, 'trunc_len_r': est_trunc_r})
        
        return params

    def _generate_inspection_files(self, summary_key: str) -> None:
        """Creates quality plots and TSV files for user inspection."""
        params = self._estimate_dada2_params(summary_key)
        summary_qzv_path = self.files[summary_key]

        with zipfile.ZipFile(summary_qzv_path, 'r') as z:
            viz_root = [i for i in z.infolist() if i.is_dir()][0].filename
            fwd_qual_path = z.extract(f"{viz_root}data/forward-seven-number-summary.csv", path=self.qiime_dir)
            df_fwd = pd.read_csv(fwd_qual_path, index_col=0)
            is_paired = self.params['library_layout'] == 'paired'
            if is_paired:
                rev_qual_path = z.extract(f"{viz_root}data/reverse-seven-number-summary.csv", path=self.qiime_dir)
                df_rev = pd.read_csv(rev_qual_path, index_col=0)

        inspection_dir = self.qiime_dir / "dada2_parameter_inspection"
        inspection_dir.mkdir(exist_ok=True)
        df_fwd.to_csv(inspection_dir / "forward_quality.tsv", sep='\t')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_fwd.index, df_fwd['50%'], label='Forward Median Quality')
        ax.fill_between(df_fwd.index, df_fwd['25%'], df_fwd['75%'], alpha=0.2)
        if is_paired:
            df_rev.to_csv(inspection_dir / "reverse_quality.tsv", sep='\t')
            ax.plot(df_rev.index, df_rev['50%'], label='Reverse Median Quality')
            ax.fill_between(df_rev.index, df_rev['25%'], df_rev['75%'], alpha=0.2)
        
        ax.axhline(y=25, color='r', linestyle='--', label='Q25 Threshold')
        ax.set_xlabel("Position in Read (bp)")
        ax.set_ylabel("Phred Quality Score")
        ax.set_title("Median Sequence Quality (Post-Filtering)")
        ax.legend()
        fig.savefig(inspection_dir / "quality_plot.png")
        plt.close(fig)

        logging.info("\n" + "="*80)
        logging.info("📊 DADA2 INSPECTION REPORT")
        logging.info(f"Inspection files generated in:\n{inspection_dir}")
        logging.info("\n--- Best Estimates (based on filtered data) ---")
        for key, value in params.items():
            logging.info(f"{key}: {value}")
        logging.info("\nUse these estimates to set the --dada2-params argument for a 'manual' run.")
        logging.info("="*80)

    def _get_locked_dada2_params(self) -> Dict[str, int]:
        user_params = self.params['dada2_params']
        is_paired = self.params['library_layout'] == 'paired'
        if is_paired:
            return {'trunc_len_f': user_params[0], 'trunc_len_r': user_params[1],
                    'trim_left_f': user_params[2], 'trim_left_r': user_params[3]}
        else:
            return {'trunc_len_f': user_params[0], 'trim_left_f': user_params[1]}

    def _get_summary_key(self) -> str:
        """Gets the key for the summary file to be used for filtering."""
        return "02_trimmed-summary" if self.params.get("trim_sequences") else "01_imported-summary"

    def _setup_file_registry(self) -> None:
        d = self.qiime_dir
        level = self.params.get('collapse_level', DEFAULT_COLLAPSE_LEVEL)
        self.files: Dict[str, Path] = {
            "seqs": d / "01_demux-sequences.qza",
            "01_imported-summary": d / "01_imported-summary.qzv",
            "trimmed_seqs": d / "02_trimmed-sequences.qza",
            "02_trimmed-summary": d / "02_trimmed-summary.qzv",
            "filtered_seqs": d / "03_filtered-sequences.qza",
            "03_filtered-summary": d / "03_filtered-summary.qzv",
            "rep_seqs": d / "04_representative-sequences.qza",
            "table": d / "04_feature-table.qza",
            "stats": d / "04_denoising-stats.qza",
            "taxonomy": d / "05_taxonomy.qza",
            "rooted_tree": d / "06_rooted-tree.qza",
            "collapsed_table": d / f"07_collapsed-table-L{level}.qza",
        }

    def _skip(self, *keys: str) -> bool:
        if self.params.get("hard_rerun", False): return False
        return all(self.files.get(key, Path()).exists() for key in keys)

    def _check_inputs(self) -> None:
        if not self.params['manifest_tsv'].is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.params['manifest_tsv']}")
        if self.params['dada2_mode'] != 'inspect' and not self.params['classifier_path'].is_file():
            raise FileNotFoundError(f"Classifier file not found: {self.params['classifier_path']}")

    def _clean_qiime_dir(self, artifacts_only: bool = False) -> None:
        if not self.qiime_dir.is_dir(): return
        if artifacts_only:
            logging.info(f"Cleaning QIIME 2 artifacts in {self.qiime_dir}...")
            for ext in ("*.qza", "*.qzv"):
                for f in self.qiime_dir.glob(ext): f.unlink()
        else:
            logging.info(f"Removing entire QIIME 2 directory: {self.qiime_dir}...")
            shutil.rmtree(self.qiime_dir)
            
    def _export_final_artifacts(self, table: Artifact, taxonomy: Artifact, 
                                rep_seqs: Artifact, rooted_tree: Artifact) -> None:
        """Exports final QIIME 2 artifacts to standard file formats."""
        export_dir = self.qiime_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        logging.info(f"Exporting final artifacts to {export_dir}...")

        # Define export paths
        export_paths = {
            "table": export_dir / "feature-table.biom",
            "taxonomy": export_dir / "taxonomy.tsv",
            "rep_seqs": export_dir / "representative-sequences.fasta",
            "rooted_tree": export_dir / "rooted-tree.nwk"
        }

        # Export each artifact
        table.export_data(export_paths["table"])
        taxonomy.export_data(export_paths["taxonomy"])
        rep_seqs.export_data(export_paths["rep_seqs"])
        rooted_tree.export_data(export_paths["rooted_tree"])

        logging.info("Artifact export complete.")

# ============================ ARGUMENT PARSING & MAIN ============================= #

def main() -> None:
    parser = argparse.ArgumentParser(description='QIIME 2 Self-Contained Amplicon Analysis Workflow.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Mode Selection ---
    mode_group = parser.add_argument_group("Workflow Mode")
    mode_group.add_argument("--dada2-mode", required=True, choices=['inspect', 'auto', 'manual'], help="Select the DADA2 parameter handling mode.")
    
    # --- I/O Parameters ---
    io_group = parser.add_argument_group("Input/Output Parameters")
    io_group.add_argument("--manifest_tsv", type=Path, required=True, help="Path to the input manifest TSV file.")
    io_group.add_argument("--qiime_dir", type=Path, required=True, help="Path to the output directory for QIIME 2 results.")
    io_group.add_argument("--classifier_path", type=Path, help="Full path to the QIIME 2 classifier artifact (.qza). Required for 'auto' and 'manual' modes.")

    # --- Sequencing Parameters ---
    seq_group = parser.add_argument_group("Sequencing Parameters")
    seq_group.add_argument("--library_layout", choices=['single', 'paired'], required=True, help="Sequencing library layout.")
    seq_group.add_argument("--fwd_primer_seq", type=str, required=True, help="Forward primer sequence (5' to 3').")
    seq_group.add_argument("--rev_primer_seq", type=str, help="Reverse primer sequence (5' to 3'). Required for paired-end.")
    
    # --- Denoising Parameters ---
    denoise_group = parser.add_argument_group("Denoising Parameters (DADA2)")
    denoise_group.add_argument("--dada2-params", type=int, nargs='+', help="[MANUAL MODE] Lock in DADA2 parameters. Order: trunc_f trunc_r trim_f trim_r (use 2 for single-end).")
    denoise_group.add_argument("--chimera_method", default="consensus", choices=["none", "consensus", "pooled"], help="Method for chimera detection.")
    
    # --- Processing Parameters ---
    proc_group = parser.add_argument_group("Processing Parameters")
    proc_group.add_argument("--n_threads", type=int, default=DEFAULT_N_THREADS, help="Number of CPU threads to use.")
    proc_group.add_argument("--min_frequency", type=int, default=DEFAULT_MIN_FREQUENCY, help="Minimum read count to retain a sample.")
    proc_group.add_argument("--collapse_level", type=int, default=DEFAULT_COLLAPSE_LEVEL, help="Taxonomic level to collapse table (1-7).")
    proc_group.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold for taxonomy assignment.")
    proc_group.add_argument("--hard_rerun", action="store_true", help="Force reprocessing of all steps.")
    proc_group.add_argument("--trim_sequences", action="store_true", help="Enable primer trimming with Cutadapt.")
    
    args = parser.parse_args()
    
    # --- Argument Validation ---
    if args.dada2_mode == 'manual' and not args.dada2_params:
        parser.error("--dada2-mode 'manual' requires the --dada2-params argument.")
    if args.dada2_mode != 'inspect' and not args.classifier_path:
        parser.error("--dada2-mode 'auto' or 'manual' requires the --classifier_path argument.")
    if args.library_layout == 'paired' and not args.rev_primer_seq:
        parser.error("--library_layout 'paired' requires the --rev_primer_seq argument.")

    try:
        workflow = QIIMEWorkflow(**vars(args))
        workflow.run()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Workflow failed: {e}", exc_info=False)
        sys.exit(1)

if __name__ == "__main__":
    main()