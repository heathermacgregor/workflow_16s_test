#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QIIME 2 Self-Contained Amplicon Analysis Workflow

This script provides a comprehensive, command-line-driven pipeline for processing
16S rRNA gene amplicon sequencing data using QIIME 2. It is a single,
self-contained file that automates all steps from data import to analysis.

The workflow includes:
1.  Importing raw sequence data from a manifest file.
2.  Trimming primer sequences using Cutadapt.
3.  Filtering samples with low read counts.
4.  Denoising sequences into Amplicon Sequence Variants (ASVs).
5.  Assigning taxonomy to ASVs using a pre-trained classifier.
6.  Generating a phylogenetic tree for diversity analyses.
7.  Collapsing the feature table to a specified taxonomic level.
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

from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt 
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
        """Initializes the workflow directly from parsed command-line arguments."""
        self.params = kwargs
        self.qiime_dir = self.params['qiime_dir']
        self._setup_file_registry()

    def run(self) -> None:
        """Execute the full QIIME 2 pipeline."""
        self._check_inputs()
        self.qiime_dir.mkdir(parents=True, exist_ok=True)

        if self.params.get("hard_rerun", False):
            self._clean_qiime_dir(artifacts_only=True)

        final_outputs = ["rep_seqs", "table", "taxonomy", "rooted_tree", "collapsed_table"]
        if self._skip(*final_outputs):
            print("All final outputs already exist. Skipping.")
            return

        print(f"Starting QIIME 2 processing in: {self.qiime_dir}")
        self._run_workflow()
        print("QIIME 2 processing completed successfully.")

    def _run_workflow(self) -> None:
        """Orchestrates the sequence of QIIME 2 processing steps."""
        seqs = self._import_sequences()
        self._summarize_sequences(seqs, "01_imported-summary")

        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)
            self._summarize_sequences(seqs, "02_trimmed-summary")
        else:
            print("Sequence trimming is disabled.")
            # If not trimming, the next step's input is the original sequences
            self.files["trimmed_seqs"] = self.files["seqs"] 

        filtered_seqs = self._filter_sequences(seqs)
        dada2_params = self._determine_dada2_params(filtered_seqs)
        rep_seqs, table, _ = self._denoise_sequences(filtered_seqs, **dada2_params)
        taxonomy = self._classify_taxonomy(rep_seqs)
        self._build_phylogenetic_tree(rep_seqs)
        self._collapse_table(table, taxonomy)

    def _import_sequences(self) -> Artifact:
        """Imports sequences from a manifest file."""
        if self._skip("seqs"):
            print(f"Skipping import: '{self.files['seqs']}' exists.")
            return Artifact.load(self.files['seqs'])
        
        print("Importing sequences...")
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
        """Generates a visualization of sequence quality statistics."""
        if self._skip(base_name):
            print(f"Skipping summary: '{self.files[base_name]} ' exists.")
            return
        
        print(f"Generating summary visualization for {base_name}...")
        summary_viz = demux.visualizers.summarize(data=seqs).visualization
        summary_viz.save(str(self.files[base_name]))

    def _trim_sequences(self, seqs: Artifact) -> Artifact:
        """Trims adapter/primer sequences using cutadapt."""
        if self._skip("trimmed_seqs"):
            print(f"Skipping trimming: '{self.files['trimmed_seqs']}' exists.")
            return Artifact.load(self.files['trimmed_seqs'])
        
        print("Trimming primer sequences with Cutadapt...")
        rev_primer_rc = str(Seq(self.params['rev_primer_seq']).reverse_complement())
        
        if self.params['library_layout'] == 'paired':
            trimmed = trim_paired(
                demultiplexed_sequences=seqs,
                front_f=[self.params['fwd_primer_seq']],
                front_r=[rev_primer_rc],
                cores=self.params['n_threads'],
            ).trimmed_sequences
        else: # single
            trimmed = trim_single(
                demultiplexed_sequences=seqs,
                front=[self.params['fwd_primer_seq']],
                cores=self.params['n_threads'],
            ).trimmed_sequences

        trimmed.save(str(self.files["trimmed_seqs"]))
        return trimmed
    
    def _filter_sequences(self, seqs: Artifact) -> Artifact:
        """Filters samples based on minimum frequency, using the latest summary."""
        if self._skip("filtered_seqs"):
            print(f"Skipping filtering: '{self.files['filtered_seqs']}' exists.")
            return Artifact.load(self.files['filtered_seqs'])

        # Determine which summary to use for filtering counts
        summary_to_use = "02_trimmed-summary" if self.params.get("trim_sequences") else "01_imported-summary"
        counts_path = self.qiime_dir / summary_to_use
        
        print(f"Filtering samples with fewer than {self.params['min_frequency']} reads...")
        
        # We need to export the summary to get the counts file
        if not counts_path.exists():
             self._summarize_sequences(seqs, summary_to_use)

        viz = Visualization.load(self.files[summary_to_use])
        df = viz.view(pd.DataFrame)
        counts_numeric = pd.to_numeric(counts_df.iloc[:, 0], errors='coerce').fillna(0)
        valid_samples = counts_df[counts_numeric >= self.params['min_frequency']].index
        #valid_samples = df[df['total-sequences'] >= self.params['min_frequency']].index
        
        metadata = Metadata(pd.DataFrame(index=valid_samples))
        filtered = demux.methods.filter_samples(demux=seqs, metadata=metadata).filtered_demux
        
        filtered.save(str(self.files["filtered_seqs"]))
        print(f"Retained {len(valid_samples)} samples after filtering.")
        return filtered

    def _denoise_sequences(self, seqs: Artifact, **dada2_params) -> Tuple[Artifact, Artifact, Artifact]:
        """Denoises sequences and creates a feature table."""
        if self._skip("rep_seqs", "table", "stats"):
            print("Skipping denoising: All output artifacts exist.")
            return (
                Artifact.load(self.files["rep_seqs"]),
                Artifact.load(self.files["table"]),
                Artifact.load(self.files["stats"])
            )
        
        print("Denoising sequences with DADA2...")
        p = self.params
        if p['library_layout'] == 'paired':
            results = denoise_paired(
                demultiplexed_seqs=seqs, chimera_method=p['chimera_method'],
                n_threads=p['n_threads'], **dada2_params
            )
        else: # single
            single_params = {
                'trunc_len': dada2_params['trunc_len_f'],
                'trim_left': dada2_params['trim_left_f']
            }
            results = denoise_single(
                demultiplexed_seqs=seqs,  chimera_method=p['chimera_method'],
                n_threads=p['n_threads'], **single_params
            )
        
        results.representative_sequences.save(str(self.files["rep_seqs"]))
        results.table.save(str(self.files["table"]))
        results.denoising_stats.save(str(self.files["stats"]))
        return results.representative_sequences, results.table, results.denoising_stats

    def _classify_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        """Assigns taxonomy to representative sequences."""
        if self._skip("taxonomy"):
            print(f"Skipping taxonomy: '{self.files['taxonomy']}' exists.")
            return Artifact.load(self.files["taxonomy"])

        print("Classifying taxonomy...")
        classifier = Artifact.load(self.params['classifier_path'])
        taxonomy = classify_sklearn(
            reads=rep_seqs,
            classifier=classifier,
            confidence=self.params['confidence'],
            n_jobs=self.params['n_threads']
        ).classification
        
        taxonomy.save(str(self.files["taxonomy"]))
        return taxonomy

    def _build_phylogenetic_tree(self, rep_seqs: Artifact) -> Artifact:
        """Builds a phylogenetic tree from representative sequences."""
        if self._skip("rooted_tree"):
            print(f"Skipping phylogeny: '{self.files['rooted_tree']}' exists.")
            return Artifact.load(self.files["rooted_tree"])

        print("Building phylogenetic tree...")
        results = align_to_tree_mafft_fasttree(
            sequences=rep_seqs, n_threads=self.params['n_threads']
        )
        results.rooted_tree.save(str(self.files["rooted_tree"]))
        return results.rooted_tree

    def _collapse_table(self, table: Artifact, taxonomy: Artifact) -> Artifact:
        """Collapses the feature table to a specified taxonomic level."""
        if self._skip("collapsed_table"):
            print(f"Skipping collapse: '{self.files['collapsed_table']}' exists.")
            return Artifact.load(self.files["collapsed_table"])
        
        print(f"Collapsing feature table to level {self.params['collapse_level']}...")
        collapsed = taxa.actions.collapse(
            table=table, taxonomy=taxonomy, level=self.params['collapse_level']
        ).collapsed_table
        
        collapsed.save(str(self.files["collapsed_table"]))
        return collapsed
    
    def _determine_dada2_params(self, seqs: Artifact) -> Dict[str, int]:
        """Analyzes sequence quality to estimate optimal DADA2 parameters,
        then prompts the user for confirmation unless parameters are locked."""
        p = self.params
        is_paired = p['library_layout'] == 'paired'
        
        # Check if params are locked via command line
        if p.get('dada2_params'):
            print("DADA2 parameters locked by user command.")
            user_params = p['dada2_params']
            if is_paired and len(user_params) != 4:
                raise ValueError("--dada2-params requires 4 values for paired-end data.")
            if not is_paired and len(user_params) != 2:
                raise ValueError("--dada2-params requires 2 values for single-end data.")
            
            final_params = {'trim_left_f': user_params[2], 'trunc_len_f': user_params[0]} if is_paired else {'trim_left_f': user_params[1], 'trunc_len_f': user_params[0]}
            if is_paired:
                final_params.update({'trim_left_r': user_params[3], 'trunc_len_r': user_params[1]})
            return final_params

        # --- Interactive Path ---
        print("Entering interactive mode to determine DADA2 parameters.")
        summary_key = "02_trimmed-summary" if p.get("trim_sequences") else "01_imported-summary"
        summary_qzv_path = self.files[summary_key]

        if not summary_qzv_path.exists():
        self._summarize_sequences(seqs, summary_key)

        # Extract data from the QZV
        with zipfile.ZipFile(summary_qzv_path, 'r') as z:
            viz_root = [i for i in z.infolist() if i.is_dir()][0].filename
            fwd_qual_path = z.extract(f"{viz_root}data/forward-seven-number-summary.csv", path=self.qiime_dir)
            df_fwd = pd.read_csv(fwd_qual_path, index_col=0)
            if is_paired:
                rev_qual_path = z.extract(f"{viz_root}data/reverse-seven-number-summary.csv", path=self.qiime_dir)
                df_rev = pd.read_csv(rev_qual_path, index_col=0)

        # Estimate ideal parameters
        est_trim_f = len(p['fwd_primer_seq']) if p.get("trim_sequences") else 0
        est_trim_r = len(p['rev_primer_seq']) if p.get("trim_sequences") else 0
        
        # Find last position where median quality is >= 25
        good_qual_fwd = df_fwd[df_fwd['50%'] >= 25]
        est_trunc_f = good_qual_fwd.index.max() if not good_qual_fwd.empty else 0
        if is_paired:
            good_qual_rev = df_rev[df_rev['50%'] >= 25]
            est_trunc_r = good_qual_rev.index.max() if not good_qual_rev.empty else 0

        # Generate inspection files
        inspection_dir = self.qiime_dir / "dada2_parameter_inspection"
        inspection_dir.mkdir(exist_ok=True)
        tsv_path = inspection_dir / "forward_quality.tsv"
        png_path = inspection_dir / "quality_plot.png"
        df_fwd.to_csv(tsv_path, sep='\t')
        
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
        ax.set_title("Median Sequence Quality")
        ax.legend()
        fig.savefig(png_path)
        plt.close(fig)

        # Prompt user
        print("\n" + "="*80)
        print("📊 DADA2 PARAMETER DETERMINATION")
        print(f"Please inspect the quality plot and TSV files located in:\n{inspection_dir}")
        print("\n--- Best Estimates ---")
        print(f"Forward trim (trim_left_f): {est_trim_f}")
        print(f"Forward trunc (trunc_len_f): {est_trunc_f}")
        if is_paired:
            print(f"Reverse trim (trim_left_r): {est_trim_r}")
            print(f"Reverse trunc (trunc_len_r): {est_trunc_r}")
        print("\nEnter new values or press Enter to accept the estimate.")
        print("="*80)

        def get_user_value(prompt: str, default: int) -> int:
            while True:
                val = input(f"{prompt} [{default}]: ")
                if not val: return default
                try: return int(val)
                except ValueError: print("Invalid input. Please enter an integer.")

        final_params = {}
        final_params['trim_left_f'] = get_user_value("Enter trim_left_f", est_trim_f)
        final_params['trunc_len_f'] = get_user_value("Enter trunc_len_f", est_trunc_f)
        if is_paired:
            final_params['trim_left_r'] = get_user_value("Enter trim_left_r", est_trim_r)
            final_params['trunc_len_r'] = get_user_value("Enter trunc_len_r", est_trunc_r)

        print(f"Using final DADA2 parameters: {final_params}")
        return final_params

    def _setup_file_registry(self) -> None:
        """Initializes paths for all input and output files."""
        d = self.qiime_dir
        level = self.params['collapse_level']
        self.files: Dict[str, Path] = {
            "seqs": d / "01_demux-sequences.qza",
            "imported_summary": d / "01_imported-summary.qzv",
            "trimmed_seqs": d / "02_trimmed-sequences.qza",
            "trimmed_summary": d / "02_trimmed-summary.qzv",
            "filtered_seqs": d / "03_filtered-sequences.qza",
            "rep_seqs": d / "04_representative-sequences.qza",
            "table": d / "04_feature-table.qza",
            "stats": d / "04_denoising-stats.qza",
            "taxonomy": d / "05_taxonomy.qza",
            "rooted_tree": d / "06_rooted-tree.qza",
            "collapsed_table": d / f"07_collapsed-table-L{level}.qza",
        }

    def _skip(self, *keys: str) -> bool:
        """Checks if all specified output files exist and hard_rerun is False."""
        if self.params.get("hard_rerun", False): return False
        return all(self.files[key].exists() for key in keys)

    def _check_inputs(self) -> None:
        """Validates the existence of required input files."""
        if not self.params['manifest_tsv'].is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.params['manifest_tsv']}")
        if not self.params['classifier_path'].is_file():
            raise FileNotFoundError(f"Classifier file not found: {self.params['classifier_path']}")

    def _clean_qiime_dir(self, artifacts_only: bool = False) -> None:
        """Removes QIIME 2 outputs."""
        if not self.qiime_dir.is_dir(): return
        if artifacts_only:
            print(f"Cleaning QIIME 2 artifacts in {self.qiime_dir}...")
            for f in self.qiime_dir.glob("*.qza"): f.unlink()
            for f in self.qiime_dir.glob("*.qzv"): f.unlink()
        else:
            print(f"Removing entire QIIME 2 directory: {self.qiime_dir}...")
            shutil.rmtree(self.qiime_dir)

# ============================ ARGUMENT PARSING & MAIN ============================= #

def path_validator(path_str: str, check_file: bool = False) -> Path:
    """Validates a path: creates directories, checks if files exist."""
    p = Path(path_str).resolve()
    if check_file:
        if not p.is_file():
            raise argparse.ArgumentTypeError(f"File not found: {p}")
    else: # is a directory
        p.mkdir(parents=True, exist_ok=True)
    return p

def main() -> None:
    """Parses arguments and runs the main workflow."""
    parser = argparse.ArgumentParser(
        description='QIIME 2 Self-Contained Amplicon Analysis Workflow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    io_group = parser.add_argument_group("Input/Output Parameters")
    io_group.add_argument("--manifest_tsv", type=lambda p: path_validator(p, check_file=True), required=True, help="Path to the input manifest TSV file.")
    io_group.add_argument("--qiime_dir", type=path_validator, required=True, help="Path to the output directory for QIIME 2 results.")
    io_group.add_argument("--classifier_path", type=lambda p: path_validator(p, check_file=True), required=True, help="Full path to the QIIME 2 classifier artifact (.qza).")

    seq_group = parser.add_argument_group("Sequencing Parameters")
    seq_group.add_argument("--library_layout", choices=['single', 'paired'], required=True, help="Sequencing library layout.")
    seq_group.add_argument("--fwd_primer_seq", required=True, help="Forward primer sequence (5' to 3').")
    seq_group.add_argument("--rev_primer_seq", required=True, help="Reverse primer sequence (5' to 3').")

    denoise_group = parser.add_argument_group("Denoising Parameters (DADA2)")
    denoise_group.add_argument("--dada2-params", type=int, nargs='+', help=("Lock in DADA2 parameters to bypass interactive prompt. Provide in order: trunc_len_f trunc_len_r trim_left_f trim_left_r. For single-end, provide trunc_len_f and trim_left_f."))
    denoise_group.add_argument("--chimera_method", default="consensus", choices=["none", "consensus", "pooled"], help="Method for chimera detection.")

    proc_group = parser.add_argument_group("Processing Parameters")
    proc_group.add_argument("--n_threads", type=int, default=DEFAULT_N_THREADS, help="Number of CPU threads to use.")
    proc_group.add_argument('--min_frequency', type=int, default=0, help='...')
    #proc_group.add_argument("--min_frequency", type=int, default=DEFAULT_MIN_FREQUENCY, help="Minimum read count to retain a sample after trimming.")
    proc_group.add_argument("--collapse_level", type=int, default=DEFAULT_COLLAPSE_LEVEL, help="Taxonomic level to collapse the feature table to (1-7).")
    proc_group.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold for retaining taxonomy assignments (sklearn only).")
    proc_group.add_argument("--hard_rerun", action="store_true", help="Force reprocessing of all steps, overwriting existing files.")
    proc_group.add_argument("--trim_sequences", action="store_true", help="Enable primer trimming with Cutadapt.")
    
    args = parser.parse_args()
    
    try:
        workflow = QIIMEWorkflow(**vars(args))
        workflow.run()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Workflow failed with the following error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()