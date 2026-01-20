#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QIIME 2 Self-Contained Amplicon Analysis Workflow (Optimized & Robust)
"""

# ===================================== IMPORTS ====================================== #

import argparse
import logging
import os
import shutil
import sys
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

# Third-party imports for analysis and data handling
import pandas as pd
from qiime2 import Artifact, Metadata, Visualization # type: ignore
from qiime2.plugins import demux, taxa, feature_table # type: ignore
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

# Reserve 1 core for OS overhead
DEFAULT_N_THREADS = max(1, (os.cpu_count() or 1) - 1)
DEFAULT_MIN_FREQUENCY = 1000
DEFAULT_COLLAPSE_LEVEL = 6  # Genus

# ================================ WORKFLOW CLASS ================================== #

class QIIMEWorkflow:
    """Manages and executes the QIIME 2 analysis pipeline."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.qiime_dir = self.params['qiime_dir']
        self._setup_file_registry()

    def run(self) -> None:
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
        logging.info("[INSPECT MODE] Starting DADA2 parameter inspection.")
        seqs = self._import_sequences()
        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)
        
        filtered_seqs = self._filter_sequences(seqs)
        self._summarize_sequences(filtered_seqs, "03_filtered-summary")
        self._generate_inspection_files("03_filtered-summary")
        logging.info("[INSPECT MODE] Inspection complete.")

    def _run_execution_workflow(self, mode: str) -> None:
        logging.info(f"[{mode.upper()} MODE] Starting full QIIME 2 artifact generation.")
        
        seqs = self._import_sequences()
        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)

        filtered_seqs = self._filter_sequences(seqs)

        # --- DADA2 Parameter Logic ---
        if mode == 'auto':
            # We assume the summary exists from the filtering step or we generate it
            summary_key = "03_filtered-summary"
            if not self.files[summary_key].exists():
                self._summarize_sequences(filtered_seqs, summary_key)
            
            dada2_params = self._estimate_dada2_params(summary_key)
            logging.info(f"Automatically determined DADA2 parameters: {dada2_params}")
        else: 
            dada2_params = self._get_locked_dada2_params()
            logging.info(f"Using manually specified DADA2 parameters: {dada2_params}")

        # --- Core Pipeline ---
        rep_seqs, table, _ = self._denoise_sequences(filtered_seqs, **dada2_params)
        taxonomy = self._classify_taxonomy(rep_seqs)
        rooted_tree = self._build_phylogenetic_tree(rep_seqs)
        self._collapse_table(table, taxonomy)
        
        self._export_final_artifacts(table, taxonomy, rep_seqs, rooted_tree)
        logging.info(f"[{mode.upper()} MODE] Pipeline completed successfully.")

    # --- Step Methods ---

    def _import_sequences(self) -> Artifact:
        if self._skip("seqs"): return Artifact.load(self.files['seqs'])
        logging.info("Importing sequences...")
        layout = self.params['library_layout']
        types = {
            "single": ("SampleData[SequencesWithQuality]", "SingleEndFastqManifestPhred33V2"),
            "paired": ("SampleData[PairedEndSequencesWithQuality]", "PairedEndFastqManifestPhred33V2"),
        }
        import_type, view_type = types[layout]
        seqs = Artifact.import_data(import_type, str(self.params['manifest_tsv']), view_type=view_type)
        seqs.save(str(self.files["seqs"]))
        return seqs

    def _summarize_sequences(self, seqs: Artifact, base_name: str) -> None:
        if self._skip(base_name): return
        logging.info(f"Generating summary visualization for {base_name}...")
        summary_viz = demux.visualizers.summarize(data=seqs).visualization
        summary_viz.save(str(self.files[base_name]))

    def _trim_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("trimmed_seqs"): return Artifact.load(self.files['trimmed_seqs'])
        
        fwd_primer = self.params.get('fwd_primer_seq', "N/A")
        if not fwd_primer or fwd_primer.upper() == "N/A":
            logging.info("Primer sequence is 'N/A', skipping Cutadapt trimming.")
            # Create a copy or softlink to maintain pipeline flow
            seqs.save(str(self.files["trimmed_seqs"]))
            return seqs

        logging.info("Trimming primer sequences with Cutadapt...")
        rev_primer_rc = self._reverse_complement(self.params.get('rev_primer_seq', ''))
        
        if self.params['library_layout'] == 'paired':
            trimmed = trim_paired(demultiplexed_sequences=seqs, front_f=[fwd_primer], front_r=[rev_primer_rc], cores=self.params['n_threads']).trimmed_sequences
        else:
            trimmed = trim_single(demultiplexed_sequences=seqs, front=[fwd_primer], cores=self.params['n_threads']).trimmed_sequences
        trimmed.save(str(self.files["trimmed_seqs"]))
        return trimmed
    
    def _filter_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("filtered_seqs"): return Artifact.load(self.files['filtered_seqs'])
        
        # Use whichever summary is most recent
        summary_to_use = "02_trimmed-summary" if self.params.get("trim_sequences") else "01_imported-summary"
        if not self.files[summary_to_use].exists():
            self._summarize_sequences(seqs, summary_to_use)
        
        logging.info(f"Filtering samples with fewer than {self.params['min_frequency']} reads...")
        
        # Extract counts using robust method
        counts_df = self._extract_file_from_qzv(
            self.files[summary_to_use], 
            "per-sample-fastq-counts.tsv",
            lambda f: pd.read_csv(f, sep='\t', comment='#', index_col=0, header=None)
        )
        # Note: Changed reader to handle TSV headers more reliably (comment='#', header=None)
        # The QIIME 2 "per-sample-fastq-counts.tsv" usually lacks a standard header or has comments at the top.
        
        if counts_df is None: 
            raise ValueError("Failed to load counts DataFrame from artifact.")
        
        # --- CRITICAL FIX: Explicitly naming the index for QIIME 2 Metadata validation ---
        counts_df.index.name = '#SampleID' 
        
        # Filter logic
        # Column 0 contains the counts
        counts_numeric = pd.to_numeric(counts_df.iloc[:, 0], errors='coerce').fillna(0)
        valid_samples = counts_df[counts_numeric >= self.params['min_frequency']].index
        #valid_samples = counts_df[counts_df.iloc[:, 0] >= self.params['min_frequency']].index
        
        if len(valid_samples) == 0:
            raise ValueError(f"No samples retained after filtering threshold {self.params['min_frequency']}.")
        
        # Create DataFrame with valid samples index and empty columns
        # QIIME 2 Metadata requires an ID header
        metadata_df = pd.DataFrame(index=valid_samples)
        metadata_df.index.name = '#SampleID'
        
        metadata = Metadata(metadata_df)
        
        filtered = demux.methods.filter_samples(demux=seqs, metadata=metadata).filtered_demux
        filtered.save(str(self.files["filtered_seqs"]))
        return filtered

    def _denoise_sequences(self, seqs: Artifact, **dada2_params) -> Tuple[Artifact, Artifact, Artifact]:
        if self._skip("rep_seqs", "table", "stats"):
            return (Artifact.load(self.files["rep_seqs"]), Artifact.load(self.files["table"]), Artifact.load(self.files["stats"]))
        
        logging.info("Denoising sequences with DADA2...")
        p = self.params
        if p['library_layout'] == 'paired':
            results = denoise_paired(demultiplexed_seqs=seqs, chimera_method=p['chimera_method'], n_threads=p['n_threads'], **dada2_params)
        else:
            # Single end signature differs
            single_params = {k: v for k, v in dada2_params.items() if 'trim_left_f' in k or 'trunc_len_f' in k}
            # Map keys to single-end expected arguments if necessary, usually just 'trim_left' and 'trunc_len'
            # But QIIME2 python plugin args are slightly different. Let's be explicit:
            results = denoise_single(
                demultiplexed_seqs=seqs, 
                chimera_method=p['chimera_method'], 
                n_threads=p['n_threads'],
                trunc_len=dada2_params['trunc_len_f'],
                trim_left=dada2_params['trim_left_f']
            )
        
        results.representative_sequences.save(str(self.files["rep_seqs"]))
        results.table.save(str(self.files["table"]))
        results.denoising_stats.save(str(self.files["stats"]))
        return results.representative_sequences, results.table, results.denoising_stats

    def _classify_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        if self._skip("taxonomy"): return Artifact.load(self.files["taxonomy"])
        logging.info("Classifying taxonomy...")
        classifier = Artifact.load(self.params['classifier_path'])
        
        # OPTIMIZATION: Cap threads for classification to prevent memory explosion/swapping
        # but increase batch size for throughput
        safe_jobs = min(self.params['n_threads'], 12) 
        
        taxonomy = classify_sklearn(
            reads=rep_seqs, 
            classifier=classifier, 
            confidence=self.params['confidence'], 
            n_jobs=safe_jobs,
            reads_per_batch=20000, # Larger batches reduce overhead
            pre_dispatch='2*n_jobs'
        ).classification
        
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
    
    def _export_final_artifacts(self, table: Artifact, taxonomy: Artifact, rep_seqs: Artifact, tree: Artifact):
        logging.info("Exporting final artifacts...")
        export_map = {
            "exported_biom": (table, "feature-table.biom"),
            "exported_taxonomy": (taxonomy, "taxonomy.tsv"),
            "exported_rep_seqs": (rep_seqs, "dna-sequences.fasta"),
            "exported_tree": (tree, "tree.nwk")
        }
        for key, (artifact, expected_filename) in export_map.items():
            if self._skip(key): continue
            try:
                artifact.export_data(self.qiime_dir)
                # Handle older QIIME export behavior where it might create a folder
                # Just ensure the file ends up in qiime_dir with expected_filename
                possible_locs = [
                    self.qiime_dir / expected_filename, 
                    self.qiime_dir / 'biom-table.tsv', 
                    self.qiime_dir / 'taxonomy.tsv'
                ]
                # Simple check if file exists, otherwise log warning
                if not (self.qiime_dir / expected_filename).exists():
                     logging.warning(f"Exported file {expected_filename} not found immediately at target path. Check output dir.")
            except Exception as e:
                logging.error(f"Export error for {expected_filename}: {e}")

    # --- Utility Methods ---

    def _extract_file_from_qzv(self, qzv_path: Path, target_filename: str, reader_func):
        """
        Robustly extracts a file from a QZV by scanning filenames instead of guessing the UUID folder.
        """
        try:
            with zipfile.ZipFile(qzv_path, 'r') as z:
                # Find the file within the zip structure
                target_path = None
                for name in z.namelist():
                    if name.endswith(target_filename) and not name.startswith('__MACOSX'):
                        target_path = name
                        break
                
                if not target_path:
                    logging.warning(f"Could not find '{target_filename}' inside {qzv_path.name}")
                    return None
                
                with z.open(target_path) as f:
                    return reader_func(f)
        except Exception as e:
            logging.warning(f"Error reading artifact {qzv_path}: {e}")
            return None

    def _estimate_dada2_params(self, summary_key: str) -> Dict[str, int]:
        """
        Intelligently estimates DADA2 parameters ensuring overlap.
        """
        is_paired = self.params['library_layout'] == 'paired'
        fwd_primer = self.params.get('fwd_primer_seq', "")
        rev_primer = self.params.get('rev_primer_seq', "")
        
        trim_f = len(fwd_primer) if self.params.get("trim_sequences") else 0
        trim_r = len(rev_primer) if is_paired and self.params.get("trim_sequences") else 0

        summary_qzv_path = self.files[summary_key]
        
        # Load Forward
        df_fwd = self._extract_file_from_qzv(
            summary_qzv_path, 
            "forward-seven-number-summary.csv",
            lambda f: pd.read_csv(f, index_col=0)
        )
        
        # Load Reverse
        df_rev = None
        if is_paired:
            df_rev = self._extract_file_from_qzv(
                summary_qzv_path, 
                "reverse-seven-number-summary.csv",
                lambda f: pd.read_csv(f, index_col=0)
            )

        if df_fwd is None or (is_paired and df_rev is None):
            logging.warning("Quality data missing from artifact. Using safe defaults.")
            return self._get_safe_defaults(is_paired)

        # Calculate Truncation Points
        def find_trunc_pos(df, threshold=25):
            bad_qual_indices = df[df['50%'] < threshold].index
            if len(bad_qual_indices) > 0:
                return int(bad_qual_indices[0])
            return int(df.index.max())

        trunc_f = find_trunc_pos(df_fwd)
        trunc_r = find_trunc_pos(df_rev) if is_paired else 0

        # Overlap Validation
        if is_paired:
            TARGET_AMPLICON_SIZE = 253 # V4 515F-806R
            REQUIRED_TOTAL_LEN = TARGET_AMPLICON_SIZE + 20 
            
            effective_f = trunc_f - trim_f
            effective_r = trunc_r - trim_r
            total_len = effective_f + effective_r
            
            if total_len < REQUIRED_TOTAL_LEN:
                logging.warning(f"Calculated truncation ({trunc_f}, {trunc_r}) risks poor overlap.")
                shortfall = REQUIRED_TOTAL_LEN - total_len
                max_f = int(df_fwd.index.max())
                max_r = int(df_rev.index.max())
                trunc_f = min(max_f, trunc_f + (shortfall // 2) + 5)
                trunc_r = min(max_r, trunc_r + (shortfall // 2) + 5)
                logging.warning(f"Adjusted to ({trunc_f}, {trunc_r}) to enforce overlap.")

        params = {'trim_left_f': trim_f, 'trunc_len_f': trunc_f}
        if is_paired:
            params.update({'trim_left_r': trim_r, 'trunc_len_r': trunc_r})
            
        return params

    def _get_safe_defaults(self, is_paired: bool) -> Dict[str, int]:
        # Conservative V4 defaults
        defaults = {'trim_left_f': 0, 'trunc_len_f': 220} # Reduced from 240 to be safer
        if is_paired:
            defaults.update({'trim_left_r': 0, 'trunc_len_r': 180}) # Reduced from 200
        return defaults

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        if not seq: return ""
        complement_map = str.maketrans("ATGCatgcRYrySWswKMkmBDHVbdhvNn", "TACGtacgYRyrWSwsMKmkVHDBvhdbNn")
        return seq.translate(complement_map)[::-1]

    def _get_locked_dada2_params(self) -> Dict[str, int]:
        user_params = self.params['dada2_params']
        is_paired = self.params['library_layout'] == 'paired'
        if is_paired:
            return {'trunc_len_f': user_params[0], 'trunc_len_r': user_params[1],
                    'trim_left_f': user_params[2], 'trim_left_r': user_params[3]}
        else:
            return {'trunc_len_f': user_params[0], 'trim_left_f': user_params[1]}

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
            "exported_biom": d / "feature-table.biom",
            "exported_taxonomy": d / "taxonomy.tsv",
            "exported_rep_seqs": d / "dna-sequences.fasta",
            "exported_tree": d / "tree.nwk",
        }

    def _skip(self, *keys: str) -> bool:
        if self.params.get("hard_rerun", False): return False
        return all(self.files.get(key, Path()).exists() for key in keys)

    def _check_inputs(self) -> None:
        manifest = self.params['manifest_tsv']
        classifier = self.params.get('classifier_path')
        if not manifest or not Path(manifest).is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest}")
        if self.params['dada2_mode'] != 'inspect' and (not classifier or not Path(classifier).is_file()):
            raise FileNotFoundError(f"Classifier file not found: {classifier}")

    def _generate_inspection_files(self, summary_key: str) -> None:
        # Placeholder for inspection logic (simplified for brevity)
        pass

    def _clean_qiime_dir(self, artifacts_only: bool = False) -> None:
        if not self.qiime_dir.is_dir(): return
        if artifacts_only:
            logging.info(f"Cleaning QIIME 2 artifacts in {self.qiime_dir}...")
            for ext in ("*.qza", "*.qzv", "*.biom", "*.tsv", "*.fasta", "*.nwk"):
                for f in self.qiime_dir.glob(ext): f.unlink()
        else:
            logging.info(f"Removing entire QIIME 2 directory: {self.qiime_dir}...")
            shutil.rmtree(self.qiime_dir)

# ============================ ARGUMENT PARSING & MAIN ============================= #

def main() -> None:
    parser = argparse.ArgumentParser(description='QIIME 2 Self-Contained Amplicon Analysis Workflow.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode_group = parser.add_argument_group("Workflow Mode")
    mode_group.add_argument("--dada2-mode", required=True, choices=['inspect', 'auto', 'manual'], help="Select the DADA2 parameter handling mode.")
    io_group = parser.add_argument_group("Input/Output Parameters")
    io_group.add_argument("--manifest_tsv", type=Path, required=True, help="Path to the input manifest TSV file.")
    io_group.add_argument("--metadata_tsv", type=Path, help="Path to the input metadata TSV file.")
    io_group.add_argument("--qiime_dir", type=Path, required=True, help="Path to the output directory for QIIME 2 results.")
    io_group.add_argument("--classifier_path", type=Path, help="Full path to the QIIME 2 classifier artifact (.qza).")
    seq_group = parser.add_argument_group("Sequencing Parameters")
    seq_group.add_argument("--library_layout", choices=['single', 'paired'], required=True, help="Sequencing library layout.")
    seq_group.add_argument("--fwd_primer_seq", type=str, required=True, help="Forward primer sequence (5' to 3').")
    seq_group.add_argument("--rev_primer_seq", type=str, help="Reverse primer sequence (5' to 3'). Required for paired-end.")
    denoise_group = parser.add_argument_group("Denoising Parameters (DADA2)")
    denoise_group.add_argument("--dada2-params", type=int, nargs='+', help="[MANUAL MODE] DADA2 params: trunc_f, trunc_r, trim_f, trim_r.")
    denoise_group.add_argument("--chimera_method", default="consensus", choices=["none", "consensus", "pooled"], help="Method for chimera detection.")
    proc_group = parser.add_argument_group("Processing Parameters")
    proc_group.add_argument("--n_threads", type=int, default=DEFAULT_N_THREADS, help="Number of CPU threads to use.")
    proc_group.add_argument("--min_frequency", type=int, default=DEFAULT_MIN_FREQUENCY, help="Minimum read count to retain a sample.")
    proc_group.add_argument("--collapse_level", type=int, default=DEFAULT_COLLAPSE_LEVEL, help="Taxonomic level to collapse table (1-7).")
    proc_group.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold for taxonomy assignment.")
    proc_group.add_argument("--hard_rerun", action="store_true", help="Force reprocessing of all steps.")
    proc_group.add_argument("--trim_sequences", action="store_true", help="Enable primer trimming with Cutadapt.")
    args = parser.parse_args()
    
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
        logging.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()