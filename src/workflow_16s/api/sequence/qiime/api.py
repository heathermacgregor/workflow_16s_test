#!/usr/-bin/env python3
# -*- coding: utf-8 -*-

"""
QIIME 2 Amplicon Sequencing Analysis Workflow

This script provides a comprehensive, command-line-driven pipeline for processing
16S rRNA gene amplicon sequencing data using QIIME 2. It automates all steps
from data importing and quality control to taxonomic classification and
phylogenetic analysis.

The workflow includes:
1.  Checking for the correct Conda environment.
2.  Importing raw sequence data from a manifest file.
3.  Trimming primer sequences using Cutadapt.
4.  Filtering samples with low read counts.
5.  Denoising sequences into Amplicon Sequence Variants (ASVs) using DADA2 or Deblur.
6.  Assigning taxonomy to ASVs using a pre-trained classifier.
7.  Generating a phylogenetic tree from representative sequences.
8.  Collapsing the feature table to a specified taxonomic level.

Usage:
    python your_script_name.py \\
        --manifest-file /path/to/your/manifest.tsv \\
        --output-dir /path/to/your/output_directory \\
        --fwd-primer-seq CCTACGGGAGGCAGCAG \\
        --rev-primer-seq GACTACHVGGGTATCTAATCC \\
        --trunc-len-f 240 \\
        --trunc-len-r 240
"""

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Union

# Third-Party Imports
from Bio.Seq import Seq
import pandas as pd
from qiime2 import Artifact, Metadata # type: ignore
from qiime2.plugins import demux, taxa  # type: ignore
from qiime2.plugins.cutadapt.methods import trim_paired, trim_single  # type: ignore
from qiime2.plugins.dada2.methods import denoise_paired, denoise_pyro, denoise_single  # type: ignore
from qiime2.plugins.deblur.methods import denoise_16S  # type: ignore
from qiime2.plugins.demux.visualizers import summarize as summarize_demux  # type: ignore
from qiime2.plugins.feature_classifier.methods import classify_sklearn  # type: ignore
from qiime2.plugins.feature_classifier.pipelines import classify_consensus_blast  # type: ignore
from qiime2.plugins.phylogeny.pipelines import align_to_tree_mafft_fasttree  # type: ignore

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================================= DEFAULT VALUES =================================== #

REQUIRED_CONDA_ENV = "qiime2-2025.8"
DEFAULT_N_THREADS = max(1, (os.cpu_count() or 2) // 2)
DEFAULT_MIN_READS = 1000
DEFAULT_LIBRARY_LAYOUT = "paired"
DEFAULT_MINIMUM_LENGTH = 100
DEFAULT_TRUNC_LEN_F = 250
DEFAULT_TRUNC_LEN_R = 250
DEFAULT_TRIM_LENGTH = 250
DEFAULT_TRIM_LEFT_F = 0
DEFAULT_TRIM_LEFT_R = 0
DEFAULT_TRUNC_Q = 2
DEFAULT_MAX_EE_F = 2
DEFAULT_MAX_EE_R = 5
DEFAULT_CHIMERA_METHOD = "consensus"
DEFAULT_DENOISE_ALGORITHM = "dada2"
DEFAULT_CLASSIFIER_DIR = "./"  # Default to current dir if not provided
DEFAULT_CLASSIFIER = "silva-138-99-515-806"
DEFAULT_CLASSIFY_METHOD = "sklearn"
DEFAULT_CONFIDENCE = 0.7
DEFAULT_TAXONOMY_LEVEL = 6  # Genus level

# ================================= HELPER FUNCTIONS ================================= #

def _save_and_export(artifact: Union[Artifact, Metadata], output_dir: Path, base_name: str):
    """Saves a QIIME 2 artifact and exports its data."""
    is_viz = str(artifact.type) == 'Visualization'
    ext = "qzv" if is_viz else "qza"
    
    # Save the artifact
    qza_path = output_dir / f"{base_name}.{ext}"
    artifact.save(str(qza_path))
    logging.info(f"Saved artifact to: {qza_path}")

    # Export the data
    export_path = output_dir / base_name
    export_path.mkdir(parents=True, exist_ok=True)
    artifact.export_data(str(export_path))
    logging.info(f"Exported data to: {export_path}")


# ================================= QIIME 2 FUNCTIONS ================================ #

def import_seqs_from_manifest(output_dir: Path, manifest_tsv: Path, library_layout: str) -> Artifact:
    """Imports sequence data from a manifest file and generates quality control stats."""
    logging.info("Importing sequences from manifest file...")
    layout = library_layout.lower()
    if layout not in {"paired", "single"}:
        raise ValueError(f"Unrecognized library layout: '{library_layout}'. Expected 'single' or 'paired'.")

    layout_config = {
        "single": ("SampleData[SequencesWithQuality]", "SingleEndFastqManifestPhred33V2"),
        "paired": ("SampleData[PairedEndSequencesWithQuality]", "PairedEndFastqManifestPhred33V2"),
    }

    import_type, view_type = layout_config[layout]
    seqs = Artifact.import_data(import_type, str(manifest_tsv), view_type=view_type)
    
    _save_and_export(seqs, output_dir, "01_imported-seqs")
    seqs_summary = summarize_demux(data=seqs).visualization
    _save_and_export(seqs_summary, output_dir, "01_demux-summary")

    return seqs


def trim_sequences(output_dir: Path, seqs: Artifact, library_layout: str, fwd_primer_seq: str,
                   rev_primer_seq: str, minimum_length: int, n_cores: int) -> Artifact:
    """Trims primer sequences from reads using Cutadapt with a fallback mechanism."""
    logging.info("Trimming primer sequences with Cutadapt...")
    rev_primer_rc = str(Seq(rev_primer_seq).reverse_complement())
    layout = library_layout.lower()
    trimmed_seqs_result = None

    try:
        if "paired" in layout:
            logging.info("Attempting paired-end trimming workflow.")
            trimmed_seqs_result = trim_paired(
                demultiplexed_sequences=seqs,
                front_f=[fwd_primer_seq],
                front_r=[rev_primer_rc],
                minimum_length=minimum_length,
                cores=n_cores,
            )
        elif "single" in layout:
            logging.info("Attempting single-end trimming workflow.")
            trimmed_seqs_result = trim_single(
                demultiplexed_sequences=seqs,
                front=[fwd_primer_seq],
                minimum_length=minimum_length,
                cores=n_cores,
            )
        else:
            raise ValueError(f"Unrecognized library layout: '{library_layout}'.")
    except Exception as e:
        logging.error(f"Primary trimming workflow failed: {e}")
        raise RuntimeError("Trimming workflow failed.")
    
    trimmed_seqs = trimmed_seqs_result.trimmed_sequences
    _save_and_export(trimmed_seqs, output_dir, "02_trimmed-seqs")
    summary = summarize_demux(data=trimmed_seqs).visualization
    _save_and_export(summary, output_dir, "02_trimmed-summary")

    return trimmed_seqs


def filter_samples_for_denoising(output_dir: Path, seqs: Artifact, min_reads: int) -> Artifact:
    """Filters samples to retain those with a minimum number of reads."""
    logging.info(f"Filtering samples to retain only those with >= {min_reads} reads...")
    
    # The summary from the previous step is needed to get the counts file
    counts_file = output_dir / "02_trimmed-summary" / "per-sample-fastq-counts.tsv"
    if not counts_file.exists():
        raise FileNotFoundError(
            "Could not find the counts file needed for filtering. "
            f"Expected at: {counts_file}"
        )
        
    df = pd.read_csv(counts_file, sep="\t", header=0)
    # The first column is the sample ID, which becomes the index
    df.set_index(df.columns[0], inplace=True)

    if 'forward sequence count' in df.columns:
        valid_samples = df[df['forward sequence count'] >= min_reads].index
    else:
        raise ValueError("Counts file must contain a 'forward sequence count' column.")
        
    # Create metadata file for filtering
    metadata_df = pd.DataFrame(index=valid_samples)
    metadata_df.index.name = "sample-id"
    
    filtered_seqs = demux.methods.filter_samples(
        demux=seqs,
        metadata=Metadata(metadata_df)
    ).filtered_demux
    
    _save_and_export(filtered_seqs, output_dir, "03_filtered-seqs")
    logging.info(f"Retained {len(valid_samples)} samples after filtering.")

    return filtered_seqs


def denoise_sequences(output_dir: Path, seqs: Artifact, library_layout: str, denoise_algorithm: str,
                      **kwargs) -> Tuple[Artifact, Artifact, Artifact]:
    """Denoises sequences using DADA2 or Deblur."""
    algorithm = denoise_algorithm.lower()
    logging.info(f"Denoising sequences with {algorithm.upper()}...")
    
    if algorithm == "dada2":
        if "single" in library_layout.lower():
            func, params = denoise_single, {
                "demultiplexed_seqs": seqs, "trunc_len": kwargs["trunc_len_f"],
                "trim_left": kwargs["trim_left_f"], "max_ee": kwargs["max_ee_f"],
            }
        else:
            func, params = denoise_paired, {
                "demultiplexed_seqs": seqs, "trunc_len_f": kwargs["trunc_len_f"],
                "trunc_len_r": kwargs["trunc_len_r"], "trim_left_f": kwargs["trim_left_f"],
                "trim_left_r": kwargs["trim_left_r"], "max_ee_f": kwargs["max_ee_f"],
                "max_ee_r": kwargs["max_ee_r"],
            }
        params.update({"chimera_method": kwargs["chimera_method"], "n_threads": kwargs["n_threads"]})
    elif algorithm == "deblur":
        func, params = denoise_16S, {
            "demultiplexed_seqs": seqs, "trim_length": kwargs["trim_length"],
            "sample_stats": True, "jobs_to_start": kwargs["n_threads"],
        }
    else:
        raise ValueError(f"Denoise algorithm '{algorithm}' not recognized.")
        
    results = func(**params)
    rep_seqs = results.representative_sequences
    table = results.table
    stats = getattr(results, 'denoising_stats', None)

    _save_and_export(rep_seqs, output_dir, "04_rep-seqs")
    _save_and_export(table, output_dir, "04_table")
    if stats:
        _save_and_export(stats, output_dir, "04_denoising-stats")

    return rep_seqs, table, stats


def classify_taxonomy(output_dir: Path, rep_seqs: Artifact, classifier_dir: Path, classifier: str,
                      classify_method: str, n_threads: int, **kwargs) -> Artifact:
    """Assigns taxonomy to representative sequences."""
    logging.info(f"Classifying taxonomy with {classify_method}...")
    
    if classify_method.lower() == "sklearn":
        clf_path = classifier_dir / f"{classifier}-classifier.qza"
        if not clf_path.exists():
            raise FileNotFoundError(f"Classifier artifact not found: {clf_path}")
        clf = Artifact.load(clf_path)
        result = classify_sklearn(
            reads=rep_seqs, classifier=clf,
            confidence=kwargs.get('confidence', DEFAULT_CONFIDENCE), n_jobs=n_threads
        )
    else:  # Assumes BLAST
        ref_tax_path = classifier_dir / f"{classifier}-tax.qza"
        ref_reads_path = classifier_dir / f"{classifier}-seqs.qza"
        ref_tax = Artifact.load(ref_tax_path)
        ref_reads = Artifact.load(ref_reads_path)
        result = classify_consensus_blast(
            query=rep_seqs, reference_reads=ref_reads, reference_taxonomy=ref_tax,
            threads=n_threads, **kwargs
        )
    
    taxonomy_artifact = result.classification
    _save_and_export(taxonomy_artifact, output_dir, "05_taxonomy")
    return taxonomy_artifact


def build_phylogenetic_tree(output_dir: Path, rep_seqs: Artifact, n_threads: int) -> Artifact:
    """Builds a phylogenetic tree using MAFFT for alignment and FastTree for tree construction."""
    logging.info("Building phylogenetic tree with MAFFT and FastTree...")
    results = align_to_tree_mafft_fasttree(sequences=rep_seqs, n_threads=n_threads)
    
    _save_and_export(results.alignment, output_dir, "06_aligned-rep-seqs")
    _save_and_export(results.unrooted_tree, output_dir, "06_unrooted-tree")
    _save_and_export(results.rooted_tree, output_dir, "06_rooted-tree")
    
    return results.rooted_tree


def collapse_taxonomy(output_dir: Path, table: Artifact, taxonomy: Artifact, level: int) -> Artifact:
    """Collapses a feature table to a specified taxonomic level."""
    logging.info(f"Collapsing feature table to level {level}...")
    collapsed_table = taxa.actions.collapse(
        table=table, taxonomy=taxonomy, level=level
    ).collapsed_table
    _save_and_export(collapsed_table, output_dir, f"07_collapsed-table-L{level}")
    return collapsed_table


# ================================== MAIN WORKFLOW =================================== #

def run_workflow(args):
    """Orchestrates the entire QIIME 2 workflow."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Import sequences
    seqs = import_seqs_from_manifest(
        output_dir=output_dir, manifest_tsv=Path(args.manifest_file),
        library_layout=args.library_layout
    )

    # Step 2: Trim primers
    trimmed_seqs = trim_sequences(
        output_dir=output_dir, seqs=seqs, library_layout=args.library_layout,
        fwd_primer_seq=args.fwd_primer_seq, rev_primer_seq=args.rev_primer_seq,
        minimum_length=args.minimum_length, n_cores=args.n_threads
    )
    
    # Step 3: Filter low-count samples
    filtered_seqs = filter_samples_for_denoising(
        output_dir=output_dir, seqs=trimmed_seqs, min_reads=args.min_reads
    )

    # Step 4: Denoise
    rep_seqs, table, _ = denoise_sequences(
        output_dir=output_dir, seqs=filtered_seqs,
        library_layout=args.library_layout,
        denoise_algorithm=args.denoise_algorithm,
        # Pass all relevant args to the function
        **vars(args)
    )

    # Step 5: Classify taxonomy
    taxonomy_artifact = classify_taxonomy(
        output_dir=output_dir, rep_seqs=rep_seqs,
        classifier_dir=Path(args.classifier_dir), classifier=args.classifier,
        classify_method=args.classify_method, n_threads=args.n_threads,
        confidence=args.confidence
    )

    # Step 6: Build phylogenetic tree
    build_phylogenetic_tree(output_dir=output_dir, rep_seqs=rep_seqs, n_threads=args.n_threads)

    # Step 7: Collapse feature table
    collapse_taxonomy(
        output_dir=output_dir, table=table,
        taxonomy=taxonomy_artifact, level=args.taxonomy_level
    )
    
    logging.info("Workflow completed successfully!")


def main():
    """Parses command-line arguments and launches the workflow."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != REQUIRED_CONDA_ENV:
        sys.stderr.write(
            f"\nIncorrect Conda Environment!\n"
            f"Required: '{REQUIRED_CONDA_ENV}'\n"
            f"You are in: '{current_env or 'Not a Conda environment'}'\n\n"
            f"Please activate the correct environment by running:\n"
            f"conda activate {REQUIRED_CONDA_ENV}\n"
        )
        sys.exit(1)
    
    logging.info(f"Correct Conda environment ('{current_env}') detected.")

    parser = argparse.ArgumentParser(description="QIIME 2 Amplicon Sequencing Analysis Workflow")

    # Required arguments
    req_group = parser.add_argument_group('Required Arguments')
    req_group.add_argument("--manifest-file", type=str, required=True, help="Path to the QIIME 2 manifest file (TSV).")
    req_group.add_argument("--output-dir", type=str, required=True, help="Directory to save all output files.")
    req_group.add_argument("--fwd-primer-seq", type=str, required=True, help="Forward primer sequence.")
    req_group.add_argument("--rev-primer-seq", type=str, required=True, help="Reverse primer sequence.")

    # Workflow configuration
    work_group = parser.add_argument_group('Workflow Configuration')
    work_group.add_argument("--library-layout", type=str, default=DEFAULT_LIBRARY_LAYOUT, choices=['single', 'paired'], help="Library layout.")
    work_group.add_argument("--n-threads", type=int, default=DEFAULT_N_THREADS, help="Number of CPU threads to use.")
    work_group.add_argument("--minimum-length", type=int, default=DEFAULT_MINIMUM_LENGTH, help="Cutadapt: Minimum sequence length to keep.")
    work_group.add_argument("--min-reads", type=int, default=DEFAULT_MIN_READS, help="Minimum number of reads to keep a sample after trimming.")

    # Denoising parameters
    denoise_group = parser.add_argument_group('Denoising Parameters')
    denoise_group.add_argument("--denoise-algorithm", type=str, default=DEFAULT_DENOISE_ALGORITHM, choices=['dada2', 'deblur'], help="Denoising algorithm.")
    denoise_group.add_argument("--trunc-len-f", type=int, default=DEFAULT_TRUNC_LEN_F, help="DADA2: Forward read truncation length.")
    denoise_group.add_argument("--trunc-len-r", type=int, default=DEFAULT_TRUNC_LEN_R, help="DADA2: Reverse read truncation length.")
    denoise_group.add_argument("--trim-left-f", type=int, default=DEFAULT_TRIM_LEFT_F, help="DADA2: Nucleotides to trim from 5' end of forward reads.")
    denoise_group.add_argument("--trim-left-r", type=int, default=DEFAULT_TRIM_LEFT_R, help="DADA2: Nucleotides to trim from 5' end of reverse reads.")
    denoise_group.add_argument("--max-ee-f", type=float, default=DEFAULT_MAX_EE_F, help="DADA2: Max expected errors for forward reads.")
    denoise_group.add_argument("--max-ee-r", type=float, default=DEFAULT_MAX_EE_R, help="DADA2: Max expected errors for reverse reads.")
    denoise_group.add_argument("--trim-length", type=int, default=DEFAULT_TRIM_LENGTH, help="Deblur: Sequence trim length.")
    denoise_group.add_argument("--chimera-method", type=str, default=DEFAULT_CHIMERA_METHOD, help="DADA2: Chimera removal method.")

    # Taxonomy parameters
    tax_group = parser.add_argument_group('Taxonomy Parameters')
    tax_group.add_argument("--classifier-dir", type=str, default=DEFAULT_CLASSIFIER_DIR, help="Directory containing the taxonomy classifier files.")
    tax_group.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER, help="Name of the classifier artifact.")
    tax_group.add_argument("--classify-method", type=str, default=DEFAULT_CLASSIFY_METHOD, choices=['sklearn', 'blast'], help="Taxonomy classification method.")
    tax_group.add_argument("--taxonomy-level", type=int, default=DEFAULT_TAXONOMY_LEVEL, help="Taxonomic level to collapse the feature table to.")
    tax_group.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Taxonomy: Confidence threshold for sklearn classifier.")

    args = parser.parse_args()
    run_workflow(args)


if __name__ == "__main__":
    main()