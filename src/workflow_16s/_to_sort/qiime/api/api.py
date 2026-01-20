# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Generator, List, Tuple, Union

# Third-Party Imports
from Bio.Seq import Seq
import pandas as pd
import qiime2
from qiime2 import Artifact
from qiime2.plugins import demux, taxa
from qiime2.plugins.cutadapt.methods import trim_paired, trim_single
from qiime2.plugins.dada2.methods import denoise_paired, denoise_pyro, denoise_single
from qiime2.plugins.deblur.methods import denoise_16S
from qiime2.plugins.demux.visualizers import summarize as summarize_demux
from qiime2.plugins.feature_classifier.methods import classify_sklearn
from qiime2.plugins.feature_classifier.pipelines import classify_consensus_blast
from qiime2.plugins.phylogeny.pipelines import align_to_tree_mafft_fasttree

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from api.api_io import (
    construct_file_path,
    load_with_print,
    output_files_exist,
    save_and_export_with_print,
    save_with_print,
)

project_root = str(Path(__file__).resolve().parents[2])
print(f"Added to path: {project_root}") # Check this points to the correct folder
sys.path.append(project_root)
print(sys.path) # Ensure the path appears in the list

# ========================== INITIALIZATION & CONFIGURATION ========================== #

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N_THREADS = 32
DEFAULT_SAVE_INTERMEDIATES = True
DEFAULT_LIBRARY_LAYOUT = "paired"
DEFAULT_MINIMUM_LENGTH = 100
DEFAULT_MIN_READS = 1000
DEFAULT_TRUNC_LEN_F = 250
DEFAULT_TRUNC_LEN_R = 250
DEFAULT_TRIM_LENGTH = 250
DEFAULT_TRIM_LEFT_F = 0
DEFAULT_TRIM_LEFT_R = 0
DEFAULT_TRUNC_Q = 2
DEFAULT_MAX_EE = 2
DEFAULT_MAX_EE_F = 2
DEFAULT_MAX_EE_R = 10
DEFAULT_CHIMERA_METHOD = "consensus"
DEFAULT_DENOISE_ALGORITHM = "DADA2"
DEFAULT_CLASSIFIER_DIR = (
    "/usr2/people/macgregor/mtv_project/references/"
    "hrm_workflow/classifier/silva-138-99-515-806"
)
DEFAULT_CLASSIFIER = "silva-138-99-515-806"
DEFAULT_CLASSIFY_METHOD = "sklearn"
DEFAULT_MAXACCEPTS = 50
DEFAULT_PERC_IDENTITY = 0.99
DEFAULT_QUERY_COV = 0.9
DEFAULT_CONFIDENCE = 0.7
DEFAULT_MSA_N_SEQUENCES = 1000000

# ==================================== FUNCTIONS ===================================== #

def capture_command_output() -> Generator[io.StringIO, None, None]:
    """Context manager for capturing command line output"""
    buffer = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = buffer
    finally:
        sys.stdout = original_stdout
        buffer.seek(0)


def get_cli_output(function: callable, *args, **kwargs) -> Tuple[Any, str]:
    """
    Execute a function while capturing its CLI output.

    Args:
        function: The function to execute
        args:     Positional arguments for the function
        kwargs:   Keyword arguments for the function

    Returns:
        tuple:    (function_result, cli_output_string)
    """
    with capture_command_output() as buffer:
        result = function(*args, **kwargs)
        output = buffer.getvalue()
    return result, output

# ================================= QIIME2 FUNCTIONS ================================ #

def import_seqs_from_manifest(
    output_dir: Union[str, Path],
    manifest_tsv: Union[str, Path],
    library_layout: str = DEFAULT_LIBRARY_LAYOUT,
) -> Artifact:
    """
    Import sequence data from a manifest file and generate quality control 
    visualizations.

    Processes sequencing data from either single-end or paired-end experiments, 
    validates input parameters, imports data into QIIME2 artifacts, and generates 
    demultiplexing summary statistics.

    Args:
        output_dir:     Directory path where output files (QZA/QZV artifacts) will be 
                        saved
        manifest_tsv:   Path to manifest TSV file containing sample-to-filepath mappings
        library_layout: Sequencing library layout, either 'paired' or 'single' 
                        (case-insensitive), defaults to 'paired'

    Returns:
        seqs:           QIIME2 artifact containing imported sequence data
    """
    if not isinstance(library_layout, str):
        raise TypeError(
            f"❓ Library layout type '{type(library_layout).__name__}' not recognized. "
            f"Expected 'str'."
        )

    library_layout = library_layout.lower()
    if library_layout not in {"paired", "single"}:
        raise ValueError(
            f"❓ Library layout '{layout}' not recognized. "
            f"Expected 'single' or 'paired'."
        )

    layout_config = {
        "single": (
            "SampleData[SequencesWithQuality]",
            "SingleEndFastqManifestPhred33V2",
        ),
        "paired": (
            "SampleData[PairedEndSequencesWithQuality]",
            "PairedEndFastqManifestPhred33V2",
        ),
    }

    import_format, view_type = layout_config[library_layout]
    seqs = qiime2.Artifact.import_data(import_format, manifest_tsv, view_type)

    save_with_print(seqs, output_dir, "seqs", "qza")
    seqs_summary = summarize_demux(data=seqs).visualization
    save_and_export_with_print(seqs_summary, output_dir, "demux-stats", "qzv")

    return seqs


def trim_sequences(
    output_dir: Union[str, Path],
    seqs: Artifact,
    library_layout: str,
    fwd_primer_seq: str,
    rev_primer_seq: str,
    minimum_length: int = DEFAULT_MINIMUM_LENGTH,
    n_cores: int = DEFAULT_N_THREADS,
    save_intermediates: bool = DEFAULT_SAVE_INTERMEDIATES,
) -> Artifact:
    """
    Trim sequences with automatic workflow fallback on failure.

    Args:
        output_dir:         Directory path where output files (QZA/QZV artifacts) will 
                            be saved
        seqs:               Input sequences artifact
        library_layout:     Sequencing layout ('paired' or 'single')
        fwd_primer_seq:     Forward primer sequence
        rev_primer_seq:     Reverse primer sequence
        minimum_length:     Minimum sequence length after trimming
        n_cores:            Number of CPU cores to use
        save_intermediates: Whether to save intermediate files

    Returns:
        trimmed_seqs:       QIIME2 artifact containing trimmed sequence data.
    """
    rev_primer_rc = str(Seq(rev_primer_seq).reverse_complement())
    layout = library_layout.lower()

    if "single" in layout:
        workflows = [
            (
                trim_single,
                "Single-end primary trimming",
                {"front": [fwd_primer_seq]},
            )
        ]
        fallback = (
            trim_paired,
            "Paired-end fallback trimming",
            {"front_f": [fwd_primer_seq], "front_r": [rev_primer_rc]},
        )
    elif "paired" in layout:
        workflows = [
            (
                trim_paired,
                "Paired-end primary trimming",
                {"front_f": [fwd_primer_seq], "front_r": [rev_primer_rc]},
            )
        ]
        fallback = (
            trim_single,
            "Single-end fallback trimming",
            {"front": [fwd_primer_seq]},
        )
    else:
        raise ValueError(
            f"❓ Library layout '{library_layout}' not recognized. "
            f"Expected 'single' or 'paired'."
        )

    workflows.append(fallback)
    errors: List[str] = []
    trimmed_seqs = None

    for method, name, params in workflows:
        try:
            print(f"Attempting {name} workflow")
            trimmed_seqs = method(
                demultiplexed_sequences=seqs,
                minimum_length=minimum_length,
                cores=n_cores,
                **params,
            ).trimmed_sequences
            break
        except Exception as e:
            errors.append(f"{name} failed: {e}")
            trimmed_seqs = None

    if not trimmed_seqs:
        raise RuntimeError("⚠️ All trimming workflows failed:\n" + "\n".join(errors))

    if save_intermediates:
        save_with_print(trimmed_seqs, output_dir, "trimmed-seqs", "qza")
        summary = summarize_demux(data=trimmed_seqs).visualization
        save_and_export_with_print(
            summary, output_dir, "trimmed-seqs_demux-stats", "qzv"
        )

    return trimmed_seqs


def filter_samples_for_denoising(
    seqs: Artifact,
    counts_file: Union[str, Path],
    min_reads: int = DEFAULT_MIN_READS,
) -> Artifact:
    """
    Filter a QIIME2 demux artifact to retain samples with sufficient read counts.

    Args:
        seqs:        Demultiplexed sequence artifact
        counts_file: TSV file with sequence counts. For paired-end reads, this should
                     include 'forward sequence count' and 'reverse sequence count'
                     columns. For single-end reads, it should include a 'sequence count'
                     column.
        min_reads:   Minimum reads required to retain a sample

    Returns:
        Filtered demultiplexed sequence artifact
    """
    counts_path = Path(counts_file)
    df = pd.read_csv(counts_path, sep="\t", index_col=0)
    
    # Determine if paired-end or single-end based on columns
    if 'forward sequence count' in df.columns and 'reverse sequence count' in df.columns:
        # Paired-end: both forward and reverse must meet min_reads
        valid = df[
            (df["forward sequence count"] >= min_reads) &
            (df["reverse sequence count"] >= min_reads)
        ].index.tolist()
    elif 'forward sequence count' in df.columns:
        # Single-end: only sequence count must meet min_reads
        valid = df[df["forward sequence count"] >= min_reads].index.tolist()
    else:
        raise ValueError(
            "⚠️ Counts file must contain either 'forward sequence count' and "
            "'reverse sequence count' columns (paired-end) "
            "or 'forward sequence count' column (single-end)."
        )
    
    keep_tsv = counts_path.parent / "keep_samples.tsv"
    with open(keep_tsv, "w") as f:
        f.write("#SampleID\tDescription\n")
        for sample in valid:
            f.write(f"{sample}\tvalid\n")
    
    metadata = qiime2.Metadata.load(str(keep_tsv))
    return demux.methods.filter_samples(demux=seqs, metadata=metadata).filtered_demux


def denoise_sequences(
    output_dir: Union[str, Path],
    seqs: Artifact,
    library_layout: str,
    instrument_platform: str,
    trunc_len_f: int = DEFAULT_TRUNC_LEN_F,
    trunc_len_r: int = DEFAULT_TRUNC_LEN_R,
    trim_length: int = DEFAULT_TRIM_LENGTH,
    chimera_method: str = DEFAULT_CHIMERA_METHOD,
    denoise_algorithm: str = DEFAULT_DENOISE_ALGORITHM,
    n_threads: int = DEFAULT_N_THREADS / 2,
    trim_left_f: int = DEFAULT_TRIM_LEFT_F,
    trim_left_r: int = DEFAULT_TRIM_LEFT_R,
    trunc_q: int = DEFAULT_TRUNC_Q,
    max_ee: Union[float, int] = DEFAULT_MAX_EE,
    max_ee_f: Union[float, int] = DEFAULT_MAX_EE_F,
    max_ee_r: Union[float, int] = DEFAULT_MAX_EE_R,
) -> Tuple[Artifact, Artifact, Artifact]:
    """
    Denoise sequences using DADA2 or Deblur depending on platform and layout.
    
    Args:
        output_dir:          Directory path where output files (QZA/QZV artifacts) will 
                             be saved.
        seqs:                The Artifact containing the demultiplexed sequences to be 
                             denoised.
        library_layout:      Either 'single' or 'paired'. Determines how reads are 
                             processed.
        instrument_platform: The sequencing platform used (e.g., ILLUMINA, MISEQ). 
                             Determines whether to use DADA2 or Deblur.
        trunc_len_f:         Position at which forward read sequences should be 
                             truncated due to decrease in quality. (3' end)
        trunc_len_r:         Position at which reverse read sequences should be 
                             truncated due to decrease in quality. (3' end)
        trim_length:         Sequence trim length.
        chimera_method:      The method used to remove chimeras.
        denoise_algorithm:   The denoising algorithm to use. Either 'DADA2' or 'Deblur'.
        n_threads:           The number of threads to use for multithreaded processing.
        trim_left_f:         Position at which forward read sequences should be trimmed 
                             due to low quality. (5' end)
        trim_left_r:         Position at which reverse read sequences should be trimmed 
                             due to low quality. (5' end)
        trunc_q:             Reads are truncated at the first instance of a quality 
                             score less than or equal to this value. If the resulting 
                             read is then shorter than `trunc-len-f` or `trunc-len-r` 
                             (depending on the direction of the read) it is discarded.
        max_ee:              Reads with number of expected errors higher than this 
                             value will be discarded.
        max_ee_f:            Forward reads with number of expected errors higher than 
                             this value will be discarded.
        max_ee_r:            Reverse reads with number of expected errors higher than 
                             this value will be discarded.
   
    Returns:
        Tuple containing:        
            rep_seqs:        The resulting feature sequences. Each feature in the 
                             feature table will be represented by exactly one sequence.
            table:           The resulting feature table.
            stats:           Per-sample denoising stats.
    """

    algorithm = denoise_algorithm.lower()
    platform = instrument_platform.lower()
    
    if "illumina" in platform:
        if algorithm == "dada2":
            if "single" in library_layout.lower():
                # https://docs.qiime2.org/2024.10/plugins/available/dada2/denoise-single/
                func = denoise_single
                args = {
                    "demultiplexed_seqs": seqs,
                    "trunc_len": trunc_len_f,
                    "chimera_method": chimera_method,
                    "trunc_q": trunc_q,
                    "max_ee": max_ee,
                    "trim_left": trim_left_f,
                    "n_reads_learn": 50000,
                    "hashed_feature_ids": True,
                    "n_threads": n_threads,
                }
            else:
                # https://docs.qiime2.org/2024.10/plugins/available/dada2/denoise-paired/
                func = denoise_paired
                args = {
                    "demultiplexed_seqs": seqs,
                    "trunc_len_f": trunc_len_f,
                    "trunc_len_r": trunc_len_r,
                    "chimera_method": chimera_method,
                    "trunc_q": trunc_q,
                    "max_ee_f": max_ee_f,
                    "max_ee_r": max_ee_r,
                    "trim_left_f": trim_left_f,
                    "trim_left_r": trim_left_r,
                    "n_reads_learn": 50000,
                    "hashed_feature_ids": True,
                    "n_threads": n_threads,
                }
        elif algorithm == "deblur":
            func = denoise_16S
            # https://docs.qiime2.org/2024.10/plugins/available/deblur/denoise-16S/
            args = {
                "demultiplexed_seqs": seqs,
                "trim_length": trim_length,
                "hashed_feature_ids": True,
                "sample_stats": True,
                "jobs_to_start": n_threads,
            }
        else:
            raise ValueError(f"❓ Denoise algorithm '{algorithm}' not recognized.")
            
    elif "454" in platform:
        func = denoise_pyro
        # https://docs.qiime2.org/2024.10/plugins/available/dada2/denoise-pyro/
        args = {
            "demultiplexed_seqs": seqs,
            "trim_left": 0,
            "trunc_len": trunc_len_f,
            "hashed_feature_ids": True,
            "n_threads": n_threads,
        }
        
    else:
        raise ValueError(f"❓ Platform '{platform}' not recognized.")

    results = func(**args)
    rep_seqs = results.representative_sequences
    table = results.table
    stats = (
        results.stats
        if hasattr(results, 'stats')
        else results.denoising_stats
    )

    save_and_export_with_print(rep_seqs, output_dir, "rep-seqs", "qza")
    save_and_export_with_print(table, output_dir, "table", "qza")
    save_and_export_with_print(stats, output_dir, "stats", "qza")

    return rep_seqs, table, stats


def classify_taxonomy(
    output_dir: Union[str, Path],
    rep_seqs: Artifact,
    classifier_dir: Union[str, Path] = DEFAULT_CLASSIFIER_DIR,
    classifier: str = DEFAULT_CLASSIFIER,
    classify_method: str = DEFAULT_CLASSIFY_METHOD,
    maxaccepts: int = DEFAULT_MAXACCEPTS,
    perc_identity: float = DEFAULT_PERC_IDENTITY,
    query_cov: float = DEFAULT_QUERY_COV,
    confidence: float = DEFAULT_CONFIDENCE,
    n_threads: int = DEFAULT_N_THREADS,
) -> Tuple[Artifact, Any]:
    """
    Classify sequences taxonomically using sklearn or BLAST-based methods.
    
    Args:
        output_dir:        Directory path where output files (QZA/QZV artifacts) will 
                           be saved.
        rep_seqs:          The representative sequences to be classified.
        classifier_dir:    Directory where the classifier files are stored.
        classifier:        Name of the classifier to use (e.g., 'silva-138-99-515-806').
        classify_method:   The classification method to use. Options include:
                           'sklearn' (naive Bayes), 'blast', or 'vsearch'.
        maxaccepts:        Maximum number of hits to keep for BLAST/vsearch alignment-
                           based methods.
        perc_identity:     Minimum percent identity required for BLAST/vsearch-based 
                           classification.
        query_cov:         Minimum query coverage for an alignment-based match to be 
                           considered valid.
        confidence:        Confidence threshold for sklearn-based classification. 
                           Lower values increase sensitivity but reduce specificity.
        n_threads:         Number of threads to use for parallel processing.
    
    Returns:
        Tuple containing:
            taxonomy:       The resulting taxonomy classification artifact.
            search_results: Raw search results from the classification process.
    """

    out_dir = Path(output_dir) / classifier
    out_dir.mkdir(parents=True, exist_ok=True)

    if classify_method == "sklearn":
        clf = load_with_print(
            Path(classifier_dir), f"{classifier}-classifier", "qza"
        )
        result = classify_sklearn(
            reads=rep_seqs,
            classifier=clf,
            confidence=confidence,
            n_jobs=n_threads,
        )
    else:
        try:
            ref_tax = load_with_print(
                Path(classifier_dir), f"{classifier}-tax", "qza"
            )
        except FileNotFoundError:
            ref_tax = load_with_print(
                Path(classifier_dir), "silva-138-99-tax-515-806", "qza"
            )
        args = {"query": rep_seqs, "reference_taxonomy": ref_tax,
                "perc_identity": perc_identity, "query_cov": query_cov}
        if classify_method == "reads":
            ref_reads = load_with_print(
                Path(classifier_dir), "silva-138-99-seqs-515-806", "qza"
            )
            args["reference_reads"] = ref_reads
        elif classify_method == "db":
            blastdb = Artifact.import_data(
                "BLASTDB", Path(classifier_dir) / "blastdb"
            )
            args["blastdb"] = blastdb
        else:
            raise ValueError(
                f"Classification method '{classify_method}' not recognized."
            )
        result = classify_consensus_blast(**args)

    taxonomy = result.classification
    save_and_export_with_print(taxonomy, out_dir, "taxonomy")
    
    try:
        search_results = result.search_results
        save_and_export_with_print(search_results, out_dir, "search-results")
    except AttributeError:
        search_results = None

    return taxonomy, search_results


def multiple_sequence_alignment(
    output_dir: Union[str, Path],
    seqs: Artifact,
    n_sequences: int = DEFAULT_MSA_N_SEQUENCES,
    n_threads: int = DEFAULT_N_THREADS,
) -> Tuple[Artifact, Artifact, Artifact]:
    """
    Align sequences and construct phylogenetic tree with MAFFT and FastTree.
    
    Args:
        output_dir:     Directory path where output files (QZA/QZV artifacts) will 
                        be saved.
        seqs:           The representative sequences (Artifact) to be aligned and used 
                        for phylogenetic tree construction.
        n_sequences:    The number of sequences to include in the multiple sequence 
                        alignment. Affects runtime and memory usage.
        n_threads:      The number of threads to use for multithreaded processing 
                        during alignment and tree construction.
    
    Returns:
        Tuple containing:
            alignment:     The resulting multiple sequence alignment artifact.
            tree:          The unrooted phylogenetic tree artifact constructed from the 
                           aligned sequences.
            rooted_tree:   The midpoint-rooted phylogenetic tree artifact.
    """

    out_dir = Path(output_dir) / "mafft_fasttree"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Aligning sequences with MAFFT and FastTree...")
    args = {"sequences": seqs, "n_threads": n_threads}
    if n_sequences > 1000000:
        args["parttree"] = True

    results = align_to_tree_mafft_fasttree(**args)
    alignment = results.alignment
    tree = results.tree
    rooted_tree = results.rooted_tree

    save_and_export_with_print(alignment, out_dir, "aligned_rep_seqs")
    save_and_export_with_print(tree, out_dir, "unrooted_tree")
    save_and_export_with_print(rooted_tree, out_dir, "rooted_tree")

    return alignment, tree, rooted_tree


def collapse_to_genus(
    output_dir: Union[str, Path],
    table: Artifact,
    taxonomy: Artifact,
) -> Artifact:
    """
    Collapse a QIIME2 feature table to the genus level (taxonomic level 6).

    Args:
        output_dir:      Directory where the collapsed table will be saved and exported.
        table:           A QIIME2 FeatureTable[Frequency] artifact containing the 
                         feature table.
        taxonomy:        A QIIME2 FeatureData[Taxonomy] artifact used to map features 
                         to taxonomy.

    Returns:
        collapsed_table: The collapsed FeatureTable[Frequency] artifact at genus level.
    """
    collapsed_table = taxa.actions.collapse(
        table=table,
        taxonomy=taxonomy,
        level=6
    ).collapsed_table
    
    save_and_export_with_print(collapsed_table, output_dir, "table_6")
    
    return collapsed_table
