# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import argparse
import os, sys
from pathlib import Path

# Local Imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from per_dataset import Dataset, WorkflowRunner

# ==================================== FUNCTIONS ===================================== #

def validate_file(path: str) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {path} does not exist")
    return path


def validate_dir(path: str) -> Path:
    """Validate/create a directory and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_or_validate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} exists but is not a directory.")
    return path


def parse_library_layout(value):
    value = value.strip().lower()
    if value in ["single", "s"]:
        return "single"
    elif value in ["paired", "p"]:
        return "paired"
    else:
        raise argparse.ArgumentTypeError(
            "Library layout must be one of: single, paired, s, p (case-insensitive)."
        )


def parse_instrument_platform(value):
    value = value.strip().lower()
    valid_platforms = ['illumina', '454', 'iontorrent', 'oxfordnanopore']
    if value not in valid_platforms:
        raise argparse.ArgumentTypeError(
            "Instrument platform must be one of: ILLUMINA, 454, IONTORRENT, OXFORDNANOPORE (case-insensitive)."
        )
    return valid_platforms[value]


def main(args):
    """Main workflow execution function with error handling."""
    try:
        print("Starting 16S processing workflow")
        completed = WorkflowRunner(vars(args)).execute()
        print(f"Workflow completed: {completed}")
        return 0
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        return 1


if __name__ == "__main__":
    # Organize arguments into logical groups
    parser = argparse.ArgumentParser(
        prog='QIIME2PerDataset',
        description='QIIME 2 per-dataset amplicon data processing script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='QIIME 2 per-dataset amplicon data processing script.'
    )
    
    # Input/Output Parameters
    io_group = parser.add_argument_group("Input/Output Parameters")
    io_group.add_argument(
        "--manifest_tsv",
        type=validate_file, # Check that the file exists and is readable
        required=True,
        help="Path to the input manifest TSV file mapping sample IDs to FASTQ paths",
    )
    io_group.add_argument(
        "--metadata_tsv",
        type=validate_file, # Check that the file exists and is readable
        required=True,
        help="Path to the input sample metadata TSV file",
    )
    io_group.add_argument(
        "--qiime_dir",
        type=create_or_validate_dir,
        required=True,
        help="Path to the output directory for QIIME2 processing results",
    )

    # Sequencing Parameters
    seq_group = parser.add_argument_group("Sequencing Parameters")
    seq_group.add_argument(
        "--library_layout",
        required=True,
        type=parse_library_layout,
        help="Sequencing library layout: 'single' or 'paired' (case-insensitive, abbreviations 's' and 'p' allowed)",
    )
    seq_group.add_argument(
        "--instrument_platform",
        required=True,
        help="Sequencing platform used (e.g., Illumina, 454, Ion_Torrent, Oxford_Nanopore — case-insensitive)",
    )
    
    # Primer Parameters
    primer_group = parser.add_argument_group("Primer Parameters")
    primer_group.add_argument(
        "--fwd_primer", 
        required=True, 
        help="Forward primer sequence (5' to 3')"
    )
    primer_group.add_argument(
        "--rev_primer", 
        required=True, 
        help="Reverse primer sequence (5' to 3')"
    )

    # Classification Parameters
    class_group = parser.add_argument_group("Classification Parameters")
    class_group.add_argument(
        "--classifier",
        required=True,
        help="Name of the pre-trained QIIME2 classifier artifact (.qza)",
    )
    class_group.add_argument(
        "--classifier_dir",
        type=validate_dir,
        required=True,
        help="Path to the directory containing classifier dependencies",
    )
    class_group.add_argument(
        "--classify_method",
        default="sklearn",
        choices=["naive-bayes", "vsearch", "sklearn"],
        help="Taxonomy classification method",
    )
    class_group.add_argument(
        "--retain_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for retaining taxonomy assignments",
    )

    # Processing Parameters
    proc_group = parser.add_argument_group("Processing Parameters")
    proc_group.add_argument(
        "--denoise_algorithm",
        required=True,
        default="DADA2",
        choices=["DADA2", "Deblur"],
        help="Denoising algorithm to use",
    )
    proc_group.add_argument(
        "--trim_sequences",
        action="store_true",
        help="Turn on sequence trimming during processing",
    )
    proc_group.add_argument(
        "--chimera_method",
        default="consensus",
        help="Method for chimera detection",
    )
    proc_group.add_argument(
        "--hard_rerun",
        action="store_true",
        help="Turn on forced reprocessing of all steps",
    )

    args = parser.parse_args()
    
    # Execute main workflow with exit code
    sys.exit(main(args))
