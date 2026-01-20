# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third-Party Imports
import qiime2
from qiime2 import Artifact

# Local Imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from workflow_16s import constants

# ==================================== FUNCTIONS ===================================== #

def output_files_exist(
    file_dir: Union[str, Path], 
    prefixes: List[str], 
    extension: str = "qza"
) -> bool:
    """Check if all expected output files exist in a directory.
    
    Args:
        file_dir:  Directory containing the files.
        prefixes:  List of file prefixes to check.
        extension: File extension to check for. Defaults to 'qza'.

    Returns:
        True if all files exist, False otherwise.
    """
    file_dir = Path(file_dir)  # Ensure file_dir is a Path object
    missing_files = [
        file_dir / f"{prefix}.{extension}"
        for prefix in prefixes
        if not (file_dir / f"{prefix}.{extension}").exists()
    ]
    return not missing_files


# Print context when importing and exporting QIIME artifacts
def construct_file_path(
    file_dir: Union[str, Path], 
    prefix: str, 
    extension: str = "qza"
) -> Path:
    """Constructs a file path with a given prefix and extension."""
    return Path(file_dir) / f"{prefix}.{extension}"


def load_with_print(
    file_dir: Union[str, Path], 
    prefix: str, 
    suffix: str = "qza", 
    n: int = constants.DEFAULT_PADDING_WIDTH
) -> Artifact:
    """Constructs a file path and loads a QIIME2 artifact from it."""
    file_path = str(construct_file_path(file_dir, prefix, suffix))
    artifact = Artifact.load(file_path)
    print(f"{'  📥 Loaded from':{n}.{n}}: {file_path}")
    return artifact


def save_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = constants.DEFAULT_PADDING_WIDTH,
) -> None:
    """Saves a QIIME2 Artifact to a file and prints a confirmation message.

    Args:
        artifact:  QIIME2 artifact to save.
        file_dir:  Directory to save the artifact in.
        prefix:    Filename prefix for the artifact.
        suffix:    File extension to use. Defaults to 'qza'.
        n:         Padding width for console output alignment. Defaults to 
                   constants.DEFAULT_PADDING_WIDTH.
    """
    file_path = str(construct_file_path(file_dir, prefix, suffix))
    artifact.save(file_path)
    print(f"{'  📤 Saved to':{n}.{n}}: {file_path}")


def export_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = constants.DEFAULT_PADDING_WIDTH,
) -> None:
    """Exports a QIIME2 Artifact's data to a directory and prints confirmation.
    Creates target directory if needed. Exported data format depends on artifact type.
    
    Args:
        artifact:  QIIME2 artifact to export.
        file_dir:  Base directory for exported data.
        prefix:    Directory name prefix for exported data.
        suffix:    Unused in path but required for filename parsing. Defaults to 'qza'.
        n:         Padding width for console output alignment. Defaults to 
                   constants.DEFAULT_PADDING_WIDTH.
    """
    file_path = construct_file_path(file_dir, prefix, suffix)
    dir_path = str(file_path).strip(suffix).strip(".")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    artifact.export_data(str(dir_path))
    print(f"{'  📤 Exported to':{n}.{n}}: {dir_path}")


def save_and_export_with_print(
    artifact: Artifact,
    file_dir: Union[str, Path],
    prefix: str,
    suffix: str = "qza",
    n: int = constants.DEFAULT_PADDING_WIDTH,
) -> None:
    """Convenience function that both saves artifact and exports its data with confirmation.
    Combines save_with_print() and export_with_print() functionality.
    
    Args:
        artifact:  QIIME2 artifact to process.
        file_dir:  Directory for saved artifact and exported data.
        prefix:    Prefix for artifact filename and export directory.
        suffix:    File extension for saved artifact. Defaults to 'qza'.
        n:         Padding width for console output alignment. Defaults to 
                   constants.DEFAULT_PADDING_WIDTH.
    """
    save_with_print(artifact, file_dir, prefix, suffix, n)
    export_with_print(artifact, file_dir, prefix, suffix, n)
