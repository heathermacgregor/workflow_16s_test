# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

# Third-Party Imports
import pandas as pd

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# ================================ CUSTOM TMP CONFIG ================================= #

#import workflow_16s.custom_tmp_config

# ========================== INITIALIZATION & CONFIGURATION ========================== #

# Suppress warnings
warnings.filterwarnings("ignore")

# ==================================== FUNCTIONS ===================================== #

def create_dir(dir_path: Union[str, Path]) -> None:
    """Create directory structure if it doesn't exist.
    
    Args:
        dir_path: Path to directory to create
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {dir_path}")
        

def get_average_lengths(
    forward_file: Union[str, Path],
    reverse_file: Optional[Union[str, Path]] = None,
) -> Tuple[float, float]:
    """Calculate average sequence lengths from QIIME2 seven-number summary files.
    
    Args:
        forward_file:   Path to forward read summary file
        reverse_file:   Optional path to reverse read summary file
    
    Returns:
        Tuple containing:
            avg_forward: Average forward read length (float)
            avg_reverse: Average reverse read length (0.0 if single-end)
    """
    def _calculate_avg_length(file_path: Union[str, Path]) -> float:
        """Calculate average length from a single summary file."""
        file_path = Path(file_path)
        if not file_path.exists():
            return 0.0
        df = pd.read_csv(file_path, sep="\t", header=None)
        count_row = df[df.iloc[:, 0] == "count"].iloc[0, 1:].astype(int)
        total_reads = count_row.iloc[0]
        return count_row.sum() / total_reads if total_reads > 0 else 0.0

    avg_forward = _calculate_avg_length(forward_file)
    avg_reverse = _calculate_avg_length(reverse_file) if reverse_file else 0.0
    return avg_forward, avg_reverse


def get_truncation_lengths(
    forward_file: Union[str, Path],
    reverse_file: Optional[Union[str, Path]] = None,
    quality_threshold: int = 25,
) -> Tuple[int, int]:
    """Determine optimal truncation positions based on quality scores.
    
    Args:
        forward_file:      Path to forward read summary file
        reverse_file:      Optional path to reverse read summary file
        quality_threshold: Quality score cutoff (default: 25)
    
    Returns:
        Tuple containing:
            trunc_forward: Forward read truncation position
            trunc_reverse: Reverse read truncation position (0 if single-end)
    """
    def _find_trunc_pos(file_path: Union[str, Path]) -> int:
        """Identify truncation position from quality metrics."""
        file_path = Path(file_path)
        if not file_path.exists():
            return 0
        df = pd.read_csv(file_path, sep="\t", header=None)
        median_qualities = df[df.iloc[:, 0] == "50%"].iloc[0, 1:].astype(float)
        return next(
            (i for i, q in enumerate(median_qualities) if q < quality_threshold),
            len(median_qualities)
        )

    trunc_forward = _find_trunc_pos(forward_file)
    trunc_reverse = _find_trunc_pos(reverse_file) if reverse_file else 0
    return trunc_forward, trunc_reverse
  
