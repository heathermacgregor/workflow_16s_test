"""
Smart plotting limit utilities to handle large result sets.

Automatically limits plot generation based on result size while preserving
the most important results.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("workflow_16s")


class PlotLimiter:
    """Intelligently limit plot generation for large result sets."""
    
    def __init__(
        self,
        max_plots: int = 1000,
        top_n: int = 500,
        random_n: int = 300
    ):
        """
        Initialize plot limiter.
        
        Parameters
        ----------
        max_plots : int
            Maximum total plots to generate
        top_n : int
            Number of top results to always include
        random_n : int
            Number of random results to include after top results
        """
        self.max_plots = max_plots
        self.top_n = top_n
        self.random_n = random_n
        self.limited_count = 0
        self.total_skipped = 0
        
    def should_limit(self, result_count: int) -> bool:
        """Check if limiting should be applied."""
        return result_count > self.max_plots
    
    def limit_results(
        self,
        results_df: pd.DataFrame,
        sort_column: str = 'p_value',
        ascending: bool = True,
        name: str = "results"
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Limit results dataframe for plotting.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results to potentially limit
        sort_column : str
            Column to use for sorting (e.g., 'p_value', 'importance')
        ascending : bool
            Sort direction (True for p-values, False for scores)
        name : str
            Name of results for logging
            
        Returns
        -------
        pd.DataFrame
            Limited results
        bool
            True if limiting was applied
        """
        if not self.should_limit(len(results_df)):
            return results_df, False
            
        # Sort by importance
        sorted_df = results_df.sort_values(sort_column, ascending=ascending)
        
        # Take top results
        top_results = sorted_df.head(self.top_n)
        
        # Take random sample from remaining
        remaining = sorted_df.iloc[self.top_n:]
        if len(remaining) > self.random_n:
            random_results = remaining.sample(
                n=self.random_n,
                random_state=42
            )
        else:
            random_results = remaining
            
        # Combine
        limited_df = pd.concat([top_results, random_results])
        
        skipped = len(results_df) - len(limited_df)
        self.limited_count += 1
        self.total_skipped += skipped
        
        logger.info(
            f"📊 {name}: Limiting {len(results_df)} → {len(limited_df)} plots "
            f"(top {len(top_results)} + random {len(random_results)}, "
            f"skipped {skipped})"
        )
        
        return limited_df, True
    
    def get_summary(self) -> str:
        """Get summary of limiting actions."""
        if self.limited_count == 0:
            return "No plot limiting applied"
            
        return (
            f"Plot limiting applied {self.limited_count} times, "
            f"skipped {self.total_skipped} total plots"
        )


# Global limiter instance
_limiter = PlotLimiter()


def get_plot_limiter() -> PlotLimiter:
    """Get the global plot limiter instance."""
    return _limiter


def limit_for_plotting(
    results_df: pd.DataFrame,
    sort_column: str = 'p_value',
    ascending: bool = True,
    max_plots: int = 1000,
    name: str = "results"
) -> Tuple[pd.DataFrame, bool]:
    """
    Convenience function to limit results for plotting.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results to potentially limit
    sort_column : str
        Column to use for sorting
    ascending : bool
        Sort direction
    max_plots : int
        Maximum plots to generate
    name : str
        Name for logging
        
    Returns
    -------
    pd.DataFrame
        Limited results
    bool
        True if limiting was applied
    """
    limiter = get_plot_limiter()
    limiter.max_plots = max_plots
    return limiter.limit_results(results_df, sort_column, ascending, name)
