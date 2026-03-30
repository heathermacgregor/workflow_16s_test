# downstream/temporal_dynamics.py

"""
Temporal Dynamics: Analyze microbiome stability, successional patterns, and resilience.

Answers:
1. Are microbiomes stable or oscillating over time?
2. Do they follow predictable successional sequences?
3. How resilient are they to environmental shocks?

Methods:
- Temporal autocorrelation (ACF, partial ACF)
- Turnover rates (taxa appearing/disappearing)
- Resilience metrics (recovery time after disturbance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import warnings

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def calculate_temporal_turnover(
    otu_table_time_series: pd.DataFrame,
    time_points: List[str],
) -> Dict:
    """
    Calculate Jaccard turnover between consecutive time points.
    
    Turnover = 1 - Jaccard(t1, t2) = proportion of OTUs that changed.
    
    Args:
        otu_table_time_series: OTU table with shape (time_points × OTUs)
        time_points: List of datetime strings or indices
    
    Returns:
        Dict with turnover rates and temporal patterns
    """
    
    if len(otu_table_time_series) < 2:
        return {"error": "Need at least 2 time points"}
    
    turnover_rates = []
    acquisitions = []  # New OTUs at each time point
    losses = []  # Lost OTUs at each time point
    
    for i in range(1, len(otu_table_time_series)):
        prev_otus = set(otu_table_time_series.iloc[i - 1][otu_table_time_series.iloc[i - 1] > 0].index)
        curr_otus = set(otu_table_time_series.iloc[i][otu_table_time_series.iloc[i] > 0].index)
        
        # Jaccard similarity
        intersection = len(prev_otus & curr_otus)
        union = len(prev_otus | curr_otus)
        
        if union == 0:
            jaccard_sim = 1.0
        else:
            jaccard_sim = intersection / union
        
        turnover = 1 - jaccard_sim
        turnover_rates.append(turnover)
        
        # Acquisitions and losses
        acquisitions.append(len(curr_otus - prev_otus))
        losses.append(len(prev_otus - curr_otus))
    
    return {
        "mean_turnover": np.mean(turnover_rates),
        "median_turnover": np.median(turnover_rates),
        "std_turnover": np.std(turnover_rates),
        "turnover_rates": turnover_rates,
        "mean_acquisitions": np.mean(acquisitions),
        "mean_losses": np.mean(losses),
        "acquisitions": acquisitions,
        "losses": losses,
        "n_time_points": len(otu_table_time_series)
    }


def calculate_stability_index(
    abundance_series: np.ndarray,
    log_transform: bool = True
) -> float:
    """
    Calculate stability index: inverse of coefficient of variation (CV).
    
    Stable communities have low CV (high stability).
    
    Args:
        abundance_series: Time series of abundance or relative abundance
        log_transform: Apply log transform before CV calculation
    
    Returns:
        Stability index (0-1, higher = more stable)
    """
    
    abundance_clean = abundance_series[abundance_series > 0]
    
    if len(abundance_clean) < 2:
        return 0.0
    
    if log_transform:
        abundance_clean = np.log10(abundance_clean + 1)
    
    cv = np.std(abundance_clean) / np.mean(abundance_clean) if np.mean(abundance_clean) > 0 else np.inf
    
    # Normalize to 0-1 (assuming CV > 2 = unstable)
    stability = max(0, 1 - (cv / 2))
    
    return float(stability)


def detect_successional_patterns(
    otu_table_ts: pd.DataFrame,
    window_size: int = 3
) -> Dict:
    """
    Detect successional patterns: ordered assembly sequences.
    
    A "pattern" = a set of OTUs that consistently appear in the same order.
    
    Args:
        otu_table_ts: OTU abundance over time
        window_size: Size of temporal window for pattern detection
    
    Returns:
        Dict with detected patterns and sequences
    """
    
    if len(otu_table_ts) < window_size:
        return {"error": "Time series too short for pattern detection"}
    
    patterns = {}
    
    # Identify dominant OTUs at each time point
    dominant_threshold = np.percentile(otu_table_ts.values, 75)
    
    for i in range(len(otu_table_ts) - window_size + 1):
        window = otu_table_ts.iloc[i:i + window_size]
        
        # OTUs above threshold in this window
        dominant_otus = []
        for t in range(window_size):
            otus_t = set(window.iloc[t][window.iloc[t] > dominant_threshold].index)
            dominant_otus.append(otus_t)
        
        # Find OTUs that appear in order
        ordered_sequence = []
        for otu in otu_table_ts.columns:
            appearances = [t for t in range(window_size) if otu in dominant_otus[t]]
            if len(appearances) > 0:
                ordered_sequence.append((otu, appearances))
        
        if len(ordered_sequence) > 2:
            key = tuple(sorted([o[0] for o in ordered_sequence]))
            if key not in patterns:
                patterns[key] = {"count": 0, "time_points": []}
            patterns[key]["count"] += 1
            patterns[key]["time_points"].append(i)
    
    # Rank patterns by frequency
    top_patterns = sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
    
    return {
        "n_patterns_detected": len(patterns),
        "top_patterns": [{"otus": p[0], "frequency": p[1]["count"]} for p in top_patterns],
        "all_patterns": patterns
    }


def estimate_resilience_after_disturbance(
    abundance_pre: np.ndarray,
    abundance_post: np.ndarray,
    recovery_window: int = 10
) -> Dict:
    """
    Estimate resilience by measuring recovery trajectory after disturbance.
    
    Assumptions:
    1. Pre-disturbance = baseline (stable)
    2. Post-disturbance = initial impact
    3. Recovery = return to baseline
    
    Args:
        abundance_pre: Abundance before disturbance
        abundance_post: Abundance immediately after disturbance
        recovery_window: Number of time points to estimate recovery
    
    Returns:
        Dict with resilience metrics
    """
    
    # Calculate initial impact
    if np.mean(abundance_pre) == 0:
        return {"error": "Zero baseline abundance"}
    
    initial_impact = 1 - (np.sum(abundance_post) / np.sum(abundance_pre))
    
    # Recovery rate: slope of return to baseline
    # Simplified: assume exponential recovery with rate constant k
    # Recovery(t) = baseline * (1 - initial_impact * exp(-k*t))
    
    # Estimate k from initial recovery
    if initial_impact > 0:
        recovery_rate = -np.log(initial_impact) / recovery_window
    else:
        recovery_rate = 0.0
    
    recovery_time = -np.log(0.05) / recovery_rate if recovery_rate > 0 else np.inf
    
    return {
        "initial_impact": initial_impact,
        "recovery_rate": recovery_rate,
        "estimated_recovery_time": recovery_time,
        "interpretation": "Highly resilient" if recovery_rate > 0.1 else "Low resilience"
    }


def calculate_temporal_autocorrelation(
    abundance_series: np.ndarray,
    max_lag: int = 10
) -> Dict:
    """
    Calculate autocorrelation function (ACF) for time series.
    
    High ACF = temporal stability; Low ACF = stochastic dynamics.
    
    Args:
        abundance_series: Time series of abundance
        max_lag: Maximum lag for ACF calculation
    
    Returns:
        Dict with ACF values and interpretation
    """
    
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError:
        logger.warning("⚠️ statsmodels not available. Computing simple ACF.")
        
        # Simple ACF calculation
        series = abundance_series - np.mean(abundance_series)
        c0 = np.dot(series, series) / len(series)
        
        acf_vals = [c0]
        for lag in range(1, min(max_lag, len(series) // 2)):
            c_lag = np.dot(series[lag:], series[:-lag]) / len(series)
            acf_vals.append(c_lag / c0)
        
        return {
            "acf_values": acf_vals,
            "significant_lags": [i for i, a in enumerate(acf_vals) if abs(a) > 0.2],
            "mean_acf_positive": np.mean([a for a in acf_vals[1:] if a > 0]),
            "interpretation": "Temporally structured" if abs(acf_vals[1]) > 0.3 else "Random dynamics"
        }
    else:
        # Use statsmodels if available
        acf_vals = acf(abundance_series, nlags=min(max_lag, len(abundance_series) - 1), fft=False)
        
        return {
            "acf_values": acf_vals.tolist(),
            "significant_lags": [i for i, a in enumerate(acf_vals) if abs(a) > 0.2],
            "mean_acf_positive": np.mean([a for a in acf_vals[1:] if a > 0]),
            "interpretation": "Temporally structured" if abs(acf_vals[1]) > 0.3 else "Random dynamics"
        }


def run_temporal_analysis(
    adata,
    time_col: str = "collection_date",
    output_dir: Path = None,
    config: Dict = None,
    group_col: Optional[str] = None
) -> Dict:
    """
    Main entry point for temporal dynamics analysis.
    
    Args:
        adata: AnnData object with samples as rows
        time_col: Column in adata.obs with timestamps
        output_dir: Output directory for results
        config: Configuration dict
        group_col: Optional grouping column (e.g., "Site", "Patient")
    
    Returns:
        Dict with temporal analysis results
    """
    
    if output_dir is None:
        output_dir = Path("./temporal_analysis")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL DYNAMICS ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input: {len(adata)} samples")
    
    # Check for time column
    if time_col not in adata.obs.columns:
        logger.error(f"❌ Time column '{time_col}' not found")
        return {"error": f"Missing {time_col} column"}
    
    # Parse timestamps
    try:
        times = pd.to_datetime(adata.obs[time_col])
    except Exception as e:
        logger.error(f"❌ Failed to parse timestamps: {e}")
        return {"error": str(e)}
    
    results = {}
    
    # 1. Global temporal turnover
    logger.info("\n✓ Computing temporal turnover...")
    
    otu_table = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    otu_table.columns = adata.var_names
    otu_table.index = adata.obs_names
    otu_table['time'] = times.values
    
    # Sort by time
    otu_table = otu_table.sort_values('time')
    otu_table_ts = otu_table.drop('time', axis=1)
    
    turnover_result = calculate_temporal_turnover(otu_table_ts, time_points=times.unique().tolist())
    logger.info(f"  Mean OTU turnover: {turnover_result.get('mean_turnover', np.nan):.3f}")
    results["global_turnover"] = turnover_result
    
    # 2. Successional patterns
    logger.info("\n✓ Detecting successional patterns...")
    succession_result = detect_successional_patterns(otu_table_ts, window_size=3)
    logger.info(f"  Patterns detected: {succession_result.get('n_patterns_detected', 0)}")
    results["succession"] = succession_result
    
    # 3. Stability by group (if specified)
    if group_col and group_col in adata.obs.columns:
        logger.info(f"\n✓ Computing stability by {group_col}...")
        
        stability_by_group = {}
        for group_val in adata.obs[group_col].unique():
            mask = adata.obs[group_col] == group_val
            group_abund = otu_table_ts[mask].sum(axis=1).values
            
            stability = calculate_stability_index(group_abund)
            stability_by_group[str(group_val)] = stability
        
        logger.info(f"  Stability values: {stability_by_group}")
        results["stability_by_group"] = stability_by_group
    
    # Save summary
    summary_lines = [
        f"Temporal Dynamics Analysis",
        f"========================",
        f"Samples: {len(adata)}",
        f"Time span: {times.min()} to {times.max()}",
        f"Mean OTU turnover: {turnover_result.get('mean_turnover', np.nan):.3f}",
        f"Mean OTU acquisitions per time point: {turnover_result.get('mean_acquisitions', np.nan):.1f}",
        f"Patterns detected: {succession_result.get('n_patterns_detected', 0)}"
    ]
    
    summary_text = "\n".join(summary_lines)
    (output_dir / "temporal_summary.txt").write_text(summary_text)
    
    logger.info(f"\n✓ Results saved to {output_dir}/")
    
    return results
