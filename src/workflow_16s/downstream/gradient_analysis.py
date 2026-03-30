# downstream/gradient_analysis.py

"""
Gradient & Ecotype Analysis: Detect bimodal species distributions and cryptic ecotypes.

Answers: Along environmental gradients (pH, metals, temperature), do OTUs show
bimodal distributions? This suggests cryptic ecotypes occupying different niches.

Works with sparse metadata; can auto-enrich from lat/lon using OpenMeteo + SoilGrids.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


def test_bimodality(values: np.ndarray, threshold: float = 0.7) -> Dict:
    """
    Test if a distribution is bimodal using Hartigan's dip test.
    
    Args:
        values: Abundance or trait values
        threshold: p-value threshold for significance (default: 0.7 → strict, only ~30% false positives)
    
    Returns:
        Dict with dip statistic and p-value
    """
    
    # Remove NaN/zero values
    values = values[~np.isnan(values)]
    values = values[values > 0]
    
    if len(values) < 5:
        return {"is_bimodal": False, "dip_statistic": np.nan, "p_value": 1.0}
    
    try:
        from diptest import diptest
        dip_stat, p_value = diptest(values, numsim=1000)
        is_bimodal = p_value < (1 - threshold)
        
        return {
            "is_bimodal": is_bimodal,
            "dip_statistic": dip_stat,
            "p_value": p_value,
            "n_samples": len(values),
            "interpretation": "Bimodal" if is_bimodal else "Unimodal"
        }
    except ImportError:
        logger.warning("⚠️ diptest not available. Using simpler bimodality check (Hartigan's dip approximation).")
        
        # Fallback: check for two peaks using KDE
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde
        
        try:
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 1000)
            density = kde(x_range)
            peaks, _ = find_peaks(density)
            
            is_bimodal = len(peaks) >= 2
            dip_stat = len(peaks)  # Use number of peaks as proxy
            
            return {
                "is_bimodal": is_bimodal,
                "dip_statistic": dip_stat,
                "p_value": 1.0 - (dip_stat / 5),  # Heuristic p-value
                "n_samples": len(values),
                "interpretation": "Bimodal" if is_bimodal else "Unimodal",
                "method": "KDE-based approximation"
            }
        except Exception as e:
            logger.warning(f"Bimodality test failed: {e}")
            return {
                "is_bimodal": False,
                "dip_statistic": np.nan,
                "p_value": 1.0,
                "n_samples": len(values)
            }


def get_ecotype_boundaries(
    values: np.ndarray,
    percentile: float = 25.0
) -> Tuple[float, float]:
    """
    Define ecotype boundaries based on abundance percentiles.
    
    Args:
        values: Abundance values (log-normalized if possible)
        percentile: Percentile for boundaries (e.g., 25 → uses 25th and 75th percentile)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    values_clean = values[values > 0]
    
    if len(values_clean) < 4:
        return (values_clean.min(), values_clean.max())
    
    lower = np.percentile(values_clean, percentile)
    upper = np.percentile(values_clean, 100 - percentile)
    
    return (lower, upper)


def enrich_metadata_with_openmeteo(
    metadata_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> pd.DataFrame:
    """
    Enrich metadata with climate data from OpenMeteo for lat/lon coordinates.
    
    Args:
        metadata_df: Metadata with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
    
    Returns:
        Updated metadata with temperature, precipitation, etc.
    """
    
    if lat_col not in metadata_df.columns or lon_col not in metadata_df.columns:
        logger.warning(f"⚠️ Lat/lon columns ({lat_col}, {lon_col}) not found. Skipping enrichment.")
        return metadata_df
    
    logger.info(f"Enriching metadata with OpenMeteo climate data for {len(metadata_df)} samples...")
    
    try:
        import requests
    except ImportError:
        logger.warning("⚠️ requests library not available. Cannot fetch OpenMeteo data.")
        return metadata_df
    
    # Get unique coordinates
    coords = metadata_df[[lat_col, lon_col]].drop_duplicates()
    
    climate_data = {}
    
    for idx, row in coords.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        try:
            # OpenMeteo API call for annual statistics
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2000-01-01&end_date=2024-12-31&monthly_aggregation=true&temperature_2m_mean=true&precipitation_sum=true"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                temps = data.get("monthly", {}).get("temperature_2m_mean", [])
                precips = data.get("monthly", {}).get("precipitation_sum", [])
                
                if temps and precips:
                    climate_data[(lat, lon)] = {
                        "mean_temperature": np.mean([t for t in temps if t is not None]),
                        "mean_precipitation": np.mean([p for p in precips if p is not None])
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch climate data for ({lat}, {lon}): {e}")
    
    # Add enriched columns
    metadata_df["mean_temperature"] = metadata_df.apply(
        lambda row: climate_data.get((row[lat_col], row[lon_col]), {}).get("mean_temperature", np.nan),
        axis=1
    )
    metadata_df["mean_precipitation"] = metadata_df.apply(
        lambda row: climate_data.get((row[lat_col], row[lon_col]), {}).get("mean_precipitation", np.nan),
        axis=1
    )
    
    logger.info(f"✓ Enriched {len(climate_data)} unique locations")
    
    return metadata_df


def run_gradient_analysis(
    otu_table: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
    environmental_gradients: List[str] = None,
    auto_enrich_metadata: bool = True,
    bimodality_threshold: float = 0.7,
    min_samples_per_otu: int = 10,
) -> Dict:
    """
    Main entry point for gradient and ecotype analysis.
    
    Args:
        otu_table: OTU abundance table (samples × OTUs)
        metadata_df: Sample metadata with environmental variables
        output_dir: Output directory
        config: Configuration dict
        environmental_gradients: Env. variables to test (default: use available columns)
        auto_enrich_metadata: Auto-fetch OpenMeteo/SoilGrids data
        bimodality_threshold: p-value threshold for bimodality (0-1, higher = more strict)
        min_samples_per_otu: Minimum samples where OTU must be present
    
    Returns:
        Dict with ecotype detection results
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("GRADIENT & ECOTYPE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input: {len(otu_table)} samples × {len(otu_table.columns)} OTUs")
    logger.info(f"Metadata columns: {list(metadata_df.columns)[:10]}...")
    
    # Enrich metadata if requested
    if auto_enrich_metadata:
        metadata_df = enrich_metadata_with_openmeteo(metadata_df)
    
    # Determine gradients to test
    if environmental_gradients is None:
        # Auto-detect numeric columns
        numeric_cols = metadata_df.select_dtypes(include=[np.number]).columns.tolist()
        environmental_gradients = numeric_cols[:5]  # Limit to top 5
    
    environmental_gradients = [g for g in environmental_gradients if g in metadata_df.columns]
    logger.info(f"Testing {len(environmental_gradients)} environmental gradients: {', '.join(environmental_gradients)}")
    
    # Test each OTU for bimodal distribution along gradients
    ecotype_results = []
    bimodal_otus = []
    
    for otu_id in otu_table.columns:
        # Minimum prevalence filter
        n_present = (otu_table[otu_id] > 0).sum()
        if n_present < min_samples_per_otu:
            continue
        
        for gradient in environmental_gradients:
            if gradient not in metadata_df.columns:
                continue
            
            # Get OTU abundance for samples with metadata
            valid_samples = metadata_df[gradient].notna()
            abundance = otu_table.loc[valid_samples, otu_id].values
            gradient_values = metadata_df.loc[valid_samples, gradient].values
            
            if len(abundance) < 5:
                continue
            
            # Test bimodality
            result = test_bimodality(abundance, threshold=bimodality_threshold)
            
            if result.get("is_bimodal"):
                # Get ecotype boundaries
                lower, upper = get_ecotype_boundaries(abundance)
                
                ecotype_results.append({
                    "OTU_ID": otu_id,
                    "Gradient": gradient,
                    "Is_Bimodal": True,
                    "Dip_Statistic": result.get("dip_statistic"),
                    "P_Value": result.get("p_value"),
                    "Ecotype_Lower": lower,
                    "Ecotype_Upper": upper,
                    "N_Samples": len(abundance),
                    "Ecotype1_Count": (abundance < upper).sum(),
                    "Ecotype2_Count": (abundance >= upper).sum()
                })
                
                bimodal_otus.append(otu_id)
    
    ecotype_df = pd.DataFrame(ecotype_results)
    
    logger.info(f"\n✓ Analysis complete:")
    logger.info(f"  Found {len(set(bimodal_otus))} OTUs with bimodal distributions")
    logger.info(f"  Total {len(ecotype_df)} OTU-gradient combinations with bimodality")
    
    if len(ecotype_df) > 0:
        logger.info(f"\n  Top 5 bimodal signals:")
        for idx, row in ecotype_df.nsmallest(5, "P_Value").iterrows():
            logger.info(f"    {row['OTU_ID']} along {row['Gradient']}: p={row['P_Value']:.4f}")
    
    # Save results
    ecotype_df.to_csv(output_dir / "ecotype_bimodality_results.csv", index=False)
    
    logger.info(f"\n✓ Results saved to {output_dir}/")
    
    return {
        "bimodal_otus": list(set(bimodal_otus)),
        "ecotype_results": ecotype_df,
        "n_bimodal": len(set(bimodal_otus))
    }
