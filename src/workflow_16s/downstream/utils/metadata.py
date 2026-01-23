"""Utility functions for metadata processing and standardization."""
from typing import Optional, Tuple
import pandas as pd
from workflow_16s.utils.logger import get_logger
logger = get_logger("workflow_16s")

TARGET_GENE_NORMALIZATION = {
    '16S': ['16S', '16S rRNA', '16S rRNA gene', '16s', '16s rrna'],
    '18S': ['18S', '18S rRNA', '18s'],
    'ITS': ['ITS', 'ITS1', 'ITS2', 'its'],
}

def normalize_target_gene(value: str) -> Optional[str]:
    if pd.isna(value) or value in ['', 'nan', 'None', 'NA']:
        return None
    value_str = str(value).strip()
    for canonical, synonyms in TARGET_GENE_NORMALIZATION.items():
        if value_str in synonyms:
            return canonical
    return value_str

def standardize_dates(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    date_patterns = ['date', 'time', 'temporal', 'year', 'month', 'day']
    date_columns = [col for col in obs_df.columns if any(pattern in col.lower() for pattern in date_patterns)]
    standardized_count = 0
    for col in date_columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(obs_df[col]):
                dt_series = pd.to_datetime(obs_df[col], errors='coerce')
                obs_df[col] = dt_series.dt.strftime('%Y-%m-%d').replace('NaT', pd.NA)
                standardized_count += 1
                continue
            if pd.api.types.is_numeric_dtype(obs_df[col]): continue
            if obs_df[col].dtype == 'object' or pd.api.types.is_string_dtype(obs_df[col]):
                dt_series = pd.to_datetime(obs_df[col], errors='coerce')
                original_valid = obs_df[col].notna().sum()
                parsed_valid = dt_series.notna().sum()
                if parsed_valid >= original_valid * 0.1 and parsed_valid > 0:
                    obs_df[col] = dt_series.dt.strftime('%Y-%m-%d').replace('NaT', pd.NA)
                    standardized_count += 1
        except Exception: continue
    return obs_df, standardized_count