import json
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger
from .manager import MetadataManager


async def process_metadata(
    df: pd.DataFrame, 
    output_path: Union[str, Path], 
    config: Optional[AppConfig] = None
) -> pd.DataFrame:
    """High-level async executor function to run the full metadata processing pipeline."""
    logger = get_logger("workflow_16s")
    if config is None: config = AppConfig() # type: ignore

    try:
        manager = MetadataManager(metadata=df, config=config)
        cleaned_df = await manager.run_pipeline()

        if not cleaned_df.empty:
            MetadataManager.export_tsv(cleaned_df, output_path)
        else:
            logger.warning("Pipeline resulted in an empty DataFrame. No file was saved.")
            return df

        report = manager.get_cleaning_report()
        
        output_path = Path(output_path)
        report_path = output_path.parent / f"{output_path.stem}_cleaning_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Metadata cleaning complete. A detailed report was saved to: {report_path}")
        
        return cleaned_df
    
    except Exception as e:
        logger.error(
            f"An error occurred during the metadata processing workflow: {e}", exc_info=True
        )
        return df 
        
def import_tsv(metadata_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(metadata_path, sep='\t', low_memory=False)

def export_tsv(metadata: pd.DataFrame, output_path: Union[str, Path]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, sep='\t', index=True)
    
def standardize_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Finds and renames latitude/longitude columns to 'lat' and 'lon'."""
    df = df.copy()
    
    if 'lat' not in df.columns:
        lat_found = False
        lat_patterns = [
            (r'^latitude$', 'exact'),
            (r'.*_lat$', 'suffix'),
            (r'^latitude_.*', 'prefix')
        ]
        for col in df.columns:
            for pattern, _ in lat_patterns:
                if re.match(pattern, col, re.IGNORECASE):
                    logger.info(f"Found latitude-like column '{col}'. Renaming to 'lat'.")
                    df.rename(columns={col: 'lat'}, inplace=True)
                    lat_found = True
                    break
            if lat_found:
                break

    if 'lon' not in df.columns:
        lon_found = False
        lon_patterns = [
            (r'^longitude$', 'exact'),
            (r'^lon$', 'exact'), 
            (r'.*_lon$', 'suffix'),
            (r'^longitude_.*', 'prefix')
        ]
        for col in df.columns:
            for pattern, _ in lon_patterns:
                if re.match(pattern, col.strip(), re.IGNORECASE):
                    logger.info(f"Found longitude-like column '{col}'. Renaming to 'lon'.")
                    df.rename(columns={col: 'lon'}, inplace=True)
                    lon_found = True
                    break
            if lon_found:
                break
                
    return df