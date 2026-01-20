import logging
from pathlib import Path
from typing import Dict, List, Optional
import anndata as ad
import pandas as pd
from workflow_16s.logger import get_logger

logger = get_logger()

class DataLoader:
    """Loads AnnData objects from .h5ad files for analysis."""
    def __init__(self, config: Dict):
        self.config = config
        self.project_dir = Path(config.get("project_dir", "."))
        self.processed_data_dir = self.project_dir / "03_processed_data"

    def discover_datasets(self) -> Dict[str, Path]:
        """Discovers all .h5ad files in the processed data directory."""
        if not self.processed_data_dir.exists():
            # In a real run this is an error, but for the example run we can skip it.
            logger.warning(f"Processed data directory not found: {self.processed_data_dir}. Cannot discover real datasets.")
            return {}
        
        h5ad_paths = list(self.processed_data_dir.glob("*.h5ad"))
        if not h5ad_paths:
            logger.warning(f"No .h5ad files found in {self.processed_data_dir}")
            return {}
        
        datasets = {p.stem: p for p in h5ad_paths}
        logger.info(f"Discovered {len(datasets)} datasets for analysis.")
        return datasets

    def run_combined(self) -> Optional[ad.AnnData]:
        """Loads and merges all discovered datasets into a single AnnData object."""
        datasets = self.discover_datasets()
        if not datasets: return None

        adata_list = [ad.read_h5ad(p) for p in datasets.values()]
        
        for i, adata in enumerate(adata_list):
            adata.obs['batch'] = list(datasets.keys())[i]
        
        merged_adata = ad.concat(adata_list, join='outer', merge='same')
        
        # Categorize columns and store in .uns
        merged_adata.uns['analysis_columns'] = self._categorize_columns_for_analysis(merged_adata.obs)
        
        logger.info(f"Merged {len(adata_list)} datasets into a single object.")
        return merged_adata

    def run_single(self, path: Path) -> Optional[ad.AnnData]:
        """Loads a single, specified dataset."""
        logger.info(f"Loading single dataset from: {path}")
        try:
            adata = ad.read_h5ad(path)
            # Categorize columns and store in .uns
            adata.uns['analysis_columns'] = self._categorize_columns_for_analysis(adata.obs)
            return adata
        except Exception as e:
            logger.error(f"Failed to load AnnData file {path}: {e}")
            return None

    def _categorize_columns_for_analysis(self, metadata_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically categorizes metadata columns for downstream analyses."""
        analysis_cols = {"group_comparison": [], "correlation_gradient": [], "potential_confounders": []}
        id_like_cols = {'sampleid', '#sampleid', 'sample_id', 'barcode', 'description'}
        lat_lon_cols = {'latitude', 'longitude', 'lat', 'lon'}
        confounder_cols = {'sequencing_run', 'run_id', 'batch_number', 'dna_extraction_kit', 'batch'}

        for col in metadata_df.columns:
            if col in id_like_cols or col in lat_lon_cols: continue
            if col in confounder_cols:
                analysis_cols["potential_confounders"].append(col)
                continue
            
            if pd.api.types.is_bool_dtype(metadata_df[col]):
                analysis_cols["group_comparison"].append(col)
                continue

            if pd.api.types.is_numeric_dtype(metadata_df[col]):
                if metadata_df[col].nunique() < 10:
                    analysis_cols["group_comparison"].append(col)
                else:
                    analysis_cols["correlation_gradient"].append(col)
            elif pd.api.types.is_object_dtype(metadata_df[col]) or pd.api.types.is_categorical_dtype(metadata_df[col]):
                if metadata_df[col].nunique() < min(50, len(metadata_df) * 0.5):
                    analysis_cols["group_comparison"].append(col)
        
        logger.info("Successfully categorized metadata columns for analysis.")
        return analysis_cols