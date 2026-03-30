# workflow_16s/api/environmental_data/nuclear_fuel_cycle/tools/_gem.py

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class GNPT:
    """A class that handles the GEM Global Nuclear Power Tracker database."""
    
    # Updated mapping to include Coordinates (saves geocoding time)
    DEFAULT_GEM_COLUMNS = {
        'facility_country': "Country/Area",
        'facility': "Project Name",
        'facility_type': "Reactor Type",
        'facility_capacity': "Capacity (MW)",
        'facility_status': "Status",
        'facility_start_year': "Start Year",
        'facility_end_year': "Retirement Year",
        'lat': "Latitude",
        'lon': "Longitude"
    }
    
    # 2024-07 Release of Global Nuclear Power Tracker
    URL = "https://zenodo.org/records/17109112/files/gem_nuclearpower_2024-07.tsv"

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: The directory where the GNPT file will be stored. 
                        A default directory './nfc_facilities/' will be used if None.
        """
        if output_dir: self.output_dir = Path(output_dir)
        else: self.output_dir = Path.cwd() / "nfc_facilities"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_dir / "gem_nuclearpower_2024-07.tsv"
        self.logger = get_logger("workflow_16s")
        self.logger.info(f"GEM database configured to use file: {self.file_path}")
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """Loads the GEM database from the local TSV file into a DataFrame."""
        
        # 1. Download if missing
        if not self.file_path.exists():
            try:
                self.logger.info(f"GEM database file not found at {self.file_path}. Downloading...")
                self._download()
            except Exception as e:
                self.logger.error(f"Failed to download GEM data: {e}")
                return pd.DataFrame()
        
        # 2. Load Data
        try:
            # Load with flexible whitespace handling
            self.df = pd.read_csv(self.file_path, sep='\t', encoding_errors='replace')
            
            # Clean column names (strip whitespace like " Capacity " -> "Capacity")
            self.df.columns = self.df.columns.str.strip()
            
            self.logger.info(f"Successfully loaded GEM data with {self.df.shape[0]} facilities.")
        except Exception as e:
            self.logger.error(f"Error loading GEM data from {self.file_path}: {e}")
            return pd.DataFrame()

        # 3. Filter and Rename Columns
        # Only keep columns that actually exist in the file
        existing_cols = {k: v for k, v in self.DEFAULT_GEM_COLUMNS.items() if v in self.df.columns}
        
        # Select and Rename
        self.df = self.df[list(existing_cols.values())]
        self.df = self.df.rename(columns={v: k for k, v in existing_cols.items()})
        
        self.df['data_source'] = "GEM"
        
        # Ensure capacity is numeric
        if 'facility_capacity' in self.df.columns:
            self.df['facility_capacity'] = pd.to_numeric(self.df['facility_capacity'], errors='coerce')

        return self.df
    
    def _download(self):
        """Downloads the GEM database file using Python requests (no wget dependency)."""
        headers = {'User-Agent': 'Mozilla/5.0 (Workflow 16S)'}
        
        with requests.get(self.URL, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(self.file_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        self.logger.info(f"Downloaded GEM database to {self.file_path}")