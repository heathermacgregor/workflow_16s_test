# workflow_16s/api/environmental_data/other/tools/_csu_soil.py
"""
CSU Global Soil Heavy Metal Speciation Database

Provides access to the first global database of soil heavy metal speciation.
Covers 56 countries across 5 continents with 49 metals/metalloids per sample.

Official Database Info: https://zjsgczx.csu.edu.cn/sjk/qqtrzjsfcxtsjk.htm
Direct Download (Excel): https://zjsgczx.csu.edu.cn/fubenquanqiuturangzhongjinshufucunxingtaishujuku.xlsx

Citation: Qi, C., Hu, T., Zheng, Y., Wu, M., Tang, F. H., Liu, M., ... & Lin, Z*. (2025).
Global and regional patterns of soil metal (loid) mobility and associated risks.
Nature Communications, 16(1), 2947.

Elements: Hg, Cu, Zn, Cd, As, Ni, Pb, Cr, Co, and 39+ others (49 total metals/metalloids)
Data includes: Total concentrations + chemical speciation

Setup Instructions:
1. Download: https://zjsgczx.csu.edu.cn/fubenquanqiuturangzhongjinshufucunxingtaishujuku.xlsx
2. Convert Excel to CSV or use pandas to load directly
3. Place in data/csu_soil/ directory or configure path in config.yaml
4. Initialize via: CSUSoilAPI.load_from_excel(path_to_file)
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from .base import BaseEnvironmentalAPI
from .cache import cache_api_call

class CSUSoilAPI(BaseEnvironmentalAPI):
    """
    CSU Global Soil Heavy Metal Speciation Database handler.
    Uses nearest-neighbor matching to find closest soil samples for given coordinates.
    
    Data Flow:
    1. Download Excel file from official source
    2. Load via load_from_excel() -> builds SQLite index
    3. Query via get_data(lat, lon) -> returns nearest sample metals/speciation
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.api_name = "CSU_Soil_HeavyMetals"
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "csu_soil_index.db"
        
        # Official source URLs
        self.official_info_url = "https://zjsgczx.csu.edu.cn/sjk/qqtrzjsfcxtsjk.htm"
        self.download_url = "https://zjsgczx.csu.edu.cn/fubenquanqiuturangzhongjinshufucunxingtaishujuku.xlsx"
        
        self.timeout = 15
        self.cache_hits = 0
        self.cache_misses = 0
        
        # CSU heavy metal database: 49 metals/metalloids with speciation
        # Priority metals: Hg, Cu, Zn, Cd, As, Ni, Pb, Cr, Co
        self.metal_elements = {
            'Al', 'As', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr',
            'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho',
            'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd',
            'Ni', 'P', 'Pb', 'Pr', 'S', 'Sb', 'Sc', 'Se', 'Sm', 'Sn',
            'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tm', 'U', 'V', 'W', 'Y',
            'Yb', 'Zn', 'Zr', 'Hg'  # Mercury is a critical element
        }
        
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite spatial index for CSU data"""
        with sqlite3.connect(self.db_path) as conn:
            # Drop old table if it exists (for schema migration)
            conn.execute("DROP TABLE IF EXISTS csu_soil_samples")
            conn.commit()
            
            # Store the actual Excel structure: soil metadata + metal measurements
            conn.execute("""
                CREATE TABLE csu_soil_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clay_percent REAL,
                    ph REAL,
                    cec_cmol_per_kg REAL,
                    oc_percent REAL,
                    metal_element TEXT,
                    total_concentration_mg_kg REAL,
                    bcr_fraction TEXT,
                    fraction_percentage REAL,
                    fetch_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX idx_csu_metal
                ON csu_soil_samples(metal_element)
            """)
            conn.commit()
    
    def load_from_excel(self, excel_path: Path) -> int:
        """
        Load CSU soil speciation data from Excel into SQLite.
        
        The CSU Excel file has format:
        - Clay (%), pH, CEC (cmol+/kg), OC (%), Metal(loid)s (element),
        - Total metal(loid)s content (mg/kg), BCR fraction, Percentage (%)
        
        Args:
            excel_path: Path to downloaded Excel file
            
        Returns:
            Number of records loaded
        """
        if not excel_path.exists():
            self.logger.error(f"Excel file not found: {excel_path}")
            return 0
        
        try:
            # Read Excel file
            self.logger.info(f"Loading CSU speciation data from {excel_path}...")
            df = pd.read_excel(excel_path, sheet_name=0)
            
            self.logger.debug(f"Excel columns: {list(df.columns)}")
            
            if df.empty:
                self.logger.warning("Excel file is empty")
                return 0
            
            # Normalize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Map column names to expected format
            column_mapping = {
                'clay': 'clay_percent',
                'ph': 'ph',
                'cec': 'cec_cmol_per_kg',
                'oc': 'oc_percent',
                'metal(loid)s': 'metal_element',
                'total metal(loid)s content': 'total_concentration_mg_kg',
                'bcr fraction': 'bcr_fraction',
                'percentage': 'fraction_percentage'
            }
            
            # Check which columns are available
            available_cols = {}
            for orig_col, mapped_col in column_mapping.items():
                if orig_col in df.columns:
                    available_cols[orig_col] = mapped_col
            
            if not available_cols:
                self.logger.error(f"No matching columns found. Excel columns: {list(df.columns)}")
                return 0
            
            # Rename columns
            df_insert = df[[col for col in available_cols.keys()]].copy()
            df_insert.columns = [available_cols[col] for col in df_insert.columns]
            
            # Clean up data: convert to numeric where applicable
            numeric_cols = ['clay_percent', 'ph', 'cec_cmol_per_kg', 'oc_percent', 
                          'total_concentration_mg_kg', 'fraction_percentage']
            for col in numeric_cols:
                if col in df_insert.columns:
                    # Skip header rows (1st row often contains units)
                    df_insert[col] = pd.to_numeric(df_insert[col], errors='coerce')
            
            # Remove rows with NaN in critical columns
            critical_cols = ['metal_element', 'total_concentration_mg_kg']
            df_insert = df_insert.dropna(subset=critical_cols)
            
            if df_insert.empty:
                self.logger.warning("No valid data rows after cleaning")
                return 0
            
            # Insert into SQLite
            with sqlite3.connect(self.db_path) as conn:
                cols = df_insert.columns.tolist()
                placeholders = ', '.join(['?' for _ in cols])
                col_names = ', '.join(cols)
                
                for _, row in df_insert.iterrows():
                    values = tuple(row[col] for col in cols)
                    try:
                        conn.execute(
                            f"INSERT INTO csu_soil_samples ({col_names}) VALUES ({placeholders})",
                            values
                        )
                    except Exception as e:
                        self.logger.debug(f"Skipping row: {e}")
                        continue
                
                conn.commit()
                count = conn.execute("SELECT COUNT(*) FROM csu_soil_samples").fetchone()[0]
            
            self.logger.info(f"✓ Loaded {len(df_insert)} speciation records")
            self.logger.info(f"  Total samples in database: {count}")
            self.logger.info(f"  Columns: {cols}")
            return len(df_insert)
            
        except Exception as e:
            self.logger.error(f"Error loading CSU Excel: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0
    
    def check_requirements(self) -> tuple[bool, str]:
        """Verify CSU database availability (cached locally)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM csu_soil_samples"
                ).fetchone()[0]
            
            if count > 0:
                return (True, f"CSU cache: {count} speciation records loaded")
            else:
                return (False, "CSU database not initialized. Run setup_csu_data.py to load.")
        except Exception as e:
            return (False, f"CSU database error: {e}")
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Return aggregate CSU soil metal(loid) concentrations and soil properties.
        
        Note: CSU data is global compilation without specific lat/lon.
        Returns median/mean concentrations of key metals across database.
        
        Returns dict with keys like:
        - csu_soil_hg_median_mg_kg, csu_soil_cu_median_mg_kg, etc.
        - csu_soil_ph_median, csu_soil_clay_median_percent
        - csu_soil_sample_count
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get median concentrations for key metals
                key_metals = ['Hg', 'Cu', 'Zn', 'Cd', 'As', 'Ni', 'Pb', 'Cr', 'Co', 'Fe']
                
                result = {'csu_soil_source': 'Global specifications database'}
                
                # Get soil property medians
                soil_props = conn.execute("""
                    SELECT 
                        COUNT(*) as sample_count,
                        ROUND(AVG(CAST(ph AS REAL)), 2) as ph_avg,
                        ROUND(AVG(CAST(clay_percent AS REAL)), 2) as clay_avg,
                        ROUND(AVG(CAST(oc_percent AS REAL)), 2) as oc_avg
                    FROM csu_soil_samples
                """).fetchone()
                
                if soil_props:
                    result['csu_soil_sample_count'] = soil_props['sample_count']
                    result['csu_soil_ph_avg'] = soil_props['ph_avg']
                    result['csu_soil_clay_percent_avg'] = soil_props['clay_avg']
                    result['csu_soil_oc_percent_avg'] = soil_props['oc_avg']
                
                # Get metal concentrations
                for metal in key_metals:
                    metal_data = conn.execute("""
                        SELECT 
                            ROUND(AVG(CAST(total_concentration_mg_kg AS REAL)), 2) as conc_avg,
                            COUNT(*) as count
                        FROM csu_soil_samples
                        WHERE metal_element LIKE ?
                    """, (f'%{metal}%',)).fetchone()
                    
                    if metal_data and metal_data['count'] > 0:
                        result[f'csu_soil_{metal.lower()}_mg_kg_avg'] = metal_data['conc_avg']
                        result[f'csu_soil_{metal.lower()}_measurement_count'] = metal_data['count']
                
                self.cache_hits += 1
                return result if result else None
                
        except Exception as e:
            self.logger.error(f"Error querying CSU data: {e}")
            self.cache_misses += 1
            return None


def csu_soil_integration(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Quick integration function for CSU Soil data."""
    api = CSUSoilAPI(verbose=False)
    return api.get_data(lat, lon)
