# workflow_16s/api/environmental_data/other/tools/_usgs_cmibs_impl.py
"""
USGS CMIBS Real Implementation (Local Database)

This replaces the stub HTTP endpoint with actual geochemical data
from the USGS Crustal Metals in Integrated Bedrock and Sediment (CMIBS) database.

The database is built locally from USGS CSV data and provides fast spatial queries
via SQLite nearest-neighbor lookups.

Reference: https://www.usgs.gov/faqs/what-crustal-metals-integrated-bedrock-and-sediment-database
Data: Download from USGS ScienceBase (search for "CMIBS")
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import sqlite3

from .base import BaseEnvironmentalAPI
from .cache import cache_api_call
from ._cmibs_database import CMIBSDatabase

logger = logging.getLogger("workflow_16s")


class USGS_CMIBS_API(BaseEnvironmentalAPI):
    """
    USGS CMIBS Database API wrapper (REAL IMPLEMENTATION).
    
    Uses local SQLite database of CMIBS geochemical samples
    for fast nearest-neighbor queries (no HTTP calls).
    """
    
    def __init__(self, verbose: bool = False, db_path: Optional[str] = None):
        """
        Initialize CMIBS API with local database.
        
        Args:
            verbose: Enable verbose logging
            db_path: Path to CMIBS SQLite database. If None, uses default from config.
        """
        super().__init__(verbose=verbose)
        self.api_name = "USGS_CMIBS"
        self.db_path = db_path or self._get_default_db_path()
        self.db = None
        self.search_radius_km = 100
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize database connection
        self._init_database()
    
    def _get_default_db_path(self) -> str:
        """Get default database path from project or config."""
        # Try standard locations in order of preference
        possible_paths = [
            Path.home() / ".workflow_16s" / "cmibs_database.db",
            Path("/data/cmibs/cmibs_database.db"),
            Path("/tmp/cmibs_database.db"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # If none exist, use home directory
        return str(possible_paths[0])
    
    def _init_database(self):
        """Initialize database connection."""
        try:
            self.db = CMIBSDatabase(self.db_path)
            if self.db.is_built:
                logger.info(f"✅ USGS CMIBS: Database ready at {self.db_path}")
            else:
                logger.warning(f"⚠️  USGS CMIBS: Database not built. Run build_from_csv() with CMIBS CSV file.")
        except Exception as e:
            logger.error(f"❌ USGS CMIBS: Failed to initialize: {e}")
            self.db = None
    
    def check_requirements(self) -> tuple:
        """
        Verify CMIBS database is available and built.
        
        Returns:
            (is_available, message)
        """
        if not self.db:
            return False, "CMIBS database connection failed"
        
        if not self.db.is_built:
            return False, "CMIBS database not built (call build_from_csv with USGS CMIBS CSV)"
        
        return True, "USGS CMIBS database ready (local SQLite)"
    
    @cache_api_call
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Query nearest CMIBS sample for crustal metal data.
        
        Args:
            lat: Sample latitude
            lon: Sample longitude
            **kwargs: Additional options (search_radius_km, etc.)
            
        Returns:
            Dict with:
            - usgs_cmibs_{metal}_ppm: Concentration in parts per million
            - usgs_cmibs_sample_distance_km: Distance to nearest sample
            - usgs_cmibs_sample_id: Sample identifier
            
            Or None if no sample found within search radius.
        """
        if not self.db or not self.db.is_built:
            return None
        
        try:
            # Get search radius from kwargs or use default
            search_radius = kwargs.get('search_radius_km', self.search_radius_km)
            
            # Query database
            nearest = self.db.get_nearest(lat, lon, radius_km=search_radius)
            
            if not nearest:
                return None
            
            # Format output with CMIBS namesp ace prefix
            result = {
                'usgs_cmibs_sample_id': nearest['sample_id'],
                'usgs_cmibs_latitude': nearest['latitude'],
                'usgs_cmibs_longitude': nearest['longitude'],
                'usgs_cmibs_sample_distance_km': nearest['distance_km'],
            }
            
            # Add all metal concentrations with prefix
            for metal in ['ni', 'mo', 'v', 'co', 'zn', 'cu', 'pb', 'as', 'cd', 'cr',
                         'mn', 'fe', 'al', 'ca', 'k', 'na', 'mg', 'si', 'ti', 'p',
                         's', 'au', 'ag', 'be', 'bi', 'ga', 'ge', 'hf', 'in', 'la',
                         'li', 'nb', 'nd', 're', 'sb', 'sc', 'se', 'sn', 'sr', 'ta',
                         'te', 'th', 'u', 'w', 'y', 'yb', 'zr', 'b', 'ce', 'dy']:
                col_name = f'{metal}_ppm'
                if col_name in nearest and nearest[col_name] is not None:
                    result[f'usgs_cmibs_{col_name}'] = nearest[col_name]
            
            return result
        
        except Exception as e:
            logger.warning(f"⚠️  USGS CMIBS get_data failed: {e}")
            return None
    
    def build_from_csv(self, csv_path: str, progress_callback=None) -> bool:
        """
        Build the CMIBS database from a USGS CSV file.
        
        Download USGS CMIBS data from ScienceBase and pass the CSV path here.
        
        Args:
            csv_path: Path to USGS CMIBS CSV file
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            True if database built successfully
        """
        try:
            if not self.db:
                self.db = CMIBSDatabase(self.db_path)
            
            count = self.db.build_from_csv(csv_path, progress_callback=progress_callback)
            logger.info(f"✅ USGS CMIBS database built with {count} samples")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to build CMIBS database: {e}")
            return False
    
    def __del__(self):
        """Clean up database connection."""
        if self.db:
            self.db.close()


# For backward compatibility with stub API
__all__ = ['USGS_CMIBS_API']
