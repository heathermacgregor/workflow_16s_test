"""
USGS CMIBS Database Builder and Query Interface

Builds a local SQLite database from USGS Crustal Metals in Integrated Bedrock
and Sediment (CMIBS) geochemical survey data.

Data source: https://www.usgs.gov/faqs/what-crustal-metals-integrated-bedrock-and-sediment-database
ScienceBase: Search for "CMIBS" or use direct link from USGS publications

Usage:
    # Build database from CSV
    db = CMIBSDatabase(db_path='/path/to/cmibs.db')
    db.build_from_csv('/path/to/CMIBS_data.csv')
    
    # Query nearest sample
    result = db.get_nearest(lat=40.7128, lon=-74.0060, radius_km=100)
    # Returns: {
    #     'sample_id': 'ABC123',
    #     'latitude': 40.71,
    #     'longitude': -74.01,
    #     'distance_km': 15.3,
    #     'ni_ppm': 45.2,
    #     'cu_ppm': 89.3,
    #     ...
    # }
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from math import radians, cos, sin, asin, sqrt

logger = logging.getLogger("workflow_16s")

# USGS CMIBS metals tracked in the database
CMIBS_METALS = [
    'Ni', 'Mo', 'V', 'Co', 'Zn', 'Cu', 'Pb', 'As', 'Cd', 'Cr',
    'Mn', 'Fe', 'Al', 'Ca', 'K', 'Na', 'Mg', 'Si', 'Ti', 'P',
    'S', 'Au', 'Ag', 'Be', 'Bi', 'Ga', 'Ge', 'Hf', 'In', 'La',
    'Li', 'Nb', 'Nd', 'Re', 'Sb', 'Sc', 'Se', 'Sn', 'Sr', 'Ta',
    'Te', 'Th', 'U', 'W', 'Y', 'Yb', 'Zr', 'B', 'Ce', 'Dy'
]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points (in km).
    
    Args:
        lat1, lon1: Point 1 coordinates
        lat2, lon2: Point 2 coordinates
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


class CMIBSDatabase:
    """
    Local SQLite database for USGS CMIBS geochemical data.
    
    Provides fast spatial queries to find nearest geochemical samples.
    """
    
    def __init__(self, db_path: str, auto_create: bool = True):
        """
        Initialize CMIBS database connection.
        
        Args:
            db_path: Path to SQLite database file
            auto_create: Create database if it doesn't exist
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.is_built = False
        
        if auto_create and not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_connection()
    
    def _init_connection(self):
        """Initialize database connection."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            
            # Check if database already has data
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cmibs_samples'")
            self.is_built = cursor.fetchone()[0] > 0
            
            if self.is_built:
                cursor.execute("SELECT COUNT(*) FROM cmibs_samples")
                sample_count = cursor.fetchone()[0]
                logger.info(f"✅ CMIBS database loaded: {sample_count} samples")
        except sqlite3.OperationalError:
            # Database doesn't exist or is empty
            self.is_built = False
    
    def create_schema(self):
        """Create database schema for CMIBS data."""
        cursor = self.conn.cursor()
        
        # Drop existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS cmibs_samples")
        
        # Create samples table
        sql = """
        CREATE TABLE cmibs_samples (
            sample_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            sample_type TEXT,
            country TEXT,
            state TEXT,
            source TEXT
        )
        """
        cursor.execute(sql)
        
        # Add metal concentration columns
        for metal in CMIBS_METALS:
            cursor.execute(f"ALTER TABLE cmibs_samples ADD COLUMN {metal.lower()}_ppm REAL")
        
        # Create spatial index (for faster queries)
        cursor.execute("""
        CREATE INDEX idx_cmibs_coords 
        ON cmibs_samples(latitude, longitude)
        """)
        
        self.conn.commit()
        logger.info(f"✅ Created CMIBS schema with {len(CMIBS_METALS)} metal columns")
    
    def build_from_csv(self, csv_path: str, sample_id_col: str = 'Sample_ID', 
                       lat_col: str = 'Latitude', lon_col: str = 'Longitude',
                       progress_callback=None) -> int:
        """
        Build database from USGS CMIBS CSV file.
        
        Args:
            csv_path: Path to CSV file
            sample_id_col: Name of sample ID column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            progress_callback: Optional callback for progress (current, total)
            
        Returns:
            Number of samples inserted
        """
        logger.info(f"📥 Reading CMIBS data from {csv_path}...")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"   Found {len(df)} samples")
        
        # Validate required columns
        required_cols = [sample_id_col, lat_col, lon_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns: {required_cols}")
        
        # Create schema
        self.create_schema()
        
        # Insert data
        cursor = self.conn.cursor()
        inserted = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract base info
                sample_id = str(row[sample_id_col])
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                # Skip invalid coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                
                # Build insert statement
                cols = ['sample_id', 'latitude', 'longitude']
                vals = [sample_id, lat, lon]
                
                # Add optional fields
                for optional_col in ['sample_type', 'country', 'state', 'source']:
                    if optional_col in df.columns:
                        cols.append(optional_col)
                        vals.append(row.get(optional_col, None))
                
                # Add metal concentrations
                for metal in CMIBS_METALS:
                    metal_col = f"{metal.lower()}_ppm"
                    # Check various column name formats in CSV
                    col_to_use = None
                    for variant in [f"{metal}_ppm", f"{metal.lower()}_ppm", f"conc_{metal}", metal]:
                        if variant in df.columns:
                            col_to_use = variant
                            break
                    
                    if col_to_use:
                        try:
                            val = float(row[col_to_use])
                            cols.append(metal_col)
                            vals.append(val if not np.isnan(val) else None)
                        except (ValueError, TypeError):
                            pass
                
                # Insert row
                placeholders = ','.join(['?' for _ in cols])
                sql = f"INSERT INTO cmibs_samples ({','.join(cols)}) VALUES ({placeholders})"
                cursor.execute(sql, vals)
                inserted += 1
                
                # Progress feedback
                if progress_callback and (idx % 1000) == 0:
                    progress_callback(idx, len(df))
                
            except Exception as e:
                logger.warning(f"   ⚠️  Error inserting sample {row.get(sample_id_col, idx)}: {e}")
        
        self.conn.commit()
        self.is_built = True
        logger.info(f"✅ Inserted {inserted}/{len(df)} samples into CMIBS database")
        
        return inserted
    
    def get_nearest(self, lat: float, lon: float, radius_km: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get nearest CMIBS sample within search radius.
        
        Args:
            lat: Sample latitude
            lon: Sample longitude
            radius_km: Maximum search radius in kilometers
            
        Returns:
            Dict with sample data and distance, or None if not found
        """
        if not self.is_built:
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Rough bounding box search first (faster)
            lat_delta = radius_km / 111.0  # ~111 km per degree latitude
            lon_delta = radius_km / (111.0 * cos(radians(lat)))
            
            sql = """
            SELECT * FROM cmibs_samples
            WHERE latitude >= ? AND latitude <= ?
            AND longitude >= ? AND longitude <= ?
            ORDER BY ABS(latitude - ?) + ABS(longitude - ?)
            LIMIT 10
            """
            
            cursor.execute(sql, [
                lat - lat_delta, lat + lat_delta,
                lon - lon_delta, lon + lon_delta,
                lat, lon
            ])
            
            rows = cursor.fetchall()
            
            # Find actual nearest among candidates
            best = None
            best_distance = float('inf')
            
            for row in rows:
                distance = haversine(lat, lon, row['latitude'], row['longitude'])
                if distance <= radius_km and distance < best_distance:
                    best = row
                    best_distance = distance
            
            if best:
                # Convert row to dict and add distance
                result = {
                    'sample_id': best['sample_id'],
                    'latitude': best['latitude'],
                    'longitude': best['longitude'],
                    'distance_km': round(best_distance, 2),
                    'sample_type': best['sample_type'],
                    'country': best['country']
                }
                
                # Add metal values
                for metal in CMIBS_METALS:
                    col_name = f"{metal.lower()}_ppm"
                    val = best[col_name]
                    if val is not None:
                        result[col_name] = val
                
                return result
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Error querying CMIBS: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("USGS CMIBS Local Database Builder")
    print("="*70)
    
    db = CMIBSDatabase(db_path="/tmp/cmibs_test.db")
    
    # Demonstrate schema creation
    print("\n✓ Database initialized")
    print("✓ Ready to build from USGS CMIBS CSV")
    print("\nUsage:")
    print("  db.build_from_csv('CMIBS_data.csv')")
    print("  result = db.get_nearest(lat=40.7, lon=-74.0, radius_km=100)")
    print("="*70 + "\n")
