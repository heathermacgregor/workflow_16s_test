import sqlite3
import math
import requests
import time
import srtm
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from workflow_16s.utils.logger import with_logger

@with_logger
class GeoContextEnricher:
    def __init__(self, adata, **kwargs):
        self.adata = adata
        self.obs = adata.obs
        from workflow_16s.utils.logger import get_logger
        self.logger = kwargs.get('logger') or get_logger("workflow_16s")
        
        # Setup Cache Directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "weather_cache.db"
        self._init_db()

        try:
            self.elevation_data = srtm.get_data()
        except:
            self.elevation_data = None
            
    def _init_db(self):
        """Initialize SQLite table for weather caching."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                key TEXT PRIMARY KEY,
                temp REAL,
                precip REAL
            )
        """)
        conn.close()

    def _get_cached_weather(self, key):
        conn = sqlite3.connect(self.db_path)
        res = conn.execute("SELECT temp, precip FROM weather WHERE key = ?", (key,)).fetchone()
        conn.close()
        return res

    def _save_to_cache(self, key, temp, precip):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO weather VALUES (?, ?, ?)", (key, temp, precip))
        conn.commit()
        conn.close()

    def run_all(self):
        self.logger.info("--- 🌍 Starting Geo-Context Enrichment (Cached) ---")
        self._add_astronomical_features()
        self._add_elevation_local()
        self._add_historical_weather()
        return self.adata

    def _add_astronomical_features(self):
        self.logger.info("   ☀️  Calculating solar/seasonal features...")
        # Check if lat column exists; skip if not
        if 'lat' not in self.obs.columns:
            self.logger.warning("   ⚠️  No 'lat' column found. Skipping astronomical features.")
            return
        def get_day_length(row):
            try:
                lat, date_obj = float(row.get('lat')), pd.to_datetime(row.get('collection_date'))
                if pd.isna(lat) or pd.isna(date_obj): return np.nan
                doy = date_obj.timetuple().tm_yday
                p = math.pi / 180
                m = 1 - math.tan(lat * p) * math.tan(23.44 * p * math.cos((doy - 172) * 2 * math.pi / 365))
                return 24 * (1 - math.acos(max(0, min(1-m, 2))) / math.pi)
            except: return np.nan
        self.obs['calc_day_length_hours'] = self.obs.apply(get_day_length, axis=1)

    def _add_elevation_local(self):
        # Check if lat/lon columns exist; skip if not
        if 'lat' not in self.obs.columns or 'lon' not in self.obs.columns:
            self.logger.warning("   ⚠️  No 'lat'/'lon' columns found. Skipping elevation lookup.")
            return
        mask = self.obs['lat'].notnull() & self.obs['lon'].notnull() & self.obs.get('elevation_m', pd.Series(np.nan)).isnull()
        if not mask.any() or self.elevation_data is None: return
        self.logger.info(f"   🏔️  SRTM Lookup: {mask.sum()} samples...")
        self.obs.loc[mask, 'elevation_m'] = self.obs[mask].apply(lambda r: self.elevation_data.get_elevation(r['lat'], r['lon']), axis=1)

    def _add_historical_weather(self):
        # Check if lat/lon columns exist; skip if not
        if 'lat' not in self.obs.columns or 'lon' not in self.obs.columns:
            self.logger.warning("   ⚠️  No 'lat'/'lon' columns found. Skipping weather lookup.")
            return

        # Check if collection_date exists and has data
        if 'collection_date' not in self.obs.columns:
            self.logger.warning("   ⚠️  No 'collection_date' column found. Skipping weather lookup.")
            return

        date_col = pd.to_datetime(self.obs['collection_date'], errors='coerce')
        if date_col.isnull().all():
            self.logger.warning("   ⚠️  All collection_date values are null/invalid. Skipping weather lookup.")
            return

        # FIX #1: Only include rows where BOTH lat AND lon AND date are valid (not just lat)
        valid_mask = (self.obs['lat'].notnull() & self.obs['lon'].notnull() & date_col.notnull())

        if not valid_mask.any():
            self.logger.warning("   ⚠️  No valid lat/lon/date combinations. Skipping weather lookup.")
            return

        self.logger.info("   🌦️  Fetching weather (SQLite-backed)...")

        # Create weather_key only for valid rows
        self.obs['weather_key'] = pd.NA
        valid_subset = self.obs.loc[valid_mask]
        weather_keys = (valid_subset['lat'].round(1).astype(str) + "|" +
                       valid_subset['lon'].round(1).astype(str) + "|" +
                       date_col[valid_mask].dt.strftime('%Y-%m-%d'))
        self.obs.loc[valid_mask, 'weather_key'] = weather_keys

        unique_keys = self.obs['weather_key'].dropna().unique()
        if len(unique_keys) == 0:
            self.logger.warning("   ⚠️  No valid weather keys generated. Skipping weather.")
            if 'weather_key' in self.obs.columns:
                del self.obs['weather_key']
            return

        count_new = 0
        count_failed = 0

        for key in unique_keys:
            # FIX #2: Defensive check - ensure key is a string (should be after above fixes)
            if not isinstance(key, str):
                self.logger.debug(f"   Weather API skipped for invalid key type: {type(key).__name__}")
                count_failed += 1
                continue

            cached = self._get_cached_weather(key)
            if cached:
                continue

            # API Fetch if not in DB
            try:
                # FIX #3: Defensive split with length check
                parts = key.split('|')
                if len(parts) != 3:
                    self.logger.debug(f"   Weather API skipped: invalid key format '{key}' (expected 3 parts, got {len(parts)})")
                    count_failed += 1
                    continue

                lat, lon, date = parts

                # FIX #4: Retry logic with exponential backoff for timeout errors
                max_retries = 3
                retry_delay = 0.5

                for attempt in range(max_retries):
                    try:
                        r = requests.get(
                            "https://archive-api.open-meteo.com/v1/archive",
                            params={
                                "latitude": lat,
                                "longitude": lon,
                                "start_date": date,
                                "end_date": date,
                                "daily": "temperature_2m_mean,precipitation_sum",
                                "timezone": "auto"
                            },
                            timeout=10  # FIX #5: Increased from 5 to 10 seconds
                        )
                        if r.status_code == 200:
                            d = r.json().get('daily', {})
                            temp = np.mean(d.get('temperature_2m_mean', [np.nan]))
                            precip = np.sum(d.get('precipitation_sum', [0]))
                            self._save_to_cache(key, temp, precip)
                            count_new += 1
                            time.sleep(0.1)  # Open-Meteo rate limiting
                            break  # Success - exit retry loop
                        else:
                            self.logger.debug(f"   Weather API returned {r.status_code} for {key}")
                            count_failed += 1
                            break  # Don't retry on non-200 responses

                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            # Wait with exponential backoff and retry
                            wait_time = retry_delay * (2 ** attempt)
                            self.logger.debug(f"   Weather API timeout for {key}, retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            # Final attempt failed
                            self.logger.debug(f"   Weather API failed after {max_retries} attempts for {key}")
                            count_failed += 1

            except Exception as e:
                self.logger.debug(f"   Weather API failed for key '{key}': {e}. Will use cache if available.")
                count_failed += 1
                continue

        # Final Join
        conn = sqlite3.connect(self.db_path)
        cache_df = pd.read_sql("SELECT * FROM weather", conn)
        conn.close()

        if cache_df.empty:
            self.logger.warning("   ⚠️  Weather cache is empty. Skipping weather columns to avoid NaN contamination.")
            # Remove weather_key to avoid incomplete data
            if 'weather_key' in self.obs.columns:
                del self.obs['weather_key']
            return

        # Only merge if we have data to merge
        before_merge = len(self.obs)
        self.obs = self.obs.merge(cache_df.rename(columns={'temp': 'weather_temp_avg', 'precip': 'weather_precip_sum'}),
                                 left_on='weather_key', right_on='key', how='left')
        self.adata.obs = self.obs

        # Check merge quality
        weather_cols = ['weather_temp_avg', 'weather_precip_sum']
        valid_weather = self.obs[weather_cols].notna().sum(axis=1) > 0
        self.logger.info(f"   ✅ Weather complete. ({count_new} new fetched, {len(unique_keys)-count_new-count_failed} from cache, {count_failed} failed). {valid_weather.sum()}/{before_merge} samples with weather data.")

def run_enrichment(adata):
    return GeoContextEnricher(adata).run_all()
