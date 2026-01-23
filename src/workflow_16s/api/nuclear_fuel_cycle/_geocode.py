# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_geocode.py
# ==================================================================================== #

import asyncio
import aiohttp
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from workflow_16s.config_schema import AppConfig
from workflow_16s.utils.progress import get_progress_bar

logger = logging.getLogger("workflow_16s")

class GeocodingService:
    """
    Async wrapper for geocoding services (OpenStreetMap/Nominatim).
    Includes caching to prevent redundant API calls and respect rate limits.
    """
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    
    def __init__(self, config: AppConfig, output_dir: Path):
        self.user_agent = config.web.user_agent or "workflow_16s/1.0 (Research)"
        self.cache_dir = output_dir.parent.parent / "cache" / "nfc_geocoding"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "geocoding_cache.pkl"
        
        # Load Cache
        self.cache: Dict[str, Optional[Dict[str, float]]] = {}
        self._load_cache()
        
    def _load_cache(self):
        """Loads existing geocoding results from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached geocodes.")
            except Exception as e:
                logger.warning(f"Failed to load geocoding cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Persists cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save geocoding cache: {e}")

    async def _fetch_single(self, session: aiohttp.ClientSession, query: str) -> Optional[Dict[str, float]]:
        """Fetches coordinates for a single query string."""
        if not query or not isinstance(query, str):
            return None
            
        clean_query = query.strip()
        
        # Check Cache
        if clean_query in self.cache:
            return self.cache[clean_query]

        params = {
            'q': clean_query,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {'User-Agent': self.user_agent}

        try:
            async with session.get(self.NOMINATIM_URL, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        result = {
                            'lat': float(data[0]['lat']),
                            'lon': float(data[0]['lon']),
                            'country': data[0].get('address', {}).get('country')
                        }
                        self.cache[clean_query] = result
                        return result
                    else:
                        # Cache negative result to stop retrying bad queries
                        self.cache[clean_query] = None
                elif resp.status == 429:
                    logger.warning("Geocoding rate limit hit. Backing off...")
                    await asyncio.sleep(5)
                    return await self._fetch_single(session, query) # Retry once
        except Exception as e:
            logger.debug(f"Geocoding failed for '{query}': {e}")
        
        # Politeness sleep
        await asyncio.sleep(1.0)
        return None

    async def geocode_batch(self, queries: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Geocodes a list of location strings (e.g., "Facility Name, Country").
        """
        unique_queries = list(set(queries))
        to_fetch = [q for q in unique_queries if q not in self.cache]
        
        logger.info(f"Geocoding: {len(queries)} total, {len(to_fetch)} new queries needed.")
        
        if to_fetch:
            sem = asyncio.Semaphore(1) 
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                with get_progress_bar() as progress:
                    task_id = progress.add_task("Fetching coordinates...", total=len(to_fetch))
                    
                    for i, q in enumerate(to_fetch):
                        # Throttled execution
                        async with sem:
                            res = await self._fetch_single(session, q)
                            progress.update(task_id, advance=1)
                            # Explicit wait
                            await asyncio.sleep(1.1) 
                        
                        # [FIX] Save periodically so Ctrl+C doesn't kill progress
                        if i % 10 == 0:
                            self._save_cache()
                
            # Final save
            self._save_cache()
            
        return [self.cache.get(q) for q in queries]