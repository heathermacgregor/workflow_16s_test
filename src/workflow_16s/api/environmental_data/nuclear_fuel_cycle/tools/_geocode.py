# workflow_16s/api/environmental_data/nuclear_fuel_cycle/tools/_geocode.py

import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from workflow_16s.config import AppConfig
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import with_logger, get_logger

from ...other.tools.cache import CacheManager

@with_logger
class GeocodingService:
    """
    Async wrapper for Nominatim geocoding.
    Now utilizes the unified SQLite CacheManager for robust persistence.
    """
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    
    def __init__(self, config: AppConfig, output_dir: Path):
        self.logger = get_logger("workflow_16s")
        self.user_agent = config.web.user_agent or "workflow_16s/1.0 (Research)"
        
        # 🚀 REFACTORED: Use the common SQLite 'env' cache
        # This allows geocodes to be shared between Arkin, NFC, and standard modules.
        cache_path = output_dir.parent.parent / "cache" / "env"
        self.cache_manager = CacheManager(cache_path)
        
    async def _fetch_single(self, session: aiohttp.ClientSession, query: str) -> Optional[Dict[str, Any]]:
        """Fetches coordinates for a single query string with SQLite look-up."""
        if not query or not isinstance(query, str):
            return None
            
        clean_query = query.strip()
        cache_key = f"geocode_{clean_query.lower().replace(' ', '_')}"
        
        # 1. Check Unified Cache
        if (cached_result := self.cache_manager.get(cache_key)) is not None:
            return cached_result

        # 2. Rate Limit Logic (Nominatim requires 1 req/sec)
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
                            'country': data[0].get('address', {}).get('country'),
                            'display_name': data[0].get('display_name')
                        }
                        # 🚀 Save immediately to SQLite (No more pickle loss!)
                        self.cache_manager.set(cache_key, result)
                        return result
                    else:
                        # Cache negative result (as empty dict) to prevent redundant API hits
                        self.cache_manager.set(cache_key, {})
                elif resp.status == 429:
                    self.logger.warning(f"Nominatim Rate limit hit. Backing off 5s for: {clean_query}")
                    await asyncio.sleep(5)
                    return await self._fetch_single(session, query)
        except Exception as e:
            self.logger.debug(f"Geocoding failed for '{query}': {e}")
        
        return None

    async def geocode_batch(self, queries: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Geocodes a list of strings using a serial semaphore for rate-limit safety."""
        unique_queries = list(set(queries))
        self.logger.info(f"Geocoding: Processing {len(unique_queries)} unique facility names.")
        
        # Nominatim strictly forbids concurrent requests from the same IP.
        # We use a Semaphore of 1 to act as a serial queue.
        sem = asyncio.Semaphore(1) 
        
        async with aiohttp.ClientSession() as session:
            # Integration with your Rich Progress Bar
            with get_progress_bar() as progress:
                task_id = progress.add_task("[yellow]Geocoding Facilities...", total=len(unique_queries))
                
                for q in unique_queries:
                    async with sem:
                        await self._fetch_single(session, q)
                        progress.update(task_id, advance=1)
                        # Required politeness delay for Nominatim
                        await asyncio.sleep(1.1) 
                
        # Return results by retrieving them from the now-populated SQLite cache
        return [self.cache_manager.get(f"geocode_{q.strip().lower().replace(' ', '_')}") for q in queries]