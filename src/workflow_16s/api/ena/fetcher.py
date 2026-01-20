# workflow_16s/api/ena/fetcher.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

# Third Party Imports
import aiohttp

# Local Imports
from .cache import CacheManager
from .constants import ENA_API_URL, BIOSAMPLES_API_URL
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class OptimizedENAFetcher:
    """
    Optimized ENA data fetcher with async requests and caching.
    
    Manages aiohttp sessions, concurrent requests, and API-specific logic
    for fetching data from ENA and BioSamples.
    """
    def __init__(
        self, email: str, max_concurrent: int = 10, chunk_size: int = 100, 
        progress: Any = None, cache_manager: Optional[CacheManager] = None
    ):
        self.email = email
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.session: Optional[aiohttp.ClientSession] = None
        self.biosamples_cache: Dict[str, Dict] = {}
        self.progress = progress
        self.cache_manager = cache_manager

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent, limit_per_host=self.max_concurrent,
            ttl_dns_cache=300, use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=120, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout,
            headers={"User-Agent": f"PythonClient/1.0 ({self.email})"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: await self.session.close()

    async def fetch_ena_batch(
        self, result_type: str, query_key: str, accessions: List[str]
    ) -> List[Dict]:
        """Async batch fetch from ENA API with concurrent chunk processing."""
        if not accessions: return []

        unique_accessions = list(set(accessions))
        chunks = [
            unique_accessions[i:i + self.chunk_size]
            for i in range(0, len(unique_accessions), self.chunk_size)
        ]

        logger.debug(f"Fetching {result_type} data in {len(chunks)} chunks...")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._fetch_chunk(semaphore, result_type, query_key, chunk, i + 1, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing failed: {result}")
            elif isinstance(result, list): all_results.extend(result)

        return all_results

    async def _fetch_chunk(
        self, semaphore: asyncio.Semaphore, result_type: str, query_key: str, 
        chunk: List[str], chunk_num: int, total_chunks: int
    ) -> List[Dict]:
        """Fetch a single chunk with caching, rate limiting, and session validation."""
        async with semaphore:
            cache_key = None
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(
                    "ena_chunk", result_type, query_key, sorted(chunk)
                )
                cached_data = await self.cache_manager.get(cache_key) 
                if cached_data is not None:
                    logger.debug(f"   Cache HIT for {result_type} chunk {chunk_num}/{total_chunks}")
                    return cached_data
            
            if self.session is None or self.session.closed:
                logger.error(f"Session closed when fetching {result_type} chunk {chunk_num}")
                return []
            
            logger.debug(f"   Fetching {result_type} chunk {chunk_num}/{total_chunks} ({len(chunk)} accessions)")
            
            query = " OR ".join(f'{query_key}="{acc}"' for acc in chunk)
            params = {
                "result": result_type, "query": query, "fields": "all",
                "format": "json", "limit": 0
            }
            
            try:
                async with self.session.get(ENA_API_URL, params=params) as response:
                    if response.status == 204: return []
                    response.raise_for_status()
                    data = await response.json()
                    logger.debug(f"       Found {len(data)} results for chunk {chunk_num}")
                    
                    if self.cache_manager and data and cache_key is not None:
                        await self.cache_manager.set(cache_key, data)
                    
                    return data
            except asyncio.CancelledError:
                logger.warning(f"Request cancelled for {result_type} chunk {chunk_num}")
                return []
            except Exception as e:
                logger.error(f"Error fetching chunk {chunk_num}: {e}")
                return []

    async def fetch_biosamples_batch(self, accessions: List[str]) -> Dict[str, Dict]:
        """Async batch fetch from BioSamples API with caching."""
        if not accessions: return {}

        unique_accessions = list(set(accessions))
        uncached_accessions = [acc for acc in unique_accessions if acc not in self.biosamples_cache]

        if not uncached_accessions:
            return {acc: self.biosamples_cache[acc] for acc in accessions if acc in self.biosamples_cache}

        logger.debug(f"Fetching BioSamples data for {len(uncached_accessions)} accessions...")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_biosample(acc):
            result = await self._fetch_biosample(semaphore, acc)
            return acc, result

        tasks = [fetch_biosample(acc) for acc in uncached_accessions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                acc, data = result
                if isinstance(data, dict): self.biosamples_cache[acc] = data
            elif isinstance(result, Exception):
                logger.error(f"BioSample fetch failed: {result}")
        
        return {acc: self.biosamples_cache.get(acc, {}) for acc in accessions}

    async def _fetch_biosample(self, semaphore: asyncio.Semaphore, accession: str) -> Dict:
        """Fetch a single BioSample with caching, rate limiting, and session validation."""
        async with semaphore:
            cache_key = None
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key("biosample", accession)
                cached_data = await self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.biosamples_cache[accession] = cached_data
                    return cached_data

            if self.session is None or self.session.closed:
                logger.error(f"Session closed when fetching BioSample {accession}")
                return {}
            
            url = f"{BIOSAMPLES_API_URL}{accession}"
            
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as response:
                    if response.status in [404, 204]: data_to_cache = {}
                    else:
                        response.raise_for_status()
                        data = await response.json()
                        characteristics = data.get("characteristics", {})
                        data_to_cache = self._flatten_biosamples_characteristics(characteristics)

                    if self.cache_manager and cache_key is not None:
                        await self.cache_manager.set(cache_key, data_to_cache)
                        
                    return data_to_cache
            except asyncio.CancelledError:
                logger.warning(f"Request cancelled for BioSample {accession}")
                return {}
            except Exception as e:
                if "404" not in str(e) and "Session is closed" not in str(e):
                    logger.error(f"Error fetching BioSample {accession}: {e}")
                return {}

    @staticmethod
    def _flatten_biosamples_characteristics(characteristics: Dict) -> Dict:
        flat_characteristics = {}
        for key, value_list in characteristics.items():
            if value_list and isinstance(value_list, list):
                text_value = value_list[0].get("text")
                if text_value:
                    clean_key = f"biosample_{key.lower().replace(' ', '_')}"
                    flat_characteristics[clean_key] = text_value
        return flat_characteristics