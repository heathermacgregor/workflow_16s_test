# workflow_16s/api/ena/fetcher.py

import asyncio
import time
from hashlib import sha256
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from .cache import SQLiteCacheManager
from .constants import ENA_API_URL, BIOSAMPLES_API_URL
from .metadata import get_samples_by_location_async

# Phase 1: Essential fields (13 fields) - fetched at startup for speed
PHASE1_FIELDS = (
    "accession,secondary_sample_accession,"
    "collection_date,lat,lon,country,"
    "scientific_name,tax_id,"
    "isolation_source,host,"
    "sample_alias,description,"
    "first_public"
)

# Phase 2: Extended fields (38 additional fields) - fetched async in background
PHASE2_FIELDS = (
    "bio_material,cell_line,cell_type,"
    "collected_by,cultivar,culture_collection,"
    "dev_stage,ecotype,environmental_sample,"
    "germline,host_body_site,host_genotype,host_phenotype,host_sex,"
    "host_status,host_tax_id,identified_by,isolate,"
    "mating_type,project_name,protocol_label,"
    "sample_collection,sample_material,sample_title,serotype,"
    "serovar,sex,specimen_voucher,strain,sub_species,sub_strain,"
    "tissue_lib,tissue_type,variety"
)

# All fields (Phase 1 + Phase 2)
ALL_FIELDS = PHASE1_FIELDS + "," + PHASE2_FIELDS

class ENAFetcher:
    """ENA data fetcher."""
    def __init__(
        self, 
        email: str, 
        max_concurrent: int = 10, 
        chunk_size: int = 100, 
        progress: Any = None, 
        progress_task_id: Optional[int] = None,
        cache_manager: Optional[SQLiteCacheManager] = None,
        log_interval: int = 10,
        fetch_phases: bool = True,
        **kwargs
    ):
        from workflow_16s.utils.logger import get_logger
        self.logger = get_logger("workflow_16s")
        self.email = email
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.session: Optional[aiohttp.ClientSession] = None
        self.biosamples_cache: Dict[str, Dict] = {} # In-memory hot cache
        self.progress = progress
        self.progress_task_id = progress_task_id
        self.cache_manager = cache_manager
        self.log_interval = log_interval
        self.last_log_time = 0.0
        self.max_retries = 3
        self.initial_backoff = 2
        self.fetch_phases = fetch_phases

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generates a stable SHA256 hash for SQLite keys."""
        representation = f"{prefix}:" + ":".join(map(str, args))
        return sha256(representation.encode()).hexdigest()

    def _update_progress(self, advance: int = 1, message: Optional[str] = None):
        """
        Safely update progress if a progress bar is available.
        
        Args:
            advance: Number of items to advance the progress bar (default: 1)
            message: Optional message to log if progress tracking is unavailable
        """
        if self.progress is not None and self.progress_task_id is not None:
            try:
                self.progress.update(self.progress_task_id, advance=advance)
            except Exception as e:
                # Progress bar might have been closed, fail gracefully
                if message:
                    self.logger.debug(f"{message} (progress update failed: {e})")
        elif message:
            self.logger.debug(message)

    def _throttled_log(self, message: str, level: str = "info"):
        """DEPRECATED: Use logger.warning/info directly instead.
        
        This method is kept for backward compatibility but should not be used
        in new code. Use logger.warning/info + progress bar updates instead.
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(message)
            self.last_log_time = current_time

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent, 
            limit_per_host=self.max_concurrent,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=150, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={"User-Agent": f"workflow_16s Client/1.0 ({self.email})"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session: await self.session.close()

    async def fetch_ena_batch(self, *args, **kwargs):
        """Alias for the newly named batch function."""
        # **kwargs passes the progress bar info down
        return await self.fetch_ena_data_in_batches(*args, **kwargs)

    async def fetch_taxonomies(self, taxon_ids: List[str], **kwargs) -> List[Dict]:
        """Restores the missing taxonomy fetcher."""
        if not taxon_ids: return []
        return await self.fetch_ena_data_in_batches(
            result_type="taxonomy", query_key="tax_id",
            accessions=taxon_ids, **kwargs 
        )
        
    async def find_nearby_samples(self, lat: float, lon: float, radius: float = 10.0) -> pd.DataFrame:
        """
        Fixed: Now explicitly returns a DataFrame and properly awaits 
        the underlying location search to prevent 'coroutine' errors.
        """
        try:
            # 🟢 CRITICAL: We await the call so 'res' is a DataFrame, not a coroutine
            res = await get_samples_by_location_async(
                lat=lat, 
                lon=lon, 
                radius=radius, 
                email=self.email, 
                fetcher=self, 
                cache_manager=self.cache_manager
            )
            
            # Ensure we return a DataFrame even if the result was None
            if res is None:
                return pd.DataFrame()
            return res
            
        except Exception as e:
            # This is where your logs were catching the 'coroutine' error
            import traceback
            self.logger.error(f"Failed spatial search at ({lat}, {lon}): {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        
    async def fetch_ena_data_in_batches(self, result_type: str, query_key: str, accessions: List[str],
        fields: Optional[str] = None,
        **kwargs 
    ) -> List[Dict]:
        if not accessions: return []
        unique_accessions = list(set(accessions))
        chunks = [unique_accessions[i:i + ((self.chunk_size or 10) or 10)] for i in range(0, len(unique_accessions), ((self.chunk_size or 10) or 10))]
        
        progress_obj = kwargs.get('progress_obj', self.progress)
        with_progress = kwargs.get('with_progress_bar', False)
        task_id = None
        
        # Initialize the progress bar for the number of chunks
        if with_progress and progress_obj:
            task_id = progress_obj.add_task(f"[cyan]Fetching {result_type} batches...", total=len(chunks))

        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Wrapper to advance progress after each chunk
        async def _tracked_fetch_chunk(chunk, chunk_num):
            res = await self._fetch_chunk(semaphore, result_type, query_key, chunk, chunk_num, len(chunks))
            if with_progress and progress_obj and task_id is not None:
                progress_obj.advance(task_id, 1)
            return res

        # Execute tracked tasks
        tasks = [_tracked_fetch_chunk(chunk, i + 1) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Clean up the UI task when finished
        if with_progress and progress_obj and task_id is not None:
            progress_obj.remove_task(task_id)

        all_results = []
        for result in results:
            if isinstance(result, Exception):
                self._throttled_log(f"Chunk processing failed: {result}", level="error")
            elif isinstance(result, list): 
                all_results.extend(result)
        return all_results

    async def _fetch_chunk(self, semaphore, result_type, query_key, chunk, chunk_num, total_chunks):
        async with semaphore:
            key = self._get_cache_key("ena_chunk", result_type, query_key, sorted(chunk))
            if self.cache_manager:
                cached = await self.cache_manager.get(key)
                if cached is not None:
                    return cached
            
            if not self.session or self.session.closed: return []
            
            query = " OR ".join(f'{query_key}="{acc}"' for acc in chunk)
            params = {"result": result_type, "query": query, "fields": "all", "format": "json", "limit": 0}
            
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(ENA_API_URL, params=params) as response:
                        if response.status == 204: return []
                        response.raise_for_status()
                        data = await response.json()
                        if self.cache_manager and data:
                            await self.cache_manager.set(key, data)
                        return data
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.initial_backoff * (2 ** attempt))
            return []

    async def fetch_biosamples_batch(self, accessions: List[str], **kwargs) -> Dict[str, Dict]:
        """
        Fetch biosamples using chunked queries to the ENA search API.
        
        Uses Rich progress bar instead of throttled logging for better visibility.
        Supports both Phase 1 (essential) and full field fetching based on config.
        
        Args:
            accessions: List of sample accessions to fetch
            **kwargs: Additional arguments including:
                - chunk_size: Number of accessions per chunk (default: 50)
                - timeout: Request timeout in seconds (default: 60)
                - with_progress_bar: Enable progress tracking (default: False)
                - progress_obj: Rich Progress object for display
                - phase_1_only: Fetch only Phase 1 fields (default: False)
        
        Returns:
            Dictionary indexed by accession with fetched fields
        """
        if not accessions:
            return {}
        
        unique = list(set(accessions))
        chunk_size = kwargs.get('chunk_size', 50) 
        timeout = kwargs.get('timeout', 60)
        phase_1_only = kwargs.get('phase_1_only', False)

        chunks = [unique[i:i+chunk_size] for i in range(0, len(unique), chunk_size)]
        
        progress_obj = kwargs.get('progress_obj', getattr(self, 'progress', None))
        with_progress = kwargs.get('with_progress_bar', False)
        task_id = None
        
        if with_progress and progress_obj:
            phase_label = "Phase 1: Essential" if phase_1_only else "All Fields"
            task_id = progress_obj.add_task(f"📥 Fetching biosamples ({phase_label})...", total=len(chunks))

        # REDUCE CONCURRENCY from 10 to 5 to prevent ENA Server Disconnects
        max_concurrent = min(getattr(self, 'max_concurrent', 5), 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = []
        for chunk in chunks:
            tasks.append(
                self._fetch_biosamples_chunk(
                    semaphore, chunk, timeout, 
                    with_progress=with_progress, 
                    progress_obj=progress_obj, 
                    task_id=task_id,
                    phase_1_only=phase_1_only
                )
            )

        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        if with_progress and progress_obj and task_id:
            progress_obj.remove_task(task_id)

        merged = {}
        failed_chunks = 0
        for res in chunk_results:
            if isinstance(res, dict):
                merged.update(res)
            elif isinstance(res, Exception):
                failed_chunks += 1
                self.logger.warning(f"Chunk fetch failed: {res}")

        if not hasattr(self, 'biosamples_cache'):
            self.biosamples_cache = {}
            
        for acc, data in merged.items():
            self.biosamples_cache[acc] = data

        if failed_chunks > 0:
            self.logger.warning(f"⚠️  {failed_chunks} chunks failed during biosamples batch fetch")

        return {acc: self.biosamples_cache.get(acc, {}) for acc in accessions if acc in self.biosamples_cache}

    async def fetch_biosamples_phase1(self, accessions: List[str], **kwargs) -> Dict[str, Dict]:
        """
        Fetch only Phase 1 (essential) fields for fast initial load via selective column request.
        
        This is the primary Phase 1 method optimized for minimal network payload.
        Phase 1 includes: accession, secondary_sample_accession, collection_date,
        lat, lon, country, scientific_name, tax_id, isolation_source, host,
        sample_alias, description, first_public, submission_date, primary_accession.
        
        Phase 2 extended fields must be fetched separately via fetch_biosamples_batch().
        
        Args:
            accessions: List of sample accessions to fetch
            **kwargs: Additional arguments (chunk_size, timeout, with_progress_bar, progress_obj, progress_task_id, etc.)
        
        Returns:
            Dictionary indexed by accession with Phase 1 fields only
        """
        # Force phase_1_only=True to use PHASE1_FIELDS instead of ALL_FIELDS
        kwargs['phase_1_only'] = True
        return await self.fetch_biosamples_batch(accessions, **kwargs)

    async def fetch_biosamples_batch_essential(self, accessions: List[str], **kwargs) -> Dict[str, Dict]:
        """
        Alias for fetch_biosamples_phase1() for backward compatibility.
        
        Args:
            accessions: List of sample accessions to fetch
            **kwargs: Additional arguments (chunk_size, timeout, with_progress_bar, etc.)
        
        Returns:
            Dictionary indexed by accession with Phase 1 fields only
        """
        return await self.fetch_biosamples_phase1(accessions, **kwargs)


    async def _fetch_biosamples_chunk(self, semaphore, chunk, timeout, with_progress=False, progress_obj=None, task_id=None, phase_1_only=False):
        """
        Fetch one chunk of biosamples via ENA search API with Exponential Backoff Retries.
        
        Args:
            semaphore: Async semaphore for concurrency control
            chunk: List of accessions to fetch in this chunk
            timeout: Request timeout in seconds
            with_progress: Whether to update progress bar
            progress_obj: Rich Progress object (if with_progress=True)
            task_id: Task ID for progress tracking
            phase_1_only: If True, fetch only Phase 1 (essential) fields; else all fields
        """
        import asyncio
        import aiohttp
        ENA_SEARCH_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
        
        async with semaphore:
            key = None
            if hasattr(self, '_get_cache_key'):
                key = self._get_cache_key("biosample_chunk", sorted(chunk))
                
            if hasattr(self, 'cache_manager') and self.cache_manager and key:
                cached = await self.cache_manager.get(key)
                if cached is not None:
                    return cached

            # Build query: check BOTH primary and secondary accessions
            # ENA uses ERS for primary, and SAMN/SRS for secondary
            query_parts = []
            for acc in chunk:
                query_parts.append(f'(accession="{acc}" OR secondary_sample_accession="{acc}")')
            
            query = " OR ".join(query_parts)

            # Select fields based on phase_1_only flag
            if phase_1_only:
                # Phase 1: Only essential fields (15 fields) for fast fetch
                fields_to_fetch = PHASE1_FIELDS
            else:
                # All fields: Full metadata (53 fields)
                all_valid_ena_fields = (
                    "accession,secondary_sample_accession,bio_material,cell_line,cell_type,"
                    "collected_by,collection_date,country,cultivar,culture_collection,"
                    "description,dev_stage,ecotype,environmental_sample,first_public,"
                    "germline,host,host_body_site,host_genotype,host_phenotype,host_sex,"
                    "host_status,host_tax_id,identified_by,isolate,isolation_source,lat,lon,"
                    "mating_type,project_name,protocol_label,sample_alias,"
                    "sample_collection,sample_material,sample_title,scientific_name,serotype,"
                    "serovar,sex,specimen_voucher,strain,sub_species,sub_strain,"
                    "tax_id,tissue_lib,tissue_type,variety"
                )
                fields_to_fetch = all_valid_ena_fields
            
            payload = {
                "result": "sample",
                "query": query,
                "fields": fields_to_fetch,
                "format": "json",
                "limit": 0,
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            data = []
            
            # --- NEW: 3-Attempt Retry Loop for Server Disconnects ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with self.session.post(ENA_SEARCH_URL, data=payload, headers=headers, timeout=client_timeout) as resp:
                        if resp.status == 204:
                            return {}
                        if resp.status != 200:
                            err_text = await resp.text()
                            if hasattr(self, '_throttled_log'):
                                self._throttled_log(f"ENA API Error {resp.status}: {err_text}", level="error")
                            return {}
                        
                        data = await resp.json()
                        break # Success! Break out of the retry loop.
                        
                except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
                    if attempt == max_retries - 1: # If this was the last attempt, fail gracefully
                        if hasattr(self, '_throttled_log'):
                            self._throttled_log(f"Failed to fetch chunk after {max_retries} attempts: {e}", level="error")
                        return {}
                    # Wait 2 seconds, then 4 seconds before retrying
                    await asyncio.sleep(2 ** (attempt + 1)) 

            # Parse the JSON
            result = {}
            for item in data:
                prim_acc = item.get('accession') or item.get('primary_accession')
                sec_acc = item.get('secondary_sample_accession')
                
                if not prim_acc and not sec_acc:
                    continue
                    
                flat_data = {k: v for k, v in item.items() if v and k != 'accession'}
                
                # Save the data under BOTH IDs so the caller can find it 
                # no matter which format (SRR/SAMN/ERS) it originally requested
                if prim_acc:
                    result[prim_acc] = flat_data
                if sec_acc:
                    result[sec_acc] = flat_data

            if hasattr(self, 'cache_manager') and self.cache_manager and result and key:
                await self.cache_manager.set(key, result)

            if with_progress and progress_obj and task_id:
                progress_obj.advance(task_id, 1)

            # Update progress bar instead of throttled logging
            if self.progress and self.progress_task_id is not None:
                self.progress.update(self.progress_task_id, advance=len(result))
            
            return result

    async def _fetch_biosample(self, semaphore, accession):
        async with semaphore:
            key = self._get_cache_key("biosample", accession)
            if self.cache_manager:
                cached = await self.cache_manager.get(key)
                if cached is not None:
                    return {**cached, "accession": accession}

            if not self.session or self.session.closed: return {}
            
            url = f"{BIOSAMPLES_API_URL}{accession}"
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                        if resp.status in [404, 204]: return {"accession": accession}
                        resp.raise_for_status()
                        data = await resp.json()
                        flat = self._flatten_biosamples_characteristics(data.get("characteristics", {}))
                        if self.cache_manager and flat:
                            await self.cache_manager.set(key, flat)
                        return {**flat, "accession": accession}
                except Exception:
                    await asyncio.sleep(self.initial_backoff * (2 ** attempt))
            return {"accession": accession}

    @staticmethod
    def _flatten_biosamples_characteristics(characteristics: Dict) -> Dict:
        """Safely flatten biosamples characteristics, handling edge cases."""
        result = {}
        for k, v in characteristics.items():
            try:
                if isinstance(v, list) and len(v) > 0:
                    # Handle list of dicts
                    item = v[0]
                    if isinstance(item, dict):
                        text_val = item.get("text") if isinstance(item, dict) else str(item)
                    else:
                        text_val = str(item)
                    
                    if text_val:  # Only add if non-empty
                        result[f"biosample_{k.lower().replace(' ', '_')}"] = text_val
                elif v:
                    # Handle direct values that aren't lists
                    result[f"biosample_{k.lower().replace(' ', '_')}"] = str(v)
            except (TypeError, KeyError, IndexError) as e:
                # Log and skip problematic characteristics
                pass
        
        return result
    
