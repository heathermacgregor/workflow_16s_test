# workflow_16s/api/ena/metadata/fetcher.py

# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

# Third Party Imports
import aiohttp
import pandas as pd # Ensure pandas is imported
from Bio import Entrez

# Local Imports
from .cache import CacheManager
from .constants import ENA_API_URL, BIOSAMPLES_API_URL
from workflow_16s.utils.progress import get_progress_bar
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ==================================================================================== #

class ENAFetcher:
    """
    Handles all asynchronous data fetching from ENA, BioSamples, and NCBI,
    with integrated caching, concurrency control, and automatic retries.
    """
    def __init__(self, email: str, max_concurrent: int, cache_manager: Optional[CacheManager] = None):
        self.email = email
        Entrez.email = self.email # Set for NCBI calls
        self.max_concurrent = max_concurrent
        self.cache_manager = cache_manager
        self.session: Optional[aiohttp.ClientSession] = None
        # Semaphores to control concurrency for different services
        self.semaphore = asyncio.Semaphore(self.max_concurrent) # General semaphore
        self.ncbi_semaphore = asyncio.Semaphore(min(10, max_concurrent)) # Stricter limit for NCBI
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_count': 0,
            'rate_limit_hits': 0
        }

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2, # Allow more potential connections
            ttl_dns_cache=300, # Cache DNS lookups
        )
        # Increased total timeout to better handle potential retries/delays
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": f"PythonClient/1.0 ({self.email})"} # Identify client
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Log final statistics
        if self.stats['total_requests'] > 0:
            logger.info(f"ENA Fetcher Statistics:")
            logger.info(f"  Total requests: {self.stats['total_requests']}")
            logger.info(f"  Cached: {self.stats['cached_requests']} ({self.stats['cached_requests']/self.stats['total_requests']*100:.1f}%)")
            logger.info(f"  Successful: {self.stats['successful_requests']} ({self.stats['successful_requests']/self.stats['total_requests']*100:.1f}%)")
            logger.info(f"  Failed: {self.stats['failed_requests']}")
            logger.info(f"  Retries: {self.stats['retry_count']}")
            logger.info(f"  Rate limit hits: {self.stats['rate_limit_hits']}")

    async def _fetch_json(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Any:
        """JSON fetcher with rate limiting, retries, exponential backoff, and statistics tracking."""
        max_retries = 4
        base_delay = 2 # Initial delay in seconds for retries

        last_exception: Optional[Exception] = None # Store last exception for final logging
        self.stats['total_requests'] += 1

        for attempt in range(max_retries):
            # Use the general semaphore for ENA/BioSamples requests
            async with self.semaphore:
                if not self.session or self.session.closed:
                    logger.error("HTTP session is closed or not initialized during fetch.")
                    self.stats['failed_requests'] += 1
                    # Don't retry if session is fundamentally broken
                    raise IOError("Session is not active.")
                try:
                    logger.debug(f"Fetching (Attempt {attempt+1}/{max_retries}): {url} with params {params}")
                    async with self.session.get(url, params=params, headers=headers) as response:
                        # Handle specific HTTP status codes
                        if response.status == 204: # No Content
                            logger.debug(f"Received 204 No Content for {url}")
                            self.stats['successful_requests'] += 1
                            return [] # Success, empty result
                        if response.status == 404: # Not Found
                            logger.warning(f"Received 404 Not Found for {url} with params {params}")
                            self.stats['successful_requests'] += 1
                            return [] # Treat as empty result for searches, don't retry
                        if response.status == 429: # Rate Limited
                            self.stats['rate_limit_hits'] += 1
                            self.stats['retry_count'] += 1
                            # Use Retry-After header if available, otherwise exponential backoff
                            try:
                                retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                            except (ValueError, TypeError):
                                retry_after = base_delay * (2 ** attempt) # Fallback if header invalid
                            logger.warning(f"Rate limited (429) by {url}. Retrying after {retry_after} seconds...")
                            last_exception = aiohttp.ClientResponseError(response.request_info, response.history, status=429, message="Rate Limited")
                            await asyncio.sleep(retry_after)
                            continue # Go to the next retry attempt

                        # Raise errors for other client/server issues (4xx besides 404/429, 5xx)
                        response.raise_for_status()

                        # Process successful response (200 OK)
                        response_text = await response.text()
                        # Check for empty response body before parsing JSON
                        if not response_text:
                            logger.debug(f"Received empty response body (status {response.status}) for {url}")
                            return []
                        try:
                            # Attempt to parse JSON
                            return json.loads(response_text)
                        except json.JSONDecodeError as json_err:
                            # Log detailed error if JSON parsing fails
                            logger.error(f"JSON decode failed for {url}. Status: {response.status}. Response text (start): {response_text[:500]}...")
                            last_exception = json_err # Store error
                            # Optionally retry on decode error? For now, let's treat it as a failure for this attempt.
                            # If you want to retry: await asyncio.sleep(...); continue
                            raise json_err # Re-raise to be caught by outer handler

                # Handle network/timeout errors
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e # Store exception
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Request failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    # No else needed, loop will terminate and raise outside

                except asyncio.CancelledError:
                    logger.warning(f"Request cancelled for {url}")
                    raise # Re-raise cancellation

        # If loop finishes without returning (i.e., all retries failed)
        logger.error(f"HTTP request failed after {max_retries} attempts for {url}. Last error: {last_exception}")
        # Depending on desired behavior, either return empty or raise the last exception
        # Returning empty list to indicate failure to retrieve data
        return []
        # Or: raise last_exception if last_exception else aiohttp.ClientError(f"Request failed after {max_retries} retries")


    async def find_nearby_samples(self, lat: float, lon: float, radius: Union[int, float]) -> List[Dict]:
        """Finds ENA samples within a geographic radius, using cache."""
        cache_key = self.cache_manager.get_cache_key(
            "nearby", 
            lat=lat, 
            lon=lon, 
            rad=radius
        ) if self.cache_manager else None

        if cache_key and self.cache_manager:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache HIT for samples near ({lat}, {lon})")
                return cached_data # Return cached list

        # Define fields to request from ENA sample search
        fields = "accession,scientific_name,collection_date,location,description,host,tax_id"
        query = f"geo_circ({lat},{lon},{radius})"
        params = {
            "result": "sample", 
            "query": query, 
            "format": "json", 
            "limit": 0, 
            "fields": fields
        }
        # Use the robust fetcher which handles retries etc.
        result = await self._fetch_json(ENA_API_URL, params)

        # Cache the result (even if it's an empty list)
        if cache_key and self.cache_manager:
            await self.cache_manager.set(cache_key, result)
        return result

    async def _fetch_ena_chunk(
        self, result_type: str, query_key: str, chunk: List[str]
    ) -> List[Dict]:
        """Fetches a single chunk from ENA API, using cache."""
        cache_key = self.cache_manager.get_cache_key(
            "ena_chunk", result_type, query_key, sorted(chunk) # Sort chunk for consistent key
        ) if self.cache_manager else None

        if cache_key and self.cache_manager:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None: return cached_data # Return cached list

        # Construct ENA API query
        query = " OR ".join(f'{query_key}="{acc}"' for acc in chunk)
        params = {
            "result": result_type, 
            "query": query, 
            "fields": "all",
            "format": "json", 
            "limit": 0 # Get all results matching query
        }
        data = await self._fetch_json(ENA_API_URL, params)

        # Cache the result (even if data is empty list)
        if cache_key and self.cache_manager:
            await self.cache_manager.set(cache_key, data)
        return data

    async def fetch_ena_data_in_batches(
        self, result_type: str, query_key: str, accessions: List[str],
        chunk_size: int = 25, # Adjusted default chunk size seems reasonable
        with_progress_bar: bool = True # Control internal progress bar
    ) -> List[Dict]:
        """Fetches ENA data concurrently in chunks, with optional progress bar."""
        if not accessions: return []
        unique_accessions = list(set(accessions)) # Process only unique IDs
        # Split unique IDs into chunks
        chunks = [unique_accessions[i:i + chunk_size] for i in range(0, len(unique_accessions), chunk_size)]
        # Create async tasks for each chunk
        tasks = [self._fetch_ena_chunk(result_type, query_key, chunk) for chunk in chunks]

        results_list = []
        desc = f"Fetching {result_type} ({len(unique_accessions)} IDs in {len(chunks)} chunks)..."

        # Use progress bar if requested
        if with_progress_bar:
            with get_progress_bar() as progress:
                task_id = progress.add_task(desc, total=len(tasks))
                # Process tasks as they complete
                for future in asyncio.as_completed(tasks):
                    try:
                        result = await future # Get result from completed future
                        if result: results_list.append(result) # Append non-empty results
                    except Exception as e:
                        # Log error from future if _fetch_json raised something unexpected
                        logger.error(f"Error awaiting ENA chunk future: {e}", exc_info=True)
                    progress.update(task_id, advance=1) # Update progress bar
        else: # Run without internal progress bar
            logger.info(desc)
            # Gather all results (or exceptions)
            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in gathered_results:
                if isinstance(result, Exception):
                    # Log exceptions returned by gather
                    logger.error(f"Error fetching ENA chunk via gather: {result}")
                elif result: # Append non-empty, non-exception results
                    results_list.append(result)

        # Flatten list of lists into a single list of dictionaries
        return [item for sublist in results_list for item in sublist]

    async def _fetch_single_biosample(self, accession: str) -> Dict:
        """Fetches and processes a single BioSample record, using cache."""
        cache_key = self.cache_manager.get_cache_key("biosample", accession) if self.cache_manager else None

        if cache_key and self.cache_manager:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None: return cached_data # Return cached dict

        url = f"{BIOSAMPLES_API_URL}{accession}"
        # Request JSON specifically
        headers = {"Accept": "application/json"}
        # Use robust fetcher which handles retries etc.
        data = await self._fetch_json(url, headers=headers)

        # Process characteristics if data is valid
        flat_characteristics = {}
        if data and isinstance(data, dict) and "characteristics" in data:
            # Ensure characteristics is a dict before iterating
            characteristics_dict = data["characteristics"]
            if isinstance(characteristics_dict, dict):
                for key, value_list in characteristics_dict.items():
                    # Ensure value_list structure is as expected before accessing 'text'
                    if (isinstance(value_list, list) and value_list and
                        isinstance(value_list[0], dict) and "text" in value_list[0]):
                        # Create prefixed, sanitized key
                        clean_key = f"biosample_{key.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                        flat_characteristics[clean_key] = value_list[0]["text"]
            else:
                logger.warning(f"Unexpected format for characteristics in BioSample {accession}: {characteristics_dict}")


        # Cache the processed characteristics (even if empty)
        if cache_key and self.cache_manager:
            await self.cache_manager.set(cache_key, flat_characteristics)
        return flat_characteristics

    async def fetch_biosamples_in_batches(self, accessions: List[str], with_progress_bar: bool = True) -> Dict[str, Dict]:
        """Fetches multiple BioSample records concurrently, with optional progress bar."""
        if not accessions: return {}
        unique_accessions = list(set(accessions)) # Fetch unique IDs only
        # Create tasks for each unique accession
        tasks = {acc: self._fetch_single_biosample(acc) for acc in unique_accessions}

        results_dict = {}
        desc = f"Fetching {len(unique_accessions)} BioSamples..."

        # Use progress bar if requested
        if with_progress_bar:
            with get_progress_bar() as progress:
                task_id = progress.add_task(desc, total=len(tasks))
                task_list = [asyncio.create_task(coro) for coro in tasks.values()]
                acc_map = {task: acc for acc, task in zip(tasks.keys(), task_list)}

                # Process as tasks complete
                for future in asyncio.as_completed(task_list):
                    acc = acc_map[future] # type: ignore
                    try:
                        # Await the future *here* inside the try block to catch task exceptions
                        result = await future
                        results_dict[acc] = result # Store result (dict)
                    except Exception as e:
                        # Log the actual exception caught from awaiting the future
                        logger.error(f"Error fetching BioSample {acc}: {e}", exc_info=True) # Add traceback
                        results_dict[acc] = {} # Store empty dict on error
                    progress.update(task_id, advance=1) # Update progress bar
        else: # Run without internal progress bar
            logger.info(desc)
            # Gather all results, catching exceptions
            gathered_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            # Map results back to accessions
            for acc, result in zip(tasks.keys(), gathered_results):
                if isinstance(result, Exception):
                    # Log the actual exception returned by gather
                    logger.error(f"Error fetching BioSample {acc} (gathered): {result}")
                    results_dict[acc] = {}
                else:
                    results_dict[acc] = result

        return results_dict

    async def _fetch_single_taxonomy(self, tax_id: int) -> Optional[str]:
        """Fetches a single taxonomy lineage from NCBI using Entrez (runs in thread)."""
        # Use NCBI-specific semaphore to limit concurrency
        async with self.ncbi_semaphore:
            cache_key = self.cache_manager.get_cache_key("taxonomy", tax_id) if self.cache_manager else None

            # Check cache first
            if cache_key and self.cache_manager:
                cached_data = await self.cache_manager.get(cache_key)
                # Note: Cache stores None for failed lookups, check explicitly
                if cached_data is not None:
                    logger.debug(f"Cache HIT for taxonomy {tax_id}")
                    return cached_data
                # Check if None is explicitly cached (indicates previous failure)
                cache_file = self.cache_manager.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cached_value = await self.cache_manager.get(cache_key)
                    if cached_value is None:
                        logger.debug(f"Cache HIT (known failure/no lineage) for taxonomy {tax_id}")
                        return None

            # --- NCBI Entrez Call (in thread pool) ---
            try:
                logger.debug(f"Fetching NCBI Taxonomy for ID: {tax_id}")
                # Use asyncio.to_thread to run the synchronous Entrez call
                handle = await asyncio.to_thread(
                    Entrez.efetch, db="taxonomy", id=str(tax_id), retmode="xml"
                )
                # Parse the result (still synchronous, but should be fast)
                records = Entrez.read(handle)
                handle.close()
                # --- End Entrez Call ---

                taxonomy_str: Optional[str] = None # Initialize
                if records and isinstance(records, list) and len(records) > 0:
                    record = records[0]
                    # Check record structure before accessing keys
                    if isinstance(record, dict):
                        lineage_list = record.get("LineageEx", []) # Safely get LineageEx
                        # Extract scientific names from the lineage list
                        if isinstance(lineage_list, list):
                            taxonomy_str = "; ".join([
                                taxon["ScientificName"] for taxon in lineage_list
                                if isinstance(taxon, dict) and "ScientificName" in taxon
                            ])
                        else:
                            logger.warning(f"Unexpected LineageEx format for tax ID {tax_id}: {lineage_list}")
                    else:
                        logger.warning(f"Unexpected record format for tax ID {tax_id}: {record}")
                else:
                    logger.warning(f"No records returned by Entrez for tax ID {tax_id}.")

                # Cache the result (string or None)
                if cache_key and self.cache_manager:
                    await self.cache_manager.set(cache_key, taxonomy_str) # Cache None on failure/not found
                return taxonomy_str

            except Exception as e:
                # Handle potential Entrez errors (network, parsing, ID invalid)
                # Make error more specific if possible (e.g., check for HTTP errors)
                logger.warning(f"Failed to fetch NCBI taxonomy for tax_id {tax_id}: {e}", exc_info=True) # Add traceback
                # Cache failure as None
                if cache_key and self.cache_manager:
                    await self.cache_manager.set(cache_key, None)
                return None

    async def fetch_taxonomies(self, tax_ids: List[Any], with_progress_bar: bool = True) -> Dict[int, Optional[str]]:
        """Fetches multiple taxonomy records concurrently, with optional progress bar."""
        if not tax_ids: return {}
        # Filter out invalid IDs (NaN, non-integer types, zero) robustly
        valid_tax_ids = set()
        for tid in tax_ids:
            try:
                # Attempt conversion to int after checking validity
                numeric_tid = pd.to_numeric(tid, errors='coerce')
                if pd.notna(numeric_tid) and numeric_tid > 0:
                    valid_tax_ids.add(int(numeric_tid))
            except (TypeError, ValueError):
                logger.debug(f"Skipping invalid tax_id format: {tid}")
                continue

        if not valid_tax_ids:
            logger.debug("No valid positive integer tax IDs found to fetch.")
            return {}

        # Create tasks for unique valid IDs
        tasks = {tid: self._fetch_single_taxonomy(tid) for tid in valid_tax_ids}

        results_dict = {}
        desc = f"Fetching {len(valid_tax_ids)} NCBI Taxonomies..."

        # Use progress bar if requested
        if with_progress_bar:
            with get_progress_bar() as progress:
                task_id = progress.add_task(desc, total=len(tasks))
                task_list = [asyncio.create_task(coro) for coro in tasks.values()]
                id_map = {task: tid for tid, task in zip(tasks.keys(), task_list)}

                for task in asyncio.as_completed(task_list):
                    tax_id = id_map[task] # type: ignore
                    try:
                        result = await task #
                        results_dict[tax_id] = result
                    except Exception as e:
                        logger.error(f"Error fetching Taxonomy {tax_id}: {e}", exc_info=True) 
                        results_dict[tax_id] = None # Store None on error
                    progress.update(task_id, advance=1)
        else: # Run without internal progress bar
            logger.info(desc)
            # Gather all results, catching exceptions
            gathered_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            # Map results back to accessions
            for tax_id, result in zip(tasks.keys(), gathered_results):
                if isinstance(result, Exception):
                    # Log the actual exception
                    logger.error(f"Error fetching Taxonomy {tax_id} (gathered): {result}")
                    results_dict[tax_id] = None
                else:
                    results_dict[tax_id] = result # Store Optional[str]

        return results_dict