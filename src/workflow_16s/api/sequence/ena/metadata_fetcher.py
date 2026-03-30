"""
ENA/SRA Metadata Fetcher with Rate Limiting and Intelligent Merging.

Provides rate-limited access to ENA and SRA metadata with:
- Token bucket rate limiting (per-endpoint)
- Batch operations for efficient API usage
- Intelligent metadata merging (ENA priority, SRA fills gaps)
- Comprehensive error handling and caching
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from workflow_16s.utils.logger import get_logger
from .cache import SQLiteCacheManager


logger = get_logger("workflow_16s")


class RateLimitTier(Enum):
    """Rate limiting tiers based on authentication level."""
    UNAUTHENTICATED = (3, 5)  # (min_req_per_sec, max_req_per_sec)
    AUTHENTICATED = (5, 10)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per endpoint."""
    min_rate: float = 3.0  # Minimum conservative rate (requests/sec)
    max_rate: float = 5.0  # Maximum rate (requests/sec)
    burst_size: int = 10   # Maximum tokens to accumulate
    refill_interval: float = 0.1  # How often to refill tokens (seconds)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with per-endpoint support.

    More accurate than fixed-wait throttling, supports burst traffic,
    and tracks rate limit compliance automatically.
    """

    def __init__(self, config: RateLimitConfig, endpoint_name: str = "default"):
        """
        Initialize token bucket limiter.

        Args:
            config: RateLimitConfig with rate parameters
            endpoint_name: Name of endpoint (e.g., "ENA", "SRA") for logging
        """
        self.config = config
        self.endpoint_name = endpoint_name
        self.tokens = config.burst_size  # Start with full bucket
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
        self.request_count = 0
        self.denied_count = 0

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from bucket. Waits if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)
            timeout: Maximum time to wait (seconds). None = wait indefinitely

        Returns:
            True if tokens acquired, False if timeout exceeded
        """
        async with self.lock:
            start_time = time.monotonic()

            while self.tokens < tokens:
                # Calculate refill amount
                now = time.monotonic()
                time_since_refill = now - self.last_refill
                tokens_to_add = time_since_refill * self.config.max_rate

                self.tokens = min(
                    self.tokens + tokens_to_add,
                    self.config.burst_size
                )
                self.last_refill = now

                # Check timeout
                if timeout is not None:
                    elapsed = now - start_time
                    if elapsed > timeout:
                        self.denied_count += 1
                        logger.warning(
                            f"Rate limit timeout on {self.endpoint_name} "
                            f"(waited {elapsed:.2f}s for {tokens} tokens)"
                        )
                        return False

                # Sleep briefly before next check
                sleep_time = min(0.01, (tokens - self.tokens) / self.config.max_rate)
                await asyncio.sleep(sleep_time)

            # Consume tokens
            self.tokens -= tokens
            self.request_count += 1
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "endpoint": self.endpoint_name,
            "total_requests": self.request_count,
            "denied_requests": self.denied_count,
            "current_tokens": self.tokens,
            "config": {
                "min_rate": self.config.min_rate,
                "max_rate": self.config.max_rate,
                "burst_size": self.config.burst_size,
            }
        }


class ENAMetadataFetcher:
    """
    Fetches comprehensive metadata from ENA REST API.

    Provides efficient batch operations with caching and rate limiting.
    Retrieves sample, run, experiment, and study metadata from ENA.
    """

    ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"

    # Comprehensive field lists for different metadata types
    SAMPLE_FIELDS = (
        "accession,secondary_sample_accession,collection_date,country,"
        "environment_biome,environment_feature,environment_material,"
        "isolation_source,lat,lon,host,host_body_site,host_genotype,"
        "host_phenotype,host_sex,host_status,host_tax_id,identified_by,"
        "isolate,scientific_name,specimen_voucher,strain,sub_species,"
        "tax_id,tissue_lib,tissue_type,sample_title,description"
    )

    RUN_FIELDS = (
        "accession,run_accession,experiment_accession,sample_accession,"
        "library_strategy,library_source,library_selection,"
        "instrument_model,instrument_platform,read_count,"
        "base_count,first_public,last_updated"
    )

    EXPERIMENT_FIELDS = (
        "accession,experiment_accession,study_accession,"
        "sample_accession,library_name,library_strategy,"
        "library_source,library_selection,instrument_model,"
        "instrument_platform,title,design_description"
    )

    STUDY_FIELDS = (
        "accession,study_accession,title,abstract,study_description,"
        "submission_date,publication_date,last_updated"
    )

    # FIX #4: Coordinate bounds for validation
    LAT_BOUNDS = (-90, 90)
    LON_BOUNDS = (-180, 180)

    def __init__(
        self,
        email: str,
        cache_manager: Optional[SQLiteCacheManager] = None,
        is_authenticated: bool = False,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize ENA metadata fetcher.

        Args:
            email: Email for ENA identification
            cache_manager: Optional SQLite cache manager
            is_authenticated: Whether authenticated with ENA (affects rate limits)
            session: Optional aiohttp session (created if not provided)
        """
        self.email = email
        self.cache_manager = cache_manager
        self.session = session
        self.owns_session = session is None  # Track if we created the session

        # Set up rate limiter based on authentication
        tier = RateLimitTier.AUTHENTICATED if is_authenticated else RateLimitTier.UNAUTHENTICATED
        rate_config = RateLimitConfig(min_rate=tier.value[0], max_rate=tier.value[1])
        self.rate_limiter = TokenBucketRateLimiter(rate_config, endpoint_name="ENA")

        logger.info(f"Initialized ENAMetadataFetcher (authenticated={is_authenticated})")

    @staticmethod
    def _validate_coordinates(lat: Any, lon: Any) -> Optional[Tuple[float, float]]:
        """
        FIX #4: Validate coordinate bounds.

        Args:
            lat: Latitude value
            lon: Longitude value

        Returns:
            Tuple of (lat, lon) if valid, None otherwise
        """
        try:
            lat_float = float(lat)
            lon_float = float(lon)

            if ENAMetadataFetcher.LAT_BOUNDS[0] <= lat_float <= ENAMetadataFetcher.LAT_BOUNDS[1]:
                if ENAMetadataFetcher.LON_BOUNDS[0] <= lon_float <= ENAMetadataFetcher.LON_BOUNDS[1]:
                    return (lat_float, lon_float)

            logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}. Out of bounds.")
            return None
        except (ValueError, TypeError):
            return None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure session exists, create if needed."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": f"ENAMetadataFetcher ({self.email})"}
            )
        return self.session

    async def close(self) -> None:
        """Close session if we own it."""
        if self.owns_session and self.session and not self.session.closed:
            await self.session.close()

    async def fetch_sample_metadata(
        self,
        accession: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a single sample accession.

        Args:
            accession: Sample accession (e.g., ERS123456)
            use_cache: Whether to use cached results

        Returns:
            Dictionary with sample metadata or None if not found
        """
        # Check cache
        if use_cache and self.cache_manager:
            cache_key = f"ena_sample:{accession}"
            cached = await self.cache_manager.get(cache_key)
            if cached is not None:
                return cached

        # Fetch from API
        query = f'accession="{accession}"'
        params = {
            "result": "sample",
            "query": query,
            "fields": self.SAMPLE_FIELDS,
            "format": "json",
            "limit": 1,
        }

        data = await self._query_api(params)

        if data and len(data) > 0:
            result = data[0]

            # FIX #4: Validate coordinates if present
            if "lat" in result and "lon" in result:
                coords = self._validate_coordinates(result.get("lat"), result.get("lon"))
                if coords is None:
                    # Remove invalid coordinates
                    result.pop("lat", None)
                    result.pop("lon", None)
                    logger.warning(f"Invalid coordinates removed from {accession}")

            # Cache result
            if use_cache and self.cache_manager:
                cache_key = f"ena_sample:{accession}"
                await self.cache_manager.set(cache_key, result)

            return result

        return None

    async def fetch_run_metadata_batch(
        self,
        accessions: List[str],
        batch_size: int = 50,
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch run metadata for multiple accessions (sample or run IDs).

        Batches requests to minimize API calls.

        Args:
            accessions: List of sample or run accessions
            batch_size: Number of accessions per API request
            use_cache: Whether to use cached results

        Returns:
            Dictionary mapping accession -> run metadata
        """
        if not accessions:
            return {}

        results = {}
        # FIX #2: Use dict.fromkeys() instead of set() to preserve order
        accessions = list(dict.fromkeys(accessions))

        # Check cache first
        to_fetch = accessions
        if use_cache and self.cache_manager:
            cache_keys = [f"ena_run:{acc}" for acc in accessions]
            cached = await self.cache_manager.get_bulk(cache_keys)

            for acc in accessions:
                cache_key = f"ena_run:{acc}"
                if cache_key in cached:
                    results[acc] = cached[cache_key]

            to_fetch = [acc for acc in accessions if acc not in results]

        # Fetch non-cached accessions
        if to_fetch:
            for i in range(0, len(to_fetch), batch_size):
                batch = to_fetch[i:i + batch_size]
                batch_results = await self._fetch_run_batch(batch, use_cache)
                results.update(batch_results)

        return results

    async def _fetch_run_batch(
        self,
        batch: List[str],
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch a single batch of run metadata."""
        # Build OR query for all accessions
        query_parts = [f'(sample_accession="{acc}" OR run_accession="{acc}")' for acc in batch]
        query = " OR ".join(query_parts)

        params = {
            "result": "read_run",
            "query": query,
            "fields": self.RUN_FIELDS,
            "format": "json",
            "limit": 0,
        }

        data = await self._query_api(params)

        results = {}
        if data:
            for item in data:
                for key in ["accession", "run_accession", "sample_accession"]:
                    if key in item:
                        acc_key = item[key]
                        results[acc_key] = item

                        # Cache individual results
                        if use_cache and self.cache_manager:
                            cache_key = f"ena_run:{acc_key}"
                            await self.cache_manager.set(cache_key, item)

        return results

    async def fetch_experiment_metadata_batch(
        self,
        accessions: List[str],
        batch_size: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch experiment metadata for multiple study accessions.

        Args:
            accessions: List of study accessions
            batch_size: Number of accessions per API request

        Returns:
            Dictionary mapping study_accession -> list of experiments
        """
        if not accessions:
            return {}

        results = {}
        # FIX #2: Use dict.fromkeys() instead of set() to preserve order
        accessions = list(dict.fromkeys(accessions))

        for i in range(0, len(accessions), batch_size):
            batch = accessions[i:i + batch_size]
            query_parts = [f'study_accession="{acc}"' for acc in batch]
            query = " OR ".join(query_parts)

            params = {
                "result": "experiment",
                "query": query,
                "fields": self.EXPERIMENT_FIELDS,
                "format": "json",
                "limit": 0,
            }

            data = await self._query_api(params)

            if data:
                for item in data:
                    study_acc = item.get("study_accession", "unknown")
                    if study_acc not in results:
                        results[study_acc] = []
                    results[study_acc].append(item)

        return results

    async def fetch_study_metadata(
        self,
        study_accession: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch study metadata.

        Args:
            study_accession: Study accession (e.g., ERP123456)

        Returns:
            Dictionary with study metadata or None
        """
        query = f'accession="{study_accession}"'
        params = {
            "result": "study",
            "query": query,
            "fields": self.STUDY_FIELDS,
            "format": "json",
            "limit": 1,
        }

        data = await self._query_api(params)

        if data and len(data) > 0:
            return data[0]

        return None

    async def _query_api(
        self,
        params: Dict[str, Any],
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a query against ENA API with rate limiting and retries.

        Args:
            params: Query parameters
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts

        Returns:
            List of results or None if failed
        """
        session = await self._ensure_session()

        # Apply rate limiting
        if not await self.rate_limiter.acquire(tokens=1, timeout=timeout):
            logger.error("Rate limiter timeout for ENA API")
            return None

        for attempt in range(max_retries):
            try:
                async with session.get(
                    self.ENA_API_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if response.status == 204:
                        return None

                    if response.status == 429:  # Rate limited
                        wait_time = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"ENA API rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status == 503:  # Service unavailable
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"ENA API unavailable, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                logger.warning(f"ENA API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(f"ENA API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None


class SRAMetadataFetcher:
    """
    Fetches additional metadata from NCBI SRA when ENA data is incomplete.

    Uses NCBI E-utilities API (esearch, esummary) to supplement ENA data.
    Provides fallback metadata retrieval and rate-limited access.
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        cache_manager: Optional[SQLiteCacheManager] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize SRA metadata fetcher.

        Args:
            email: Email for NCBI identification (required)
            api_key: NCBI API key for higher rate limits (optional)
            cache_manager: Optional SQLite cache manager
            session: Optional aiohttp session
        """
        self.email = email
        self.api_key = api_key
        self.cache_manager = cache_manager
        self.session = session
        self.owns_session = session is None

        # Set up rate limiter (NCBI: 3 req/sec without API key, 10 req/sec with)
        rate = 10.0 if api_key else 3.0
        rate_config = RateLimitConfig(min_rate=rate * 0.6, max_rate=rate)
        self.rate_limiter = TokenBucketRateLimiter(rate_config, endpoint_name="NCBI-SRA")

        logger.info(f"Initialized SRAMetadataFetcher (email={email}, authenticated={bool(api_key)})")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": f"SRAMetadataFetcher ({self.email})"}
            )
        return self.session

    async def close(self) -> None:
        """Close session if we own it."""
        if self.owns_session and self.session and not self.session.closed:
            await self.session.close()

    async def fetch_sra_details(
        self,
        accession: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch SRA metadata for a sample accession.

        Args:
            accession: SRA accession (SAMN, SRS, etc.)
            use_cache: Whether to use cached results

        Returns:
            Dictionary with SRA metadata or None
        """
        # Check cache
        if use_cache and self.cache_manager:
            cache_key = f"sra_details:{accession}"
            cached = await self.cache_manager.get(cache_key)
            if cached is not None:
                return cached

        # Search for the accession
        search_result = await self._search_accession(accession)
        if not search_result:
            return None

        # Get summary
        uid = search_result
        summary = await self._get_summary(uid)

        # Cache and return
        if summary and use_cache and self.cache_manager:
            cache_key = f"sra_details:{accession}"
            await self.cache_manager.set(cache_key, summary)

        return summary

    async def _search_accession(self, accession: str) -> Optional[str]:
        """
        Search NCBI SRA for an accession and return UID.

        Args:
            accession: Accession to search for

        Returns:
            NCBI UID or None if not found
        """
        session = await self._ensure_session()

        if not await self.rate_limiter.acquire(tokens=1, timeout=10.0):
            return None

        params = {
            "db": "sra",
            "term": accession,
            "retmax": 1,
            "rettype": "json",
            "tool": "SRAMetadataFetcher",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        # FIX #6: Add retry logic with exponential backoff for SRA (was missing)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(self.ESEARCH_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:  # Rate limited
                        wait_time = int(resp.headers.get("Retry-After", 5))
                        logger.warning(f"SRA API rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status == 503:  # Service unavailable
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"SRA API unavailable, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

                    if data.get("esearchresult", {}).get("idlist"):
                        return data["esearchresult"]["idlist"][0]

            except asyncio.TimeoutError:
                logger.warning(f"SRA search timeout (attempt {attempt + 1}/{max_retries}) for {accession}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(f"SRA search error (attempt {attempt + 1}/{max_retries}) for {accession}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None

    async def _get_summary(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get summary for a NCBI UID."""
        session = await self._ensure_session()

        if not await self.rate_limiter.acquire(tokens=1, timeout=10.0):
            return None

        params = {
            "db": "sra",
            "id": uid,
            "rettype": "json",
            "tool": "SRAMetadataFetcher",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        # FIX #6: Add retry logic with exponential backoff for SRA (was missing)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(self.ESUMMARY_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:  # Rate limited
                        wait_time = int(resp.headers.get("Retry-After", 5))
                        logger.warning(f"SRA API rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status == 503:  # Service unavailable
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"SRA API unavailable, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

                    result = data.get("result", {})
                    if uid in result:
                        return result[uid]

            except asyncio.TimeoutError:
                logger.warning(f"SRA summary timeout (attempt {attempt + 1}/{max_retries}) for {uid}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(f"SRA summary error (attempt {attempt + 1}/{max_retries}) for {uid}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None


class MetadataMerger:
    """
    Intelligently merges ENA and SRA metadata results.

    Rules:
    - ENA data takes precedence (it's more reliable)
    - SRA fills gaps where ENA data is missing
    - No data loss - preserves all available fields
    """

    # Fields that should be prioritized from ENA
    ENA_PRIORITY_FIELDS = {
        "collection_date",
        "lat", "lon",
        "country",
        "host",
        "isolation_source",
        "environment_biome",
        "scientific_name",
        "strain",
    }

    @staticmethod
    def merge_metadata(
        ena_data: Optional[Dict[str, Any]],
        sra_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge ENA and SRA metadata with ENA taking precedence.

        Args:
            ena_data: Metadata from ENA (preferred)
            sra_data: Metadata from SRA (fills gaps)

        Returns:
            Merged metadata dictionary
        """
        result = {}

        # Start with ENA data
        if ena_data:
            result.update(ena_data)

        # Fill gaps with SRA data
        if sra_data:
            for key, value in sra_data.items():
                # FIX #5: Check if merged value is empty/None before preventing SRA from filling
                merged_val = result.get(key)
                if (merged_val is None or
                    (isinstance(merged_val, str) and not merged_val.strip())) and value is not None and value != "":
                    result[key] = value

        # Add source tracking
        sources = []
        if ena_data:
            sources.append("ENA")
        if sra_data:
            sources.append("SRA")

        result["_metadata_sources"] = sources

        logger.debug(f"Merged metadata from sources: {sources}")

        return result

    @staticmethod
    def merge_batch(
        ena_batch: Dict[str, Dict[str, Any]],
        sra_batch: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge batch results from ENA and SRA.

        Args:
            ena_batch: Dictionary of accession -> ENA metadata
            sra_batch: Dictionary of accession -> SRA metadata

        Returns:
            Dictionary of accession -> merged metadata
        """
        result = {}

        # Get all unique accessions
        all_accessions = set(ena_batch.keys()) | set(sra_batch.keys())

        for acc in all_accessions:
            ena_data = ena_batch.get(acc)
            sra_data = sra_batch.get(acc)
            result[acc] = MetadataMerger.merge_metadata(ena_data, sra_data)

        return result
