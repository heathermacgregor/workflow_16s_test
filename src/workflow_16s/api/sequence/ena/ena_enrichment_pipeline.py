"""
ENA Enrichment Pipeline - Orchestrates Location/Date Extraction and Metadata Enrichment.

This module provides a unified orchestration layer that:
1. Parses sample IDs (ENA/SRA accessions or raw IDs)
2. Resolves projects using metadata
3. Fetches comprehensive metadata from ENA/SRA
4. Extracts location data (lat/lon, coordinates)
5. Extracts and standardizes collection dates (ISO 8601)
6. Returns enriched DataFrame with geographic and temporal data

Architecture:
- Modular composition (integrates with existing modules)
- Async-first design with sync wrappers
- Graceful degradation (partial failures are acceptable)
- Configuration-driven behavior
- Observable (logging, progress bars, statistics)
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import pandas as pd

from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger, with_logger
from workflow_16s.utils.progress import get_progress_bar

from .cache import SQLiteCacheManager
from .sample_parser import SampleParser, ParsedSample, ProjectInfo
from .metadata_fetcher import ENAMetadataFetcher, SRAMetadataFetcher, MetadataMerger
from .metadata_enrichment import (
    LocationParser,
    DateParser,
    enrich_metadata_with_location,
    enrich_metadata_with_dates,
)

logger = get_logger("workflow_16s")


# ===================== DATA CLASSES ===================== #


@dataclass
class EnrichmentStats:
    """Statistics from enrichment pipeline execution."""

    samples_processed: int = 0
    samples_with_location: int = 0
    samples_with_date: int = 0
    samples_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Error tracking
    parse_errors: int = 0
    fetch_errors: int = 0
    location_extraction_errors: int = 0
    date_parsing_errors: int = 0

    # Detailed error messages
    failed_samples: Dict[str, str] = field(default_factory=dict)

    # FIX #11: Track failed accessions for better error reporting
    failed_accessions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "samples_processed": self.samples_processed,
            "samples_with_location": self.samples_with_location,
            "samples_with_date": self.samples_with_date,
            "samples_failed": self.samples_failed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "parse_errors": self.parse_errors,
            "fetch_errors": self.fetch_errors,
            "location_extraction_errors": self.location_extraction_errors,
            "date_parsing_errors": self.date_parsing_errors,
            "failed_samples": self.failed_samples,
            "failed_accessions": self.failed_accessions,
        }

    def log_summary(self) -> None:
        """Log statistics summary."""
        logger.info(f"ENA Enrichment Pipeline Statistics:")
        logger.info(f"  Samples processed: {self.samples_processed}")
        logger.info(f"  Samples with location: {self.samples_with_location} "
                   f"({100*self.samples_with_location/max(1, self.samples_processed):.1f}%)")
        logger.info(f"  Samples with date: {self.samples_with_date} "
                   f"({100*self.samples_with_date/max(1, self.samples_processed):.1f}%)")
        logger.info(f"  Samples failed: {self.samples_failed}")
        logger.info(f"  Cache hits: {self.cache_hits}")
        logger.info(f"  Cache misses: {self.cache_misses}")

        if self.parse_errors > 0:
            logger.warning(f"  Parse errors: {self.parse_errors}")
        if self.fetch_errors > 0:
            logger.warning(f"  Fetch errors: {self.fetch_errors}")
        if self.location_extraction_errors > 0:
            logger.warning(f"  Location extraction errors: {self.location_extraction_errors}")
        if self.date_parsing_errors > 0:
            logger.warning(f"  Date parsing errors: {self.date_parsing_errors}")


# ===================== ENA ENRICHMENT PIPELINE ===================== #


@with_logger
class ENAEnrichmentPipeline:
    """
    Orchestrates the complete ENA enrichment flow.

    Input: Sample IDs (mixed ENA/SRA accessions or raw IDs)
    Output: Enriched DataFrame with location (lat/lon) and collection_date
    """

    def __init__(
        self,
        config: AppConfig,
        cache_manager: Optional[SQLiteCacheManager] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the ENA enrichment pipeline.

        Args:
            config: Application configuration with credentials
            cache_manager: Optional cache manager instance
            cache_dir: Optional cache directory (used if cache_manager not provided)
        """
        self.config = config
        self.cache_manager = cache_manager

        # Initialize cache if not provided
        if self.cache_manager is None and cache_dir:
            self.cache_manager = SQLiteCacheManager(cache_dir)
        elif self.cache_manager is None:
            # Use default cache location
            default_cache_dir = Path.home() / ".cache" / "workflow_16s" / "ena"
            default_cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_manager = SQLiteCacheManager(default_cache_dir)

        # Extract credentials
        self.email = config.credentials.ena_email or config.credentials.email
        self.ncbi_api_key = config.credentials.ncbi_api_key

        if not self.email:
            logger.warning("ENA email not configured; using default. Some API calls may be rate-limited.")

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.owns_session = False

        # Statistics
        self.stats = EnrichmentStats()

    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=10,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "workflow_16s/ENAEnrichmentPipeline/1.0"},
        )
        self.owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - always ensures cleanup even on exception."""
        # FIX #10: Ensure __aexit__ always runs cleanup, even if exception occurred
        if self.owns_session and self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing session in __aexit__: {e}")

        # Always return False to propagate any exceptions
        return False

    async def enrich_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a DataFrame with location and date information.

        Expects DataFrame with sample IDs in a recognized column.

        Args:
            df: Input DataFrame with samples

        Returns:
            Enriched DataFrame with location and date columns
        """
        # Ensure we have a session
        if self.session is None:
            await self.__aenter__()

        self.stats = EnrichmentStats()

        # Find sample ID column
        sample_id_col = self._find_sample_id_column(df)
        if not sample_id_col:
            logger.error("No sample ID column found in DataFrame")
            return df.copy()

        sample_ids = df[sample_id_col].dropna().unique().tolist()
        logger.info(f"Enriching {len(sample_ids)} unique samples...")

        # Main enrichment pipeline
        enriched_data = await self._enrich_samples_batch(sample_ids)

        # Merge enriched data back to DataFrame
        result_df = df.copy()

        # FIX #8: Use vectorized merge instead of O(n²) row-by-row iteration
        # Convert enriched_data dict to DataFrame for efficient merging
        if enriched_data:
            enriched_rows = []
            for sample_id, data in enriched_data.items():
                row = {sample_id_col: sample_id}
                row.update(data)
                enriched_rows.append(row)

            enriched_df = pd.DataFrame(enriched_rows)
            # Merge using pandas' efficient merge operation
            result_df = result_df.merge(enriched_df, on=sample_id_col, how='left', suffixes=('', '_enriched'))

            # Clean up duplicate columns if any
            for col in result_df.columns:
                if col.endswith('_enriched'):
                    base_col = col[:-10]  # Remove '_enriched' suffix
                    if base_col in result_df.columns and base_col != sample_id_col:
                        # Use enriched value where base is NaN
                        mask = result_df[base_col].isna()
                        result_df.loc[mask, base_col] = result_df.loc[mask, col]
                        result_df = result_df.drop(col, axis=1)

        # Log statistics
        self.stats.log_summary()

        return result_df

    async def enrich_by_ids(self, sample_ids: List[str]) -> pd.DataFrame:
        """
        Enrich samples by their IDs.

        Args:
            sample_ids: List of sample IDs (ENA/SRA accessions or raw IDs)

        Returns:
            DataFrame with enriched data
        """
        # Ensure we have a session
        if self.session is None:
            await self.__aenter__()

        self.stats = EnrichmentStats()

        logger.info(f"Enriching {len(sample_ids)} samples...")

        # Enrich all samples
        enriched_data = await self._enrich_samples_batch(sample_ids)

        # Convert to DataFrame
        rows = []
        for sample_id, data in enriched_data.items():
            row = {"sample_id": sample_id}
            row.update(data)
            rows.append(row)

        result_df = pd.DataFrame(rows)

        # Log statistics
        self.stats.log_summary()

        return result_df

    async def _enrich_samples_batch(self, sample_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Internal batch enrichment of samples.

        Args:
            sample_ids: List of sample IDs to enrich

        Returns:
            Dictionary mapping sample_id -> enriched_data
        """
        enriched_data = {}

        with get_progress_bar() as progress:
            task = progress.add_task("Enriching samples", total=len(sample_ids))

            for sample_id in sample_ids:
                try:
                    enriched = await self._enrich_single_sample(sample_id)
                    enriched_data[sample_id] = enriched
                    self.stats.samples_processed += 1

                except Exception as e:
                    logger.error(f"Error enriching {sample_id}: {e}")
                    self.stats.samples_failed += 1
                    self.stats.failed_samples[sample_id] = str(e)
                    # FIX #11: Track failed accessions for better error reporting
                    self.stats.failed_accessions.append(sample_id)

                finally:
                    progress.update(task, advance=1)

        return enriched_data

    async def _enrich_single_sample(self, sample_id: str) -> Dict[str, Any]:
        """
        Enrich a single sample.

        Steps:
        1. Parse sample ID
        2. Fetch metadata from ENA/SRA
        3. Extract location
        4. Extract and standardize date

        Args:
            sample_id: Sample ID to enrich

        Returns:
            Dictionary with enriched data
        """
        enriched = {}

        try:
            # Step 1: Parse sample ID
            parsed = await self._parse_sample_id(sample_id)
            if not parsed.is_valid:
                logger.debug(f"Could not parse sample ID: {sample_id}")
                self.stats.parse_errors += 1
                return enriched

            # Step 2: Fetch metadata from ENA/SRA
            metadata = await self._fetch_sample_metadata(parsed)
            if not metadata:
                logger.debug(f"No metadata found for: {sample_id}")
                self.stats.fetch_errors += 1
                return enriched

            # Step 3: Extract location
            try:
                location = self._extract_location(metadata)
                if location:
                    # FIX #7: Use 'lat'/'lon' column names for MetadataManager integration
                    enriched["lat"], enriched["lon"] = location
                    enriched["location_confidence"] = "extracted"
                    self.stats.samples_with_location += 1
            except Exception as e:
                logger.warning(f"Error extracting location for {sample_id}: {e}")
                self.stats.location_extraction_errors += 1

            # Step 4: Extract and standardize date
            try:
                date_info = self._extract_date(metadata)
                if date_info:
                    enriched["collection_date"], enriched["collection_date_precision"] = date_info
                    self.stats.samples_with_date += 1
            except Exception as e:
                logger.warning(f"Error extracting date for {sample_id}: {e}")
                self.stats.date_parsing_errors += 1

            # Add any additional metadata that might be useful
            enriched["sample_title"] = metadata.get("sample_title")
            enriched["scientific_name"] = metadata.get("scientific_name")
            enriched["country"] = metadata.get("country")

        except Exception as e:
            logger.error(f"Unexpected error enriching {sample_id}: {e}")
            self.stats.failed_samples[sample_id] = str(e)

        return enriched

    async def _parse_sample_id(self, sample_id: str) -> ParsedSample:
        """Parse a sample ID into normalized accessions."""
        async with SampleParser(cache_manager=self.cache_manager) as parser:
            parsed_dict = await parser.parse_sample_ids_async([sample_id])
            return parsed_dict.get(sample_id)

    async def _fetch_sample_metadata(self, parsed: ParsedSample) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive metadata from ENA and SRA.

        Uses ENA as primary source, SRA as fallback.

        Args:
            parsed: ParsedSample with accessions

        Returns:
            Merged metadata dictionary or None
        """
        ena_data = None
        sra_data = None

        # Try ENA first
        if parsed.primary_accession or parsed.secondary_accession:
            accession = parsed.primary_accession or parsed.secondary_accession
            try:
                ena_fetcher = ENAMetadataFetcher(
                    email=self.email,
                    cache_manager=self.cache_manager,
                    session=self.session,
                )
                ena_data = await ena_fetcher.fetch_sample_metadata(accession)
                if ena_data:
                    # FIX #9: Increment cache_hits when data is found (was incrementing misses)
                    self.stats.cache_hits += 1
            except Exception as e:
                logger.debug(f"Error fetching ENA data for {accession}: {e}")

        # Try SRA as fallback/supplement
        if parsed.secondary_accession and not ena_data:
            try:
                sra_fetcher = SRAMetadataFetcher(
                    email=self.email,
                    api_key=self.ncbi_api_key,
                    cache_manager=self.cache_manager,
                    session=self.session,
                )
                sra_data = await sra_fetcher.fetch_sra_details(parsed.secondary_accession)
                if sra_data:
                    # FIX #9: Increment cache_hits when data is found (was incrementing misses)
                    self.stats.cache_hits += 1
            except Exception as e:
                logger.debug(f"Error fetching SRA data for {parsed.secondary_accession}: {e}")

        # Merge results
        if ena_data or sra_data:
            return MetadataMerger.merge_metadata(ena_data, sra_data)

        return None

    def _extract_location(self, metadata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from metadata.

        Tries multiple sources in order of preference.

        Args:
            metadata: Metadata dictionary

        Returns:
            Tuple of (latitude, longitude) or None
        """
        # Try explicit lat/lon fields first
        coords = LocationParser.extract_coordinates(metadata)
        if coords:
            return coords

        # Try location string parsing
        for loc_field in ["location", "environment", "isolation_source", "sample_title"]:
            if loc_field in metadata:
                coords = LocationParser.extract_from_location_string(str(metadata[loc_field]))
                if coords:
                    return coords

        return None

    def _extract_date(self, metadata: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Extract and standardize collection date from metadata.

        Returns ISO 8601 format and precision.

        Args:
            metadata: Metadata dictionary

        Returns:
            Tuple of (date_string, precision) or None
        """
        # Try collection_date field first
        if "collection_date" in metadata:
            date_str, precision = DateParser.parse_date(
                metadata["collection_date"],
                return_precision=True,
            )
            if date_str:
                return (date_str, precision)

        # Try other date-like fields
        for date_field in ["sampling_date", "sample_date", "date"]:
            if date_field in metadata:
                date_str, precision = DateParser.parse_date(
                    metadata[date_field],
                    return_precision=True,
                )
                if date_str:
                    return (date_str, precision)

        return None

    @staticmethod
    def _find_sample_id_column(df: pd.DataFrame) -> Optional[str]:
        """
        Find the sample ID column in a DataFrame.

        Tries common column names in order.

        Args:
            df: Input DataFrame

        Returns:
            Column name or None
        """
        candidates = [
            "#sampleid",
            "sample_id",
            "sample id",
            "sampleid",
            "run_accession",
            "sample_accession",
            "accession",
        ]

        for col in candidates:
            if col.lower() in df.columns.str.lower().tolist():
                # Find the actual column name (case-insensitive)
                for actual_col in df.columns:
                    if actual_col.lower() == col.lower():
                        return actual_col

        return None


# ===================== CONVENIENCE FUNCTIONS ===================== #


async def enrich_samples(
    df: pd.DataFrame,
    config: AppConfig,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Enrich samples in a DataFrame with location and date information.

    Async version.

    Args:
        df: Input DataFrame with samples
        config: Application configuration
        cache_dir: Optional cache directory

    Returns:
        Enriched DataFrame
    """
    async with ENAEnrichmentPipeline(config, cache_dir=cache_dir) as pipeline:
        return await pipeline.enrich_samples(df)


async def enrich_by_ids(
    sample_ids: List[str],
    config: AppConfig,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Enrich samples by their IDs.

    Async version.

    Args:
        sample_ids: List of sample IDs
        config: Application configuration
        cache_dir: Optional cache directory

    Returns:
        DataFrame with enriched data
    """
    async with ENAEnrichmentPipeline(config, cache_dir=cache_dir) as pipeline:
        return await pipeline.enrich_by_ids(sample_ids)


def enrich_samples_sync(
    df: pd.DataFrame,
    config: AppConfig,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Enrich samples in a DataFrame with location and date information.

    Sync wrapper around async version.

    Args:
        df: Input DataFrame with samples
        config: Application configuration
        cache_dir: Optional cache directory

    Returns:
        Enriched DataFrame
    """
    return asyncio.run(enrich_samples(df, config, cache_dir))


def enrich_by_ids_sync(
    sample_ids: List[str],
    config: AppConfig,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Enrich samples by their IDs.

    Sync wrapper around async version.

    Args:
        sample_ids: List of sample IDs
        config: Application configuration
        cache_dir: Optional cache directory

    Returns:
        DataFrame with enriched data
    """
    return asyncio.run(enrich_by_ids(sample_ids, config, cache_dir))
