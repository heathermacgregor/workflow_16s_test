# ===================================== IMPORTS ====================================== #

# Standard Imports
import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third Party Imports
import aiohttp
import numpy as np
import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class MetadataEnricher:
    """
    Handles all external API-based metadata enrichment.
    
    This class is responsible for:
    1.  Reverse geocoding coordinates (Nominatim).
    2.  Fetching ENVO labels from codes (EBI OLS).
    3.  Finding publication DOIs from accessions (NCBI E-utils).
    
    It manages API sessions, rate limiting, retries, and caching.
    """
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        ncbi_api_key: Optional[str] = None,
        cache_path: Optional[Path] = None
    ):
        self.session = session
        self.ncbi_api_key = ncbi_api_key
        self.cache_path = cache_path
        
        # State for NCBI rate limiting
        self.ncbi_semaphore = asyncio.Semaphore(10)
        self.ncbi_pacing_lock = asyncio.Lock()
        self.last_ncbi_request_time = 0
        
        # Initialize caches
        if self.cache_path:
            self._initialize_caches()
        
        # Statistics tracking
        self.stats = {
            'geocoding': {'total': 0, 'cached': 0, 'failed': 0},
            'envo': {'total': 0, 'cached': 0, 'failed': 0, 'batch_requests': 0},
            'publications': {'total': 0, 'cached': 0, 'failed': 0}
        }
    
    def _initialize_caches(self):
        """Initialize SQLite caches for geocoding and ENVO codes."""
        if not self.cache_path:
            return
        
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.cache_path) as conn:
            # Geocoding cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS geocoding_cache (
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    location TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (lat, lon)
                )
            """)
            
            # ENVO code cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS envo_cache (
                    code TEXT PRIMARY KEY,
                    label TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_geocoding_timestamp ON geocoding_cache(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_envo_timestamp ON envo_cache(timestamp)")
    
    def _get_cached_location(self, lat: float, lon: float) -> Optional[str]:
        """Retrieve cached geocoding result."""
        if not self.cache_path:
            return None
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    "SELECT location FROM geocoding_cache WHERE lat = ? AND lon = ?",
                    (lat, lon)
                )
                result = cursor.fetchone()
                if result:
                    self.stats['geocoding']['cached'] += 1
                    return result[0]
        except sqlite3.Error as e:
            logger.warning(f"Geocoding cache lookup error: {e}")
        
        return None
    
    def _cache_location(self, lat: float, lon: float, location: str):
        """Cache geocoding result."""
        if not self.cache_path:
            return
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO geocoding_cache (lat, lon, location) VALUES (?, ?, ?)",
                    (lat, lon, location)
                )
        except sqlite3.Error as e:
            logger.warning(f"Geocoding cache write error: {e}")
    
    def _get_cached_envo_codes(self, codes: Set[str]) -> Dict[str, str]:
        """Retrieve cached ENVO code labels in batch."""
        if not self.cache_path or not codes:
            return {}
        
        try:
            placeholders = ','.join('?' * len(codes))
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    f"SELECT code, label FROM envo_cache WHERE code IN ({placeholders})",
                    tuple(codes)
                )
                cached = {row[0]: row[1] for row in cursor.fetchall()}
                self.stats['envo']['cached'] += len(cached)
                return cached
        except sqlite3.Error as e:
            logger.warning(f"ENVO cache lookup error: {e}")
            return {}
    
    def _cache_envo_codes(self, code_label_map: Dict[str, str]):
        """Cache ENVO code labels in batch."""
        if not self.cache_path or not code_label_map:
            return
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO envo_cache (code, label) VALUES (?, ?)",
                    code_label_map.items()
                )
        except sqlite3.Error as e:
            logger.warning(f"ENVO cache write error: {e}")

    # ========================== GEOLOCATION ENRICHMENT ============================ #

    async def enrich_location_from_coords(self, df: pd.DataFrame) -> None:
        """Fills 'location' column by reverse-geocoding 'lat'/'lon' with caching."""
        if 'location' not in df.columns: df['location'] = np.nan
        if 'lat' not in df.columns or 'lon' not in df.columns: return

        rows_to_check = df[df['location'].isnull() & df['lat'].notna()]
        if rows_to_check.empty: return

        logger.info(f"Found {len(rows_to_check)} rows to enrich with geocoding...")
        semaphore = asyncio.Semaphore(1) # Nominatim policy: 1 request/sec
        
        geolocator = Nominatim(
            user_agent="metadata_analysis_script_v3", adapter_factory=self.session.get
        )

        tasks = [
            self._fetch_single_location(geolocator, semaphore, index, row['lat'], row['lon'])
            for index, row in rows_to_check.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        for index, location_str in results:
            if location_str: df.loc[index, 'location'] = location_str

    async def _fetch_single_location(
        self, geolocator, semaphore, index, lat, lon
    ) -> Tuple[int, Optional[str]]:
        self.stats['geocoding']['total'] += 1
        
        # Check cache first
        cached = self._get_cached_location(lat, lon)
        if cached:
            return index, cached
        
        try:
            async with semaphore:
                location = await geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
                if location and hasattr(location, 'raw') and 'address' in location.raw:
                    addr = location.raw['address']
                    city = addr.get('city', addr.get('town', addr.get('village', '')))
                    country = addr.get('country', '')
                    location_str = ", ".join(filter(None, [city, country]))
                    
                    # Cache the result
                    self._cache_location(lat, lon, location_str)
                    return index, location_str
        except (GeocoderTimedOut, GeocoderServiceError, asyncio.TimeoutError) as e:
            logger.warning(f"Geocoding service error for index {index}: {e}")
            self.stats['geocoding']['failed'] += 1
        
        return index, None

    # ============================ ENVO CODE ENRICHMENT ============================ #

    async def convert_envo_codes(self, df: pd.DataFrame) -> None:
        """Converts ENVO codes in ontology columns to human-readable labels using batch API and caching."""
        envo_cols = ['env_material', 'env_feature', 'env_biome']
        all_codes_to_lookup = set()
        envo_pattern = re.compile(r'(ENVO[:_]\d{7})', re.IGNORECASE)
        
        existing_cols = [c for c in envo_cols if c in df.columns]
        if not existing_cols:
            return

        all_unique_values = pd.concat(
            [df[c].dropna() for c in existing_cols]
        ).astype(str).unique()

        for val in all_unique_values:
            matches = envo_pattern.findall(val)
            for match in matches:
                all_codes_to_lookup.add(match.upper().replace('_', ':'))
        
        if not all_codes_to_lookup:
            return
        
        logger.info(f"Found {len(all_codes_to_lookup)} unique ENVO codes to look up.")
        self.stats['envo']['total'] = len(all_codes_to_lookup)
        
        # Check cache first
        code_to_label_map = self._get_cached_envo_codes(all_codes_to_lookup)
        codes_to_fetch = all_codes_to_lookup - set(code_to_label_map.keys())
        
        if codes_to_fetch:
            logger.info(f"Fetching {len(codes_to_fetch)} ENVO codes from API (cache hit: {len(code_to_label_map)}).")
            
            # Fetch in batches using EBI OLS batch endpoint
            batch_size = 50  # OLS can handle batch requests
            code_list = list(codes_to_fetch)
            
            for i in range(0, len(code_list), batch_size):
                batch = code_list[i:i + batch_size]
                self.stats['envo']['batch_requests'] += 1
                batch_results = await self._fetch_envo_labels_batch(batch)
                code_to_label_map.update(batch_results)
            
            # Cache new results
            new_labels = {code: label for code, label in code_to_label_map.items() if code in codes_to_fetch}
            self._cache_envo_codes(new_labels)
        else:
            logger.info(f"All {len(all_codes_to_lookup)} ENVO codes found in cache.")

        for col in existing_cols:
            df[col] = df[col].astype(str).replace(code_to_label_map, regex=True)
        
        logger.info(f"ENVO enrichment stats: {self.stats['envo']}")

    async def _fetch_envo_label(self, code) -> Tuple[str, Optional[str]]:
        """Fetch single ENVO label (legacy method, prefer batch)."""
        url = "https://www.ebi.ac.uk/ols/api/ontologies/envo/terms"
        iri = f"http://purl.obolibrary.org/obo/{code.replace(':', '_')}"
        params = {'iri': iri}
        try:
            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    terms = data.get('_embedded', {}).get('terms', [])
                    if terms and 'label' in terms[0]:
                        return code, terms[0]['label']
        except Exception as e:
            logger.warning(f"OLS API request failed for {code}: {e}")
            self.stats['envo']['failed'] += 1
        return code, None
    
    async def _fetch_envo_labels_batch(self, codes: List[str]) -> Dict[str, str]:
        """Fetch multiple ENVO labels in a batch to reduce API calls."""
        if not codes:
            return {}
        
        results = {}
        
        # EBI OLS supports searching by multiple IRIs
        # We'll fetch them concurrently but limit concurrency
        semaphore = asyncio.Semaphore(10)
        
        async def fetch_with_semaphore(code):
            async with semaphore:
                return await self._fetch_envo_label(code)
        
        tasks = [fetch_with_semaphore(code) for code in codes]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, tuple) and result[1]:
                results[result[0]] = result[1]
            elif isinstance(result, Exception):
                logger.warning(f"Exception in batch ENVO fetch: {result}")
        
        return results

    # ========================== PUBLICATION ENRICHMENT ========================== #

    async def find_publications(self, df: pd.DataFrame) -> None:
        """Finds publication DOIs from accession numbers."""
        if 'publication_doi' not in df.columns: df['publication_doi'] = np.nan
        
        acc_pattern = r'^(run|sample|exp|study|proj|sra|ena|ddbj|bio)_?(acc|alias)$|^acc$'
        all_acc_cols = [c for c in df.columns if re.search(acc_pattern, c, re.I)]
        if not all_acc_cols: return
        
        priority = ['project', 'study', 'biosample', 'sra', 'ena',
                    'sample', 'experiment', 'run']
        sorted_acc_cols = sorted(all_acc_cols,
            key=lambda c: next((i for i, p in enumerate(priority)
                                if p in c.lower()), len(priority)))
        
        df['search_accession'] = df[sorted_acc_cols].bfill(axis=1).iloc[:, 0]
        rows_to_search = df[df['publication_doi'].isnull() & df['search_accession'].notna()]
        if rows_to_search.empty: return

        unique_accessions = rows_to_search['search_accession'].unique()
        logger.info(f"Found {len(unique_accessions)} unique accessions to check for publications.")
        
        tasks = [
            self._fetch_single_doi(acc) for acc in unique_accessions
        ]
        results = await asyncio.gather(*tasks)

        accession_to_doi = {acc: doi for acc, doi in results if doi}
        
        if accession_to_doi:
            doi_map = df['search_accession'].map(accession_to_doi)
            df['publication_doi'].fillna(doi_map, inplace=True)
        
        df.drop(columns=['search_accession'], inplace=True)

    async def _fetch_single_doi(
        self, accession
    ) -> Tuple[str, Optional[str]]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        api_key_str = f"&api_key={self.ncbi_api_key}" if self.ncbi_api_key else ""
        
        async def get_xml(url):
            max_retries = 5
            for attempt in range(max_retries):
                async with self.ncbi_semaphore:
                    async with self.ncbi_pacing_lock:
                        now = time.monotonic()
                        elapsed = now - self.last_ncbi_request_time
                        wait_time = (0.11 if self.ncbi_api_key else 0.34) - elapsed
                        if wait_time > 0: await asyncio.sleep(wait_time)
                        self.last_ncbi_request_time = time.monotonic()

                    try:
                        async with self.session.get(url, timeout=60) as resp:
                            if resp.status == 429:
                                retry_after = int(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                                logger.warning(f"Rate limited for {accession}. Retrying in {retry_after} seconds... (Attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(retry_after)
                                continue
                            resp.raise_for_status()
                            return ET.fromstring(await resp.text())
                    except aiohttp.ClientError as e:
                        logger.warning(
                            f"NCBI request for {accession} failed: {e}", exc_info=True
                        )
                        await asyncio.sleep(1 * (attempt + 1))
            
            logger.error(f"All {max_retries} retries failed for NCBI request: {accession}")
            return None

        pmids = []
        for db in ['bioproject', 'sra', 'biosample']:
            uid_root = await get_xml(f"{base_url}esearch.fcgi?db={db}&term={accession}&retmode=xml{api_key_str}")
            if uid_root is not None and (id_elem := uid_root.find('.//Id')) is not None and id_elem.text:
                uid = id_elem.text
                link_root = await get_xml(f"{base_url}elink.fcgi?dbfrom={db}&db=pubmed&id={uid}&retmode=xml{api_key_str}")
                if link_root is not None:
                    pmids = [id_elem.text for id_elem in link_root.findall(".//LinkSetDb[DbTo='pubmed']//Id") if id_elem.text]
                if pmids: break
        
        if not pmids:
            pm_root = await get_xml(f"{base_url}esearch.fcgi?db=pubmed&term={accession}[accn]&retmode=xml{api_key_str}")
            if pm_root is not None:
                pmids = [id_elem.text for id_elem in pm_root.findall('.//Id') if id_elem.text]

        if pmids:
            summary_root = await get_xml(f"{base_url}esummary.fcgi?db=pubmed&id={pmids[0]}&retmode=xml{api_key_str}")
            if summary_root is not None and (doi_elem := summary_root.find(".//Item[@Name='DOI']")) is not None and doi_elem.text:
                logger.info(f"SUCCESS: Found DOI {doi_elem.text} for {accession}")
                return accession, doi_elem.text
        
        return accession, None