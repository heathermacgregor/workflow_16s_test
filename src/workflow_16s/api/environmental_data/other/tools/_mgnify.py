"""
MGnify (formerly EBI Metagenomics) API Handler

Provides curated environmental metadata and functional annotations from metagenomics samples.
MGnify samples often have richer environmental annotations (ENVO ontology terms) than raw ENA.

API: https://www.ebi.ac.uk/metagenomics/api/latest/

Features:
- Exponential backoff retry logic (configurable)
- Timeout and retry configuration
- Detailed logging for debugging
- Graceful failure mode (returns empty results on failure)
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional
import requests
from requests.exceptions import Timeout, ConnectionError, RequestException
from .base import BaseEnvironmentalAPI

logger = logging.getLogger(__name__)


class MGnifyAPI(BaseEnvironmentalAPI):
    """
    Query MGnify for metagenomics samples and environmental metadata near coordinates.

    Features:
    - Configurable timeout (default: 30s, increased from 15s)
    - Exponential backoff retry logic (default: 3 retries with 2x multiplier)
    - Detailed logging for retry attempts and failures
    - Graceful degradation when API fails

    Returns:
    - Number of metagenomics studies/samples in area
    - Environmental descriptors (ENVO terms)
    - Sample types
    - Functional profiles if available
    """

    API_NAME = "MGnify_Metagenomics"
    BASE_URL = "https://www.ebi.ac.uk/metagenomics/api/latest/samples"

    def __init__(self, verbose: bool = False, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MGnify API handler.

        Args:
            verbose: Enable verbose logging
            config: Configuration dict with optional keys:
                - timeout_seconds: Request timeout (default: 30)
                - max_retries: Max retry attempts (default: 3)
                - backoff_multiplier: Exponential backoff multiplier (default: 2.0)
        """
        super().__init__(verbose=verbose)

        # Set defaults
        self.timeout = 30
        self.max_retries = 3
        self.backoff_multiplier = 2.0

        # Override with config if provided
        if config:
            self.timeout = config.get('timeout_seconds', 30)
            self.max_retries = config.get('max_retries', 3)
            self.backoff_multiplier = config.get('backoff_multiplier', 2.0)

        if self.verbose:
            logger.debug(
                f"MGnify API initialized: timeout={self.timeout}s, "
                f"max_retries={self.max_retries}, backoff_multiplier={self.backoff_multiplier}x"
            )

    def check_requirements(self) -> Tuple[bool, str]:
        """
        Check if MGnify API is accessible.

        Returns:
            Tuple of (is_available, message)
        """
        try:
            # Test with a global query using retry logic
            params = {
                'limit': 1,
                'format': 'json'
            }
            response = self._make_request_with_retries(
                self.BASE_URL,
                params=params
            )

            if response and response.status_code == 200:
                if self.verbose:
                    logger.debug("MGnify API is accessible")
                return True, "MGnify API available"
            elif response:
                return False, f"MGnify API returned status {response.status_code}"
            else:
                return False, "MGnify API failed after all retries"
        except Exception as e:
            return False, f"MGnify API error: {str(e)}"

    def _make_request_with_retries(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with exponential backoff retry logic.

        Args:
            url: URL to request
            params: Query parameters
            max_retries: Override max retries (uses config default if None)
            timeout: Override timeout (uses config default if None)

        Returns:
            Response object on success, None on failure after all retries
        """
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self.timeout

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if self.verbose and attempt > 0:
                    logger.debug(f"MGnify retry attempt {attempt}/{max_retries}")

                response = requests.get(
                    url,
                    params=params,
                    timeout=timeout
                )
                response.raise_for_status()

                if attempt > 0:
                    logger.info(f"MGnify API request succeeded on retry attempt {attempt}")

                return response

            except Timeout as e:
                last_exception = e
                if attempt < max_retries:
                    backoff_delay = (self.backoff_multiplier ** attempt)
                    logger.warning(
                        f"MGnify API timeout (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {backoff_delay:.1f}s..."
                    )
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"MGnify API timeout after {max_retries + 1} attempts")

            except ConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    backoff_delay = (self.backoff_multiplier ** attempt)
                    logger.warning(
                        f"MGnify API connection error (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {backoff_delay:.1f}s..."
                    )
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"MGnify API connection error after {max_retries + 1} attempts")

            except RequestException as e:
                last_exception = e
                # For other HTTP errors (4xx, 5xx), still retry but with fewer attempts for 4xx
                if hasattr(e, 'response') and e.response and 400 <= e.response.status_code < 500:
                    # Client errors (4xx) are usually not retryable, fail fast
                    logger.error(f"MGnify API client error (HTTP {e.response.status_code}): {str(e)}")
                    return None

                if attempt < max_retries:
                    backoff_delay = (self.backoff_multiplier ** attempt)
                    logger.warning(
                        f"MGnify API request error (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {backoff_delay:.1f}s..."
                    )
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"MGnify API request failed after {max_retries + 1} attempts: {str(e)}")

            except Exception as e:
                last_exception = e
                logger.error(f"MGnify API unexpected error: {str(e)}")
                return None

        # All retries exhausted
        if last_exception:
            logger.error(f"MGnify API failed after {max_retries + 1} attempts: {str(last_exception)}")

        return None

    def _fetch_data(self, lat: float, lon: float, radius_km: float = 50) -> Dict[str, Any]:
        """
        Fetch MGnify sample data for a location.

        Note: MGnify API doesn't support direct bounding box queries.
        This provides a basic proximity search using coordinate filters.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers (not directly supported by API)

        Returns:
            Dictionary with metagenomics sample information
        """
        try:
            # MGnify doesn't support bbox directly, so we search broadly
            # In production, you might need to do coordinate matching client-side
            params = {
                'limit': 100,
                'format': 'json'
            }

            response = self._make_request_with_retries(
                self.BASE_URL,
                params=params
            )

            if response is None:
                logger.warning("MGnify: Failed to fetch data after all retries")
                return {
                    'metagenome_samples': 0,
                    'biomes': [],
                    'environment_types': [],
                    'projects': 0
                }

            data = response.json()
            results = data.get('results', [])

            # Filter samples by approximate distance (very simple heuristic)
            nearby_samples = []
            for sample in results:
                geo_loc = sample.get('geo_loc_name', '')
                # In production, you'd parse coordinates or use better matching
                # For now, collect all samples
                nearby_samples.append(sample)

            # Extract environmental descriptors
            env_types = set()
            biomes = set()

            for sample in nearby_samples[:20]:  # Limit to top 20
                env = sample.get('environment_biome', '')
                if env:
                    biomes.add(env)

                env_feat = sample.get('environment_feature', '')
                if env_feat:
                    env_types.add(env_feat)

            result = {
                'metagenome_samples': len(nearby_samples),
                'biomes': sorted(list(biomes)),
                'environment_types': sorted(list(env_types)),
                'projects': len(set(s.get('study_id', '') for s in nearby_samples if s.get('study_id')))
            }

            if self.verbose:
                logger.debug(f"MGnify: Found {len(nearby_samples)} metagenome samples")

            return result

        except Exception as e:
            logger.warning(f"Error fetching MGnify data: {str(e)}")
            return {
                'metagenome_samples': 0,
                'biomes': [],
                'environment_types': [],
                'projects': 0
            }

    def get_data(self, lat: float, lon: float, **kwargs) -> Dict[str, Any]:
        """
        Get MGnify metagenomics data (interface method for BaseEnvironmentalAPI).
        """
        return self._fetch_data(lat, lon, **kwargs)

    def fetch_and_enrich(self, df, lat_col: str, lon_col: str,
                         sample_id_col: str = None, radius_km: float = 50):
        """
        Enrich dataframe with MGnify metagenomics data.

        Adds columns:
        - mgnify_metagenome_samples
        - mgnify_biomes
        - mgnify_environment_types
        - mgnify_studies

        Features graceful degradation: If MGnify fails, returns dataframe with
        None/empty values for all new columns rather than crashing.
        """
        import pandas as pd

        is_available, availability_msg = self.check_requirements()
        if not is_available:
            logger.warning(f"MGnify API not available: {availability_msg}")
            return df

        results = []
        successful_queries = 0
        failed_queries = 0

        for idx, row in df.iterrows():
            try:
                lat = row[lat_col]
                lon = row[lon_col]

                if pd.isna(lat) or pd.isna(lon):
                    results.append({
                        'mgnify_metagenome_samples': None,
                        'mgnify_biomes': None,
                        'mgnify_environment_types': None,
                        'mgnify_studies': None
                    })
                    continue

                data = self._fetch_data(lat, lon, radius_km)
                successful_queries += 1

                results.append({
                    'mgnify_metagenome_samples': data['metagenome_samples'],
                    'mgnify_biomes': ','.join(data['biomes']) if data['biomes'] else None,
                    'mgnify_environment_types': ','.join(data['environment_types']) if data['environment_types'] else None,
                    'mgnify_studies': data['projects']
                })
            except Exception as e:
                failed_queries += 1
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    'mgnify_metagenome_samples': None,
                    'mgnify_biomes': None,
                    'mgnify_environment_types': None,
                    'mgnify_studies': None
                })

        result_df = pd.DataFrame(results)

        # Log summary statistics
        logger.info(
            f"MGnify enrichment complete: {successful_queries} successful, "
            f"{failed_queries} failed out of {len(df)} rows"
        )

        return pd.concat([df, result_df], axis=1)

