"""
Environmental Data Aggregation Script

This script integrates multiple environmental APIs to collect comprehensive
environmental data for a given location. It's designed to be production-ready
with proper error handling, logging, and configuration management.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from pathlib import Path
# Constants
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 24
REQUEST_TIMEOUT = 30
MAX_WORKERS = 5

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++ API INTEGRATION CLASSES +++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BaseEnvironmentalAPI:
    """A base class for environmental API wrappers with retry logic."""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        self.api_key = api_key
        self.base_url = ""
        self.api_name = self.__class__.__name__
        self.verbose = verbose
        self.session = self._create_session_with_retries()
        
    def _create_session_with_retries(self) -> requests.Session:
        """Creates a requests.Session with a retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=3, read=3, connect=3, backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data if it exists and is not expired."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        # Check if cache is expired
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > CACHE_EXPIRY_HOURS * 3600:
            return None
            
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def _set_cached_data(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")
    
    def get_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Abstract method to be implemented by subclasses.
        Polls the API for data at a given latitude and longitude.
        Should return a dictionary of processed data or None on failure.
        """
        raise NotImplementedError