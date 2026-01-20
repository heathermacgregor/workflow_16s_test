# workflow_16s/api/environmental_data/other/__init__.py

"""
Defines the abstract base class for all environmental API wrappers and the caching decorator.
"""

import logging
import pickle
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("workflow_16s")

# Constants
CACHE_DIR = Path("./cache/env_other")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRY_HOURS = 720 # 30 days
REQUEST_TIMEOUT = 30
MAX_WORKERS = 5

# ==============================================================================
# ============================= CACHING UTILITIES ==============================
# ==============================================================================

class CacheManager:
    """Manages reading and writing API responses to a local file cache."""
    def __init__(self, cache_dir: Path, expiry_hours: int = CACHE_EXPIRY_HOURS):
        self.cache_dir = cache_dir
        self.expiry_delta = timedelta(hours=expiry_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Creates a unique and safe filename from the cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a non-expired result from the cache."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        file_mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - file_mod_time > self.expiry_delta:
            cache_path.unlink()
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            return None

    def set(self, key: str, data: Any):
        """Saves a result to the cache."""
        if data is None: return # Do not cache None responses
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

def cache_api_call(func):
    """Decorator to cache the results of an API method call."""
    @wraps(func)
    def wrapper(self: 'BaseEnvironmentalAPI', *args, **kwargs):
        # Create a stable key from the function name and arguments
        arg_str = str(args) + str(sorted(kwargs.items()))
        key_str = f"{self.api_name}_{func.__name__}_{arg_str}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()

        # Check for cached data
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            self.cache_hits += 1  # Increment hit counter
            return cached_data

        # If not cached, call the original function
        self.cache_misses += 1  # Increment miss counter
        result = func(self, *args, **kwargs)

        # Cache the new result
        self.cache_manager.set(cache_key, result)
        return result
    return wrapper

# ==============================================================================
# ======================== BASE API INTEGRATION CLASS ==========================
# ==============================================================================

class BaseEnvironmentalAPI(ABC):
    """
    An abstract base class for environmental API wrappers.
    """
    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.base_url: str = ""
        self.api_name: str = self.__class__.__name__
        self.session: requests.Session = self._create_session_with_retries()
        # Each API instance gets its own cache directory
        self.cache_manager = CacheManager(CACHE_DIR / self.api_name)
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_session_with_retries(self) -> requests.Session:
        """Creates a requests.Session with a robust retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=3, read=3, connect=3, backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def check_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Checks if all prerequisites (e.g., API keys) are met.
        """
        return True, None

    @abstractmethod
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Abstract method to fetch and process data from the specific API.
        """
        raise NotImplementedError