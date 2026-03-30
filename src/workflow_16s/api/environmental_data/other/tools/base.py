# workflow_16s/api/environmental_data/other/tools/base.py

import logging
import pickle
import hashlib
import requests
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from requests.adapters import HTTPAdapter
from typing import Any, Dict, Optional, Tuple
from urllib3.util.retry import Retry

from workflow_16s.utils.dir_utils import Project

from .cache import CacheManager
from workflow_16s.utils.logger import get_logger

from .constants import (
    CACHE_DB_PATH, CACHE_DIR, CACHE_EXPIRY_HOURS, 
    REQUEST_TIMEOUT, MAX_WORKERS
)

class BaseEnvironmentalAPI(ABC):
    """
    An abstract base class for environmental API wrappers.
    All logic remains identical, but now uses the SQLite-backed CacheManager.
    """
    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.logger = get_logger("workflow_16s")  # Initialize logger instance
        self.base_url: str = ""
        self.api_name: str = self.__class__.__name__
        self.session: requests.Session = self._create_session_with_retries()
        # Initialized with a Path object for compatibility
        self.cache_manager = CacheManager(CACHE_DB_PATH.parent / self.api_name)
        self.cache_hits = 0
        self.cache_misses = 0

    def _create_session_with_retries(self) -> requests.Session:
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
        return True, None

    @abstractmethod
    def get_data(self, lat: float, lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError