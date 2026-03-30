# workflow_16s/api/publication/apis/base.py

import json
import requests
import logging
import os
import random
import re
import time
import requests
import pandas as pd
from functools import wraps
from typing import Any, Optional, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def with_http_backoff(max_retries=5, base_delay=1.0, max_delay=32.0):
    """Decorator for handling HTTP 429 Too Many Requests errors with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        if retries >= max_retries:
                            raise e
                        
                        sleep_time = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * sleep_time)
                        total_sleep = sleep_time + jitter
                        
                        print(f"[WARNING] Rate Limit Hit. Retrying in {total_sleep:.2f}s... (Attempt {retries + 1}/{max_retries})")
                        time.sleep(total_sleep)
                        retries += 1
                    else:
                        raise e
        return wrapper
    return decorator


class BaseAPI:
    def __init__(self, config: Any, email: str, timeout: tuple = (10, 30)):
        self.config = config
        self.email = email
        self.timeout = timeout
        
        # Setup rate limiting tracking structures
        self.last_request_times = {}
        self.rate_limits = {'default': 0.5} # 2 requests per second default
        
        self._initialize_api_keys()
        
        # Initialize the robust, pooled HTTP session
        self.session = self._build_robust_session()

    def _build_robust_session(self) -> requests.Session:
        """Creates a persistent HTTP session with connection pooling and retries."""
        session = requests.Session()
        
        # Default headers for all API requests
        session.headers.update({
            "User-Agent": f"BioProjectPublicationExtractor/2.0 (mailto:{self.email})",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        
        # Configure automatic retries for transient server errors
        retry_strategy = Retry(
            total=2,  # Try 4 times before giving up
            backoff_factor=0.5,  # Wait 1s, 2s, 4s, 8s between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these HTTP codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Create a connection pool (match pool size to your max_workers in fetcher.py)
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _initialize_api_keys(self):
        # API keys with fallback to environment variables
        # STANDARDIZED: Config first, then environment variable fallback
        self.dimensions_api_key = self.config.credentials.dimensions_api_key or os.getenv("DIMENSIONS_API_KEY")
        self.ieee_api_key = self.config.credentials.ieee_api_key or os.getenv("IEEE_XPLORE_API_KEY")
        self.mendeley_api_key = self.config.credentials.mendeley_api_key or os.getenv("MENDELEY_API_KEY")
        self.springer_api_key = self.config.credentials.springer_api_key or os.getenv("SPRINGER_NATURE_API_KEY")
        self.unpaywall_email = self.email  # Unpaywall requires email

        # REMOVED: os.environ writes (security risk, creates confusion)
        # Credentials are now accessed directly from self.* attributes 
            
    def _wait_for_rate_limit(self, api_name: str = 'default'):
        """🟢 Added: The explicit method NCBIAPI was looking for."""
        current = time.time()
        last_request = self.last_request_times.get(api_name, 0)
        elapsed = current - last_request
        wait_time = self.rate_limits.get(api_name, self.rate_limits['default']) - elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
            
        self.last_request_times[api_name] = time.time()

    def _rate_limit(self, api_name: str):
        """Alias for _wait_for_rate_limit to support different coding styles."""
        self._wait_for_rate_limit(api_name)
        
    def _build_smart_queries(self, ena_metadata: pd.DataFrame) -> List[str]:
        """Generates fuzzy/Boolean queries from metadata using dynamic column matching."""
        if ena_metadata is None or ena_metadata.empty:
            return []

        smart_queries = []
        # Expanded stopwords to keep the query focused on unique biological/location concepts
        stopwords = {
            "the", "and", "of", "to", "in", "a", "is", "for", "from", "with", "by", "on", 
            "as", "an", "this", "that", "at", "16s", "rrna", "amplicon", "sequencing", 
            "microbiome", "microbiota", "community", "analysis", "data", "study", 
            "samples", "using", "bacterial", "bacterium", "bacteria", "based", "high", 
            "throughput", "environmental", "project", "gene", "diversity"
        }

        # 1. Dynamically find Author/Center columns
        author_keywords = ['center', 'broker', 'investigator', 'author', 'submitter', 'institute']
        author_cols = [col for col in ena_metadata.columns if any(k in col.lower() for k in author_keywords)]
        
        authors = []
        for col in author_cols:
            vals = ena_metadata[col].dropna().unique()
            if len(vals) > 0:
                first_val_str = str(vals[0]).strip()
                if first_val_str and first_val_str.lower() != 'nan':
                    authors.append(first_val_str.split()[0])

        # 2. Dynamically find Title/Description columns
        text_keywords = ['title', 'description', 'abstract', 'summary', 'objective', 'name']
        text_cols = [col for col in ena_metadata.columns if any(k in col.lower() for k in text_keywords)]
        
        text_corpus = ""
        for col in text_cols:
            text_corpus += " " + " ".join(str(v) for v in ena_metadata[col].dropna().unique())

        if text_corpus.strip():
            words = re.findall(r'\b[a-zA-Z]{5,}\b', text_corpus.lower())
            
            from collections import Counter
            word_counts = Counter([w for w in words if w not in stopwords])
            
            keywords = [word for word, count in word_counts.most_common(4)]
            
            if keywords:
                base_query = " AND ".join(keywords)
                if authors:
                    smart_queries.append(f"({authors[0]}) AND ({base_query})")
                else:
                    smart_queries.append(f"({base_query})")

        return smart_queries