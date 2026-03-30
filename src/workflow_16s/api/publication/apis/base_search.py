# workflow_16s/api/publication/apis/base_search.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class BaseSearchAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.base-search.net/v2/search"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = [] 
        params = {
            "q": accession, 
            "format": "json", 
            "sort": "date:asc", 
            "limit": 10
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Use .get() chains to avoid 'tuple' errors if 'response' or 'docs' are missing
            docs = data.get('response', {}).get('docs', [])
            
            for item in docs:
                # 🟢 FIX: Be explicit with variables to avoid unpacking errors
                raw_year = item.get('year')
                raw_date = item.get('date', 'N/A')
                year = raw_year if raw_year else raw_date[:4]
                
                doi = item.get('doi')
                if isinstance(doi, list): 
                    doi = doi[0] if doi else None
                
                publications.append({
                    "bioproject_accession": accession, 
                    "publication_title": item.get('title', "Unknown Title"), 
                    "pub_year": str(year), 
                    "doi": doi, 
                    "status": "Ready (BASE)"
                })
        except Exception as e:
            self.logger.warning(f"BASE Search failed for {accession}: {e}")
            
        return publications