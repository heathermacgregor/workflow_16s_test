# workflow_16s/api/publication/apis/ieee_xplore.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class IEEExploreAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff() 
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        if not self.ieee_api_key: return []
        publications = []
        params = {
            "querytext": f'"{accession}"', 
            "apikey": self.ieee_api_key, 
            "max_records": 10, 
            "sortfield": "publication_year", 
            "sortorder": "asc"
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('articles', []):
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('publication_year', 'N/A')), 
                "doi": item.get('doi'), 
                "status": "Ready (IEEE)"
            })
        return publications