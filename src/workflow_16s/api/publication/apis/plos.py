# workflow_16s/api/publication/apis/plos.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class PLOSAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "http://api.plos.org/search"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = [] 
        params = {
            "q": f'"{accession}"', 
            "fl": "id,publication_date,title", 
            "wt": "json", 
            "rows": 10, 
            "sort": "publication_date asc"
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('response', {}).get('docs', []):
            year = item.get('publication_date', 'N/A')[:4]
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(year), 
                "doi": item.get('id'), 
                "status": "Ready (PLOS)"
            })
        return publications