# workflow_16s/api/publication/apis/mendeley.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class MendeleyAPI(BaseAPI):
    def __init__(self, config: Any, email: str, api_key: str, logger):
        super().__init__(config, email)
        self.api_key = api_key
        self.logger = logger
        self.base_url = "https://api.mendeley.com/catalog"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        if not self.api_key: return []
        publications = [] 
        headers = {"Authorization": f"Bearer {self.api_key}"}, 
        params = {
            "query": f'"{accession}"', 
            "view": "all", 
            "limit": 10, 
            "sort": "year", 
            "direction": "asc"
        }
        response = self.session.get("https://api.mendeley.com/catalog", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json():
            doi = item.get('identifiers', {}).get('doi')
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('year', 'N/A')), 
                "doi": doi, 
                "status": "Ready (Mendeley)"
            })
        return publications