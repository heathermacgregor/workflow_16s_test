# workflow_16s/api/publication/apis/core.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class CoreAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.core.ac.uk/v3/search/works"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        publications = [] 
        query_data = {
            "q": accession, 
            "limit": 10, 
            "sort": "yearPublished:asc"
        }
        response = self.session.post("https://api.core.ac.uk/v3/search/works", json=query_data, timeout=15)
        response.raise_for_status()
        for item in response.json().get('results', []):
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('yearPublished', 'N/A')), 
                "doi": item.get('doi'), 
                "status": "Ready (CORE)"
            })
        return publications