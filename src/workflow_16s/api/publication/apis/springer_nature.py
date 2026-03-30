# workflow_16s/api/publication/apis/springer_nature.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class SpringerNatureAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "http://api.springernature.com/openaccess/json"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()   
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        if not self.springer_api_key: return []
        publications = [] 
        params = {
            "q": f'fulltext:"{accession}"', 
            "api_key": self.springer_api_key, 
            "p": 10
        }
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        for item in response.json().get('records', []):
            year = item.get('publicationDate', 'N/A')[:4]
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(year), 
                "doi": item.get('doi'), 
                "status": "Ready (Springer)"
            })
        return publications