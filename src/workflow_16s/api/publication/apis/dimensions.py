# workflow_16s/api/publication/apis/dimensions.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class DimensionsAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.dimensions.ai/api/dsl.json"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    @with_http_backoff()
    def get_publications_from_accession(self, accession: str) -> List[Dict[str, Any]]:
        if not self.dimensions_api_key: return []
        publications = [] 
        query = f'search publications for "{accession}" return publications[title,year,doi] sort by year asc limit 10'
        response = self.session.post(self.base_url, data=query.encode('utf-8'), timeout=15)
        response.raise_for_status()
        for item in response.json().get('publications', []):
            publications.append({
                "bioproject_accession": accession, 
                "publication_title": item.get('title', "Unknown Title"), 
                "pub_year": str(item.get('year', 'N/A')), 
                "doi": item.get('doi'), 
                "status": "Ready (Dimensions)"
            })
        return publications