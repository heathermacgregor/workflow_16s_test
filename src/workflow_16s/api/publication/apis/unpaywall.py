# workflow_16s/api/publication/apis/unpaywall.py

import json
import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional

from workflow_16s.api.publication.apis.base import BaseAPI, with_http_backoff

class UnpaywallAPI(BaseAPI):
    def __init__(self, config, email, logger):
        # FIX: Pass config and email to the BaseAPI
        super().__init__(config, email)
        self.logger = logger
        self.base_url = "https://api.unpaywall.org/v2"
        self.rate_limit_seconds = 0.34 # 3 requests/sec
    
    
    def get_pdf_url(self, doi: str) -> Optional[str]:
        """
        Fetches open access PDF URL from Unpaywall API.
        
        Args:
            doi: DOI of the publication
            
        Returns:
            URL to open access PDF if available, None otherwise
        """
        if not doi:
            return None
        
        try:
            self._rate_limit('unpaywall')
            url = f"{self.base_url}/{doi}"
            params = {"email": self.email}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 404:
                self.logger.debug(f"DOI not found in Unpaywall: {doi}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Check for best_oa_location which has the free PDF
            best_oa = data.get('best_oa_location')
            if best_oa and best_oa.get('url_for_pdf'):
                self.logger.debug(f"Found open access PDF for {doi}")
                return best_oa['url_for_pdf']
            
            # Fallback to any OA location
            oa_locations = data.get('oa_locations', [])
            for location in oa_locations:
                if location.get('url_for_pdf'):
                    return location['url_for_pdf']
            
            self.logger.debug(f"No open access PDF found for {doi}")
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Unpaywall API error for DOI {doi}: {e}")
            return None