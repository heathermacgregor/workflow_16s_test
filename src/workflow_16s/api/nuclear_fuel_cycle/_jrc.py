# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_jrc.py
# ==================================================================================== #

import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger("workflow_16s")

class JRC:
    """
    Scrapes the Joint Research Centre (JRC) Open Access Research Infrastructures.
    Focuses on EU Nuclear Laboratories (Euratom).
    """
    
    # Official list of JRC Research Infrastructures
    URL = "https://joint-research-centre.ec.europa.eu/open-access-jrc-research-infrastructures_en"

    # Hardcoded locations for key JRC sites (often not explicit in the HTML list)
    SITE_COORDS = {
        "Geel": (51.1606, 5.0049),        # JRC Geel (Belgium)
        "Karlsruhe": (49.0969, 8.4312),   # JRC Karlsruhe (Germany)
        "Petten": (52.7886, 4.6874),      # JRC Petten (Netherlands)
        "Ispra": (45.8037, 8.6277),       # JRC Ispra (Italy)
        "Seville": (37.4054, -6.0084)     # JRC Seville (Spain)
    }

    def load(self) -> pd.DataFrame:
        logger.info("Scraping JRC/Euratom Research Infrastructures...")
        try:
            response = requests.get(self.URL, headers={'User-Agent': 'Workflow16S/1.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            facilities = []
            
            # The page typically lists facilities under headers like "Nuclear Laboratories"
            # We look for link blocks or list items that mention specific facilities
            content_div = soup.find('div', class_='ecl-editor')
            if content_div:
                # Naive strategy: find all strong tags or links that look like facility names
                # Improved strategy: Look for the specific known nuclear blocks
                
                # Keywords to identify JRC Nuclear facilities
                keywords = ["ActUsLab", "EMMA", "EUFRAT", "GELINA", "HADES", "MONNET", "RADMET", "SMPA"]
                
                text_blocks = content_div.get_text().split('\n')
                for line in text_blocks:
                    for kw in keywords:
                        if kw in line:
                            # Attempt to infer site
                            site = "Unknown"
                            coords = (None, None)
                            
                            if "Geel" in line: site = "Geel"
                            elif "Karlsruhe" in line: site = "Karlsruhe"
                            elif "Petten" in line: site = "Petten"
                            elif "Ispra" in line: site = "Ispra"
                            
                            if site in self.SITE_COORDS:
                                coords = self.SITE_COORDS[site]
                                
                            facilities.append({
                                'facility': f"JRC {kw} ({site})",
                                'facility_type': "Research Infrastructure (Euratom)",
                                'facility_status': "Operating",
                                'country': "European Union", # Generic, or specific per site
                                'lat': coords[0],
                                'lon': coords[1],
                                'data_source': "JRC"
                            })
                            break # Avoid double adding

            df = pd.DataFrame(facilities)
            df = df.drop_duplicates(subset=['facility'])
            
            # Map specific countries
            country_map = {
                "Geel": "Belgium", "Karlsruhe": "Germany", "Petten": "Netherlands", "Ispra": "Italy"
            }
            if not df.empty:
                df['country'] = df['facility'].apply(lambda x: next((v for k, v in country_map.items() if k in x), "European Union"))

            logger.info(f"Loaded {len(df)} JRC facilities.")
            return df

        except Exception as e:
            logger.error(f"JRC Scraper failed: {e}")
            return pd.DataFrame()