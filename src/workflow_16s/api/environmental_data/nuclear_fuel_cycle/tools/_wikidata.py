# workflow_16s/api/environmental_data/nuclear_fuel_cycle/tools/_wikidata.py

import requests
from typing import Optional

import pandas as pd

from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class Wikidata:
    """Retrieves global nuclear facility data using SPARQL queries."""
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    # Q-codes: Nuclear Power Station (Q134447), Uranium Mine (Q1059483), Reprocessing Plant (Q1140309)
    QUERY = """
    SELECT ?item ?itemLabel ?coord ?countryLabel ?typeLabel ?capacity ?start_time WHERE {
      VALUES ?type { wd:Q134447 wd:Q1059483 wd:Q1140309 }
      ?item wdt:P31/wdt:P279* ?type .
      ?item wdt:P625 ?coord .
      OPTIONAL { ?item wdt:P17 ?country . }
      OPTIONAL { ?item wdt:P2109 ?capacity . }
      OPTIONAL { ?item wdt:P580 ?start_time . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """
    logger = get_logger("workflow_16s")
    def __init__(self):
        pass

    def load(self) -> pd.DataFrame:
        self.logger.info("Querying Wikidata for global nuclear facilities...")
        try:
            response = requests.get(
                self.SPARQL_ENDPOINT, 
                params={'format': 'json', 'query': self.QUERY},
                headers={'User-Agent': 'Workflow16S/1.0 (Research)'}
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse JSON response
            rows = []
            for item in data['results']['bindings']:
                # Extract coordinates "Point(-123.45 67.89)"
                wkt = item.get('coord', {}).get('value', '')
                if 'Point(' in wkt:
                    lon, lat = wkt.replace('Point(', '').replace(')', '').split()
                else:
                    lat, lon = None, None

                rows.append({
                    'facility': item.get('itemLabel', {}).get('value'),
                    'facility_type': item.get('typeLabel', {}).get('value'),
                    'country': item.get('countryLabel', {}).get('value'),
                    'lat': float(lat) if lat else None,
                    'lon': float(lon) if lon else None,
                    'facility_capacity': item.get('capacity', {}).get('value'),
                    'facility_start_year': item.get('start_time', {}).get('value'),
                    'data_source': 'Wikidata'
                })
            
            df = pd.DataFrame(rows)
            # Filter out generic labels or bad data
            df = df[df['facility'] != df['facility_type']] 
            self.logger.info(f"Loaded {len(df)} facilities from Wikidata.")
            return df
            
        except Exception as e:
            self.logger.error(f"Wikidata query failed: {e}")
            return pd.DataFrame()