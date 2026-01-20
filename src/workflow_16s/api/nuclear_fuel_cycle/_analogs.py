# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_analogs.py
# ==================================================================================== #

import logging
import pandas as pd
import requests

logger = logging.getLogger("workflow_16s")

class Analogs:
    """
    Retrieves 'Contamination Analog' sites that share characteristics with 
    nuclear facilities (Heavy Metals, NORM, Acids, Heat, Salinity).
    """
    
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    # --- EXPANDED Q-CODES FOR ROBUST ANALOGS ---
    # 1. Chemical & Radiological (NORM)
    # Q1130068:  Superfund Site (Mixed Hazardous Waste)
    # Q1063637:  Coal-fired Power Station (Fly Ash / Radionuclides)
    # Q1955586:  Rare Earth Mine (Thorium/Uranium tailings)
    # Q15070265: Potash Mine (Potassium-40 / High Salinity)
    # Q162607:   Oil Platform (Scale/Sludge often high in Radium-226)
    
    # 2. Fertilizers & Acids (Nitrates / Phosphates)
    # Q21075778: Fertilizer Plant (Nitrate/Phosphate runoff)
    # Q13360639: Phosphate Mine (Uranium by-product / Acids)
    
    # 3. Metal Mining (Acid Mine Drainage / Heavy Metals)
    # Q1062633:  Gold Mine (Arsenic / Mercury / Cyanide)
    # Q1122672:  Copper Mine (Sulfides / Acid Drainage)
    # Q953606:   Aluminium Smelter (Fluorides / PAHs / Heat)
    
    # 4. Physical Stressors (Heat / Brine)
    # Q1056562:  Desalination Plant (Hyper-saline brine / Thermal pollution)
    # Q689745:   Geothermal Power Station (Radioactive scale / Thermal discharge)

    QUERY = """
    SELECT ?item ?itemLabel ?coord ?countryLabel ?typeLabel WHERE {
      VALUES ?type { 
        wd:Q1130068 wd:Q1063637 wd:Q1955586 wd:Q15070265 wd:Q162607
        wd:Q21075778 wd:Q13360639 wd:Q1062633 wd:Q1122672 wd:Q953606
        wd:Q1056562 wd:Q689745
      }
      ?item wdt:P31/wdt:P279* ?type .
      ?item wdt:P625 ?coord .
      OPTIONAL { ?item wdt:P17 ?country . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """

    def load(self) -> pd.DataFrame:
        logger.info("Querying Wikidata for expanded Analog sites (Mines, Fertilizers, Desalination, etc.)...")
        try:
            response = requests.get(
                self.SPARQL_ENDPOINT, 
                params={'format': 'json', 'query': self.QUERY},
                headers={'User-Agent': 'Workflow16S/1.0 (Research)'}
            )
            response.raise_for_status()
            data = response.json()
            
            rows = []
            for item in data['results']['bindings']:
                # Parse Coordinates
                wkt = item.get('coord', {}).get('value', '')
                lat, lon = None, None
                if 'Point(' in wkt:
                    try:
                        parts = wkt.replace('Point(', '').replace(')', '').split()
                        lon, lat = float(parts[0]), float(parts[1])
                    except: pass

                # Granular Categorization
                raw_type = item.get('typeLabel', {}).get('value', 'Unknown').lower()
                facility_type = "Analog - Industrial"
                
                if "superfund" in raw_type:     facility_type = "Analog - Superfund Site"
                elif "coal" in raw_type:        facility_type = "Analog - Coal Power (NORM)"
                elif "rare" in raw_type:        facility_type = "Analog - Rare Earth Mine"
                elif "phosphate" in raw_type:   facility_type = "Analog - Phosphate Mine"
                elif "potash" in raw_type:      facility_type = "Analog - Potash Mine"
                elif "fertilizer" in raw_type:  facility_type = "Analog - Fertilizer Plant"
                elif "gold" in raw_type:        facility_type = "Analog - Gold Mine"
                elif "copper" in raw_type:      facility_type = "Analog - Copper Mine"
                elif "aluminium" in raw_type:   facility_type = "Analog - Aluminium Smelter"
                elif "desalination" in raw_type: facility_type = "Analog - Desalination Plant"
                elif "geothermal" in raw_type:  facility_type = "Analog - Geothermal Plant"
                elif "oil" in raw_type:         facility_type = "Analog - Oil Platform (NORM)"

                rows.append({
                    'facility': item.get('itemLabel', {}).get('value'),
                    'facility_type': facility_type,
                    'facility_status': 'Unknown', 
                    'country': item.get('countryLabel', {}).get('value'),
                    'lat': lat,
                    'lon': lon,
                    'data_source': 'WIKIDATA_ANALOG',
                    'is_nuclear': False 
                })
            
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[~df['facility'].str.match(r'^Q\d+$')]
                
            logger.info(f"Loaded {len(df)} expanded analog sites.")
            return df
            
        except Exception as e:
            logger.error(f"Analog sites query failed: {e}")
            return pd.DataFrame()