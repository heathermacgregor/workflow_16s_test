# workflow_16s/api/environmental_data/nuclear_fuel_cycle/tools/_analogs.py

import pandas as pd
import requests

from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
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
    logger = get_logger("workflow_16s")

    def load(self) -> pd.DataFrame:
        self.logger.info("Querying Wikidata for expanded Analog sites (Splitting queries to avoid timeout)...")
        
        expected_cols = [
            'facility', 'facility_type', 'facility_status', 'country', 
            'lat', 'lon', 'data_source', 'is_nuclear'
        ]

        target_types = [
            ("wd:Q15070265", "Analog - Gold Mine"),
            ("wd:Q21075778", "Analog - Rare Earth Mine"),
            ("wd:Q13360639", "Analog - Phosphate Mine"),
            ("wd:Q1062633",  "Analog - Potash Mine"),
            ("wd:Q1130068",  "Analog - In-situ Leach Mine"),
            ("wd:Q162607",   "Analog - Coal Power (NORM)"),
            ("wd:Q953606",   "Analog - Fertilizer Plant"),
            ("wd:Q1056562",  "Analog - Aluminium Smelter"),
            ("wd:Q689745",   "Analog - Desalination Plant"),
            ("wd:Q1955586",  "Analog - Underground Mine"),
            ("wd:Q1063637",  "Analog - Open Pit Mine"),
            ("wd:Q1122672",  "Analog - Industrial Mine"), 
        ]
        
        url = "https://query.wikidata.org/sparql"
        headers = {'User-Agent': 'Workflow16S/1.0 (Research)'}
        all_rows = []
        
        import requests
        import time

        for qid, type_label in target_types:
            # [FIX] Added PREFIXES to ensure the query is valid
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            
            SELECT ?item ?itemLabel ?coord ?countryLabel WHERE {{
              ?item wdt:P31/wdt:P279* {qid} .
              ?item wdt:P625 ?coord .
              OPTIONAL {{ ?item wdt:P17 ?country . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            LIMIT 2000
            """
            
            success = False
            for attempt in range(3):
                try:
                    r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers, timeout=45)
                    r.raise_for_status()
                    data = r.json()
                    
                    batch_rows = []
                    for entry in data['results']['bindings']:
                        try:
                            coord_raw = entry.get('coord', {}).get('value', '')
                            lat, lon = None, None
                            if "Point(" in coord_raw:
                                parts = coord_raw.replace("Point(", "").replace(")", "").split()
                                lon, lat = float(parts[0]), float(parts[1])

                            facility_id = entry.get('item', {}).get('value', '')
                            
                            batch_rows.append({
                                'facility': entry.get('itemLabel', {}).get('value', 'Unknown'),
                                'facility_id': facility_id, 
                                'facility_type': type_label,
                                'facility_status': 'Unknown',
                                'country': entry.get('countryLabel', {}).get('value', None),
                                'lat': lat, 
                                'lon': lon,
                                'data_source': 'WIKIDATA_ANALOG',
                                'is_nuclear': False
                            })
                        except Exception: 
                            continue
                            
                    all_rows.extend(batch_rows)
                    if len(batch_rows) > 0:
                        self.logger.info(f"  + Loaded {len(batch_rows)} items for {type_label}")
                    success = True
                    break 

                except Exception as e:
                    self.logger.debug(f"  - Attempt {attempt+1} failed for {type_label}: {e}")
                    time.sleep(2)
            
            if not success:
                self.logger.warning(f"  ! Skipping {type_label} (No data or timeout).")
            
            time.sleep(1)

        df = pd.DataFrame(all_rows)
        
        if df.empty:
            self.logger.warning("No analog data retrieved from Wikidata.")
            return pd.DataFrame(columns=expected_cols)

        if 'facility_id' in df.columns:
            df = df.drop_duplicates(subset=['facility_id'])
            df = df.drop(columns=['facility_id'])

        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[~df['facility'].str.match(r'^Q\d+$', na=False)]
        
        self.logger.info(f"Successfully compiled {len(df)} total analog facilities.")
        return df