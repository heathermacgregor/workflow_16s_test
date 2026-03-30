# workflow_16s/api/osm/universal_finder.py

import asyncio
import aiohttp
import pandas as pd
import logging
import argparse
from pathlib import Path

# Setup basic logging for standalone execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("workflow_16s.universal_finder")

def build_osm_tags(key: str, value: str) -> list:
    """Helper to generate node, way, and relation queries for a given OSM tag."""
    query = f'"{key}"="{value}"'
    return [f'node[{query}]', f'way[{query}]', f'relation[{query}]']

class UniversalFacilityFetcher:
    """
    Fetches global coordinates for any type of industrial, agricultural, 
    or commercial site using the OpenStreetMap Overpass API.
    """
    
    # Clean, scalable dictionary mapping plain-English requests to OSM tagging schemas
    FACILITY_TAGS = {
        # --- HEAVY INDUSTRY & ENERGY ---
        "mine": build_osm_tags("landuse", "quarry") + build_osm_tags("industrial", "mine"),
        "oil_refinery": build_osm_tags("man_made", "petroleum_works"),
        "chemical_plant": build_osm_tags("industrial", "chemical"),
        "power_plant": build_osm_tags("power", "plant"),
        "metal_processing": build_osm_tags("industrial", "metallurgical"),

        # --- WASTE & SANITATION ---
        "wastewater_plant": build_osm_tags("man_made", "wastewater_plant"),
        "water_treatment": build_osm_tags("man_made", "water_works"),
        "landfill": build_osm_tags("landuse", "landfill"),
        "waste_transfer": build_osm_tags("amenity", "waste_transfer_station"),
        "slaughterhouse": build_osm_tags("industrial", "slaughterhouse"),

        # --- AGRICULTURE & AQUACULTURE ---
        "farm_crop": build_osm_tags("landuse", "farmland"),
        "farm_animal": build_osm_tags("landuse", "animal_keeping"),
        "orchard": build_osm_tags("landuse", "orchard"),
        "aquaculture": build_osm_tags("landuse", "aquaculture"),
        "salt_pond": build_osm_tags("landuse", "salt_pond"),

        # --- "CONTROL" ENVIRONMENTS ---
        "national_park": build_osm_tags("boundary", "national_park"),
        "nature_reserve": build_osm_tags("leisure", "nature_reserve"),
        "glacier": build_osm_tags("natural", "glacier"),
        "wetland": build_osm_tags("natural", "wetland"),
        # --- BIOREMEDIATION & CONTAMINATION ---
        "brownfield": build_osm_tags("landuse", "brownfield"), # Abandoned industrial sites
        "oil_well": build_osm_tags("industrial", "wellsite") + build_osm_tags("man_made", "petroleum_well"),
        "scrap_yard": build_osm_tags("industrial", "scrap_yard"), # High heavy-metal runoff
        "paper_mill": build_osm_tags("industrial", "paper"), # High cellulose and chemical runoff
        # --- FOOD, BEVERAGE & FERMENTATION ---
        "vineyard": build_osm_tags("landuse", "vineyard"), # High yeast/fungal soil diversity
        "brewery": build_osm_tags("industrial", "brewery") + build_osm_tags("craft", "brewery"),
        "greenhouse": build_osm_tags("landuse", "greenhouse_horticulture"), # Controlled soil environments
        "dairy_processing": build_osm_tags("industrial", "dairy"),
        # --- EXTREMOPHILES & PRISTINE BASELINES ---
        "hot_spring": build_osm_tags("natural", "hot_spring"), # Thermophiles
        "tundra": build_osm_tags("natural", "tundra"), # Psychrophiles / Permafrost
        "bare_rock": build_osm_tags("natural", "bare_rock"), # Endoliths (rock-eating microbes)
        "desert_sand": build_osm_tags("natural", "sand"), # Desiccation-resistant microbes
        "grassland": build_osm_tags("natural", "grassland"), # Excellent standard baseline soil
        # --- HYDROLOGY & AQUATIC ---
        "water_well": build_osm_tags("man_made", "water_well"), # Groundwater/Aquifer access
        "reservoir": build_osm_tags("landuse", "reservoir"), # Stagnant surface water
        "marina": build_osm_tags("leisure", "marina"), # Coastal water with high boat traffic/hydrocarbons
    }

    def __init__(self):
        self.api_url = "https://overpass-api.de/api/interpreter"

    def _build_query(self, facility_type: str) -> str:
        if facility_type not in self.FACILITY_TAGS:
            valid = ", ".join(self.FACILITY_TAGS.keys())
            raise ValueError(f"Unknown facility type '{facility_type}'. Valid options: {valid}")
        
        tags = ";\n  ".join(self.FACILITY_TAGS[facility_type])
        
        # Extended timeout to 900s (15m) because global queries are massive
        query = f"""
        [out:json][timeout:900];
        (
          {tags};
        );
        out center;
        """
        return query

    async def fetch_locations(self, facility_type: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetches locations asynchronously with retry logic for API limits."""
        query = self._build_query(facility_type)
        logger.info(f"Querying OpenStreetMap for all '{facility_type}' locations globally...")

        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(self.api_url, data={'data': query}) as response:
                        if response.status in [429, 504]:
                            wait_time = 30 * (2 ** attempt)
                            logger.warning(f"Overpass API busy (HTTP {response.status}). Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        response.raise_for_status()
                        data = await response.json()
                        break # Success, exit retry loop
                except aiohttp.ClientError as e:
                    logger.error(f"Network error: {e}. Retrying...")
                    await asyncio.sleep(10)
            else:
                logger.error("Max retries exceeded. Failed to fetch from Overpass API.")
                return pd.DataFrame()

        facilities = []
        for element in data.get('elements', []):
            lat = element.get('lat') or element.get('center', {}).get('lat')
            lon = element.get('lon') or element.get('center', {}).get('lon')
            
            if lat and lon:
                tags = element.get('tags', {})
                facilities.append({
                    'facility_id': f"OSM_{element['type']}_{element['id']}",
                    'name': tags.get('name', f"Unnamed {facility_type.replace('_', ' ').title()}"),
                    'type': facility_type,
                    'latitude': lat,
                    'longitude': lon,
                    'country': tags.get('addr:country', 'Unknown'),
                    'animal_type': tags.get('animal', tags.get('livestock', 'Unknown')),
                    'operator': tags.get('operator', tags.get('brand', 'Unknown')),
                    'landuse': tags.get('landuse', tags.get('building', 'Unknown')),
                    'raw_metadata': str(tags)
                })

        df = pd.DataFrame(facilities)
        logger.info(f"Successfully found {len(df)} '{facility_type}' locations.")
        return df

    def save_to_tsv(self, df: pd.DataFrame, output_dir: str, facility_type: str):
        """Saves the DataFrame in the exact format expected by your ENA fetcher."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        file_path = path / f"{facility_type}.tsv"
        df.to_csv(file_path, sep='\t', index=False)
        logger.info(f"Saved facilities to {file_path}")

# ==================================================================================== #
# CLI Execution Block
# ==================================================================================== #
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Fetch global coordinates for environmental/industrial sites.")
    parser.add_argument(
        "-t", "--type", 
        type=str, 
        required=True, 
        choices=list(UniversalFacilityFetcher.FACILITY_TAGS.keys()),
        help="The type of facility to query from OpenStreetMap."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="project_01/01_raw_data/_nfc_facilities/",
        help="The output directory for the TSV file."
    )
    
    args = parser.parse_args()

    async def main():
        fetcher = UniversalFacilityFetcher()
        df = await fetcher.fetch_locations(args.type) 
        if not df.empty:
            fetcher.save_to_tsv(df, args.output, args.type)

    asyncio.run(main())