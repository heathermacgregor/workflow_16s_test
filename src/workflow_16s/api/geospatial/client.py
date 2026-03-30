# workflow_16s/api/osm/client.py

import httpx
import logging
from typing import Dict, List, Tuple
from workflow_16s.utils.logger import get_logger

class OSMOverpassClient:
    """
    Client for the OpenStreetMap Overpass API to discover facility coordinates.
    """
    def __init__(self):
        self.url = "https://overpass-api.de/api/interpreter"
        self.logger = get_logger("workflow_16s")

    async def find_coordinates_by_tags(self, tags: Dict[str, str]) -> List[Tuple[float, float]]:
        """
        Queries OSM for features (nodes, ways, relations) matching specific tags.
        Returns a list of (latitude, longitude) tuples representing centroids.
        """
        # Build tag string: ["amenity"="landfill"]["status"="active"]
        tag_filter = "".join([f'["{k}"="{v}"]' for k, v in tags.items()])
        
        # [out:json] returns machine-readable data
        # [timeout:25] prevents hanging on massive global queries
        # out center; is critical—it reduces complex shapes (polygons) to a single point.
        query = f"""
        [out:json][timeout:25];
        (
          node{tag_filter};
          way{tag_filter};
          relation{tag_filter};
        );
        out center;
        """
        
        self.logger.debug(f"Executing Overpass Query: {query.strip()}")

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(self.url, data={"data": query})
                response.raise_for_status()
                data = response.json()
                
                elements = data.get('elements', [])
                coords = []
                
                for e in elements:
                    # 'lat'/'lon' exist for nodes. 
                    # 'center' dictionary exists for ways/relations due to 'out center'
                    lat = e.get('lat') or e.get('center', {}).get('lat')
                    lon = e.get('lon') or e.get('center', {}).get('lon')
                    
                    if lat is not None and lon is not None:
                        coords.append((float(lat), float(lon)))
                
                self.logger.info(f"OSM Query successful: Found {len(coords)} geometric centers.")
                return coords

            except httpx.HTTPStatusError as e:
                self.logger.error(f"OSM Overpass API Error ({e.response.status_code}): {e.response.text}")
                return []
            except Exception as e:
                self.logger.error(f"Unexpected error during OSM discovery: {e}")
                return []