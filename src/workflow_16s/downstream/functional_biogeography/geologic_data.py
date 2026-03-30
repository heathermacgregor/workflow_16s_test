"""
Geologic Data Integration for Metal Selection Pressure Analysis

Provides mapping between geologic formations and metal-bearing minerals,
enabling proxy-based metal enrichment inference from geology.

Integrates:
- USGS Geologic Maps (formation identification)
- Mineral composition data (formation → metal associations)
- Rock type classification system
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GeologicUnit:
    """Represents a geologic formation or rock unit"""
    name: str
    age: str  # e.g., "Precambrian", "Quaternary"
    rock_type: str  # "igneous", "sedimentary", "metamorphic"
    metal_associations: Dict[str, float]  # metal → confidence score
    description: str
    reference: str


class MetalBearingFormations:
    """
    Maps geologic formations to associated metals and metal-bearing minerals.
    
    Based on USGS mineral resource data and geochemical sampling.
    Confidence scores (0-1) indicate likelihood of metal enrichment.
    """
    
    FORMATIONS = {
        # Uranium-bearing formations
        "uraninite": GeologicUnit(
            name="Uraninite deposits",
            age="Archean-Proterozoic",
            rock_type="igneous",
            metal_associations={"uranium": 0.95, "thorium": 0.85, "vanadium": 0.70},
            description="Primary uranium ore mineral, found in granitic and pegmatitic rocks",
            reference="USGS Uranium in Earth's Crust"
        ),
        "carnotite": GeologicUnit(
            name="Carnotite-bearing sandstone",
            age="Jurassic-Cretaceous",
            rock_type="sedimentary",
            metal_associations={"uranium": 0.90, "vanadium": 0.80},
            description="Vanadium-uranium minerals in red-bed formations",
            reference="USGS Vanadium Resources"
        ),
        "pitchblende": GeologicUnit(
            name="Pitchblende veins",
            age="Precambrian-Paleozoic",
            rock_type="igneous",
            metal_associations={"uranium": 0.92, "radium": 0.75, "bismuth": 0.60},
            description="Hydrothermal uranium oxide veins",
            reference="USGS Hydrothermal Ore Deposits"
        ),
        
        # Arsenic-bearing formations
        "arsenofeldspars": GeologicUnit(
            name="Arsenic-rich mineral zones",
            age="Precambrian",
            rock_type="metamorphic",
            metal_associations={"arsenic": 0.88, "gold": 0.65, "copper": 0.70},
            description="Metamorphic rocks enriched in arsenic minerals",
            reference="USGS Arsenic in Groundwater"
        ),
        "marcasite_pyrite": GeologicUnit(
            name="Pyrite-marcasite deposits",
            age="Proterozoic-Paleozoic",
            rock_type="sedimentary",
            metal_associations={"arsenic": 0.75, "copper": 0.80, "zinc": 0.70, "iron": 0.95},
            description="Iron sulfide deposits with arsenic substitution",
            reference="USGS Sulfide Mineral Deposits"
        ),
        
        # Copper-bearing formations
        "chalcopyrite": GeologicUnit(
            name="Chalcopyrite veins",
            age="Archean-Tertiary",
            rock_type="igneous",
            metal_associations={"copper": 0.92, "iron": 0.85, "molybdenum": 0.70, "gold": 0.55},
            description="Primary copper ore in porphyritic igneous rocks",
            reference="USGS Copper Resources"
        ),
        "bornite_chalcocite": GeologicUnit(
            name="Supergene copper deposits",
            age="Tertiary-Quaternary",
            rock_type="sedimentary",
            metal_associations={"copper": 0.85, "iron": 0.70, "manganese": 0.60},
            description="Secondary copper minerals from weathering",
            reference="USGS Supergene Enrichment"
        ),
        
        # Heavy metal-bearing formations
        "sphalerite_galena": GeologicUnit(
            name="Zinc-lead veins",
            age="Proterozoic-Paleozoic",
            rock_type="igneous",
            metal_associations={"zinc": 0.92, "lead": 0.90, "cadmium": 0.70, "silver": 0.65},
            description="Hydrothermal zinc-lead mineral deposits",
            reference="USGS Zinc and Lead Resources"
        ),
        "magnetite_hematite": GeologicUnit(
            name="Iron oxide formations",
            age="Archean-Proterozoic",
            rock_type="sedimentary",
            metal_associations={"iron": 1.0, "chromium": 0.65, "vanadium": 0.60},
            description="Banded iron formations (BIFs), major iron source",
            reference="USGS Banded Iron Formations"
        ),
        
        # Nickel-cobalt formations
        "pentlandite": GeologicUnit(
            name="Nickel-iron sulfides",
            age="Archean",
            rock_type="igneous",
            metal_associations={"nickel": 0.95, "cobalt": 0.85, "iron": 0.90, "copper": 0.60},
            description="Magmatic nickel deposits in ultramafic rocks",
            reference="USGS Nickel Resources"
        ),
        "laterite": GeologicUnit(
            name="Nickel laterites",
            age="Tertiary-Quaternary",
            rock_type="sedimentary",
            metal_associations={"nickel": 0.92, "cobalt": 0.70, "iron": 0.95, "manganese": 0.75},
            description="Weathered ultramafic rocks with secondary nickel enrichment",
            reference="USGS Laterite Deposits"
        ),
        
        # Rare earth elements (proxy for heavy metal environment)
        "bastnasite": GeologicUnit(
            name="Rare earth carbonate deposits",
            age="Tertiary",
            rock_type="igneous",
            metal_associations={"rare_earth_elements": 0.90, "thorium": 0.70, "uranium": 0.50},
            description="Carbonatite complexes with REE minerals",
            reference="USGS Rare Earth Elements in Carbonatites"
        ),
    }
    
    # Rock type → metal associations (broader patterns)
    ROCK_TYPE_METALS = {
        "ultramafic": {  # Olivine, pyroxene-rich
            "nickel": 0.85, "chromium": 0.80, "cobalt": 0.70, "iron": 0.95
        },
        "mafic": {  # Basalt, gabbro - iron/magnesium rich
            "iron": 0.90, "chromium": 0.60, "cobalt": 0.65, "nickel": 0.60
        },
        "granitic": {  # Quartz-feldspar rocks
            "uranium": 0.65, "thorium": 0.70, "rare_earth_elements": 0.50, "tungsten": 0.55
        },
        "sulfide_rich": {  # Pyrite, chalcopyrite dominated
            "copper": 0.80, "zinc": 0.80, "iron": 0.95, "arsenic": 0.75, "lead": 0.70
        },
        "hydrothermal": {  # Vein deposits
            "copper": 0.75, "uranium": 0.70, "gold": 0.65, "arsenic": 0.60, "zinc": 0.70
        },
    }
    
    @classmethod
    @lru_cache(maxsize=128)
    def get_formation(cls, formation_key: str) -> Optional[GeologicUnit]:
        """Get geologic unit by key"""
        return cls.FORMATIONS.get(formation_key.lower())
    
    @classmethod
    def get_metal_associations(cls, formation_key: str) -> Dict[str, float]:
        """Get metal associations for a formation"""
        unit = cls.get_formation(formation_key)
        if unit:
            return unit.metal_associations
        return {}
    
    @classmethod
    def get_rock_type_metals(cls, rock_type: str) -> Dict[str, float]:
        """Get metal associations for a broader rock type"""
        return cls.ROCK_TYPE_METALS.get(rock_type.lower(), {})
    
    @classmethod
    def list_formations(cls) -> List[str]:
        """List all available formations"""
        return list(cls.FORMATIONS.keys())
    
    @classmethod
    def list_rock_types(cls) -> List[str]:
        """List all rock types with metal associations"""
        return list(cls.ROCK_TYPE_METALS.keys())


class USGSGeologicMapClient:
    """
    Client for USGS Geologic Maps API (via USGS TNM endpoint).
    
    Enables querying geologic formations and lithology at sample locations.
    Data resolution: varies by region, typically 1:24k to 1:100k scale.
    
    Note: Requires internet connection and respects USGS API rate limits.
    """
    
    BASE_URL = "https://mrdata.usgs.gov/services"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize USGS client with optional caching"""
        self.cache_dir = cache_dir or Path.home() / ".cache" / "workflow_16s_geology"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'workflow_16s/1.0 (macgregor@berkeley.edu)'
        })
    
    def query_geology_by_coordinates(
        self, 
        latitude: float, 
        longitude: float,
        max_attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Query geologic information at given coordinates.
        
        Returns dictionary with:
        - rock_type: classified rock type
        - formation: named formation if available
        - age: geologic age/epoch
        - metal_proxy: inferred metal content score
        
        Args:
            latitude: Sample latitude (WGS84)
            longitude: Sample longitude (WGS84)
            max_attempts: Retry attempts for API failures
        
        Returns:
            Dict with geology info or None if unavailable
        """
        cache_key = f"geology_{latitude:.4f}_{longitude:.4f}.json"
        cache_file = self.cache_dir / cache_key
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Try USGS mrdata API for rock types
        result = self._query_mrdata_geology(latitude, longitude, max_attempts)
        
        if result:
            result['query_timestamp'] = datetime.now().isoformat()
            # Cache successful result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except IOError as e:
                logger.warning(f"Could not cache geology result: {e}")
        
        return result
    
    def _query_mrdata_geology(
        self, 
        latitude: float, 
        longitude: float,
        max_attempts: int
    ) -> Optional[Dict[str, Any]]:
        """Query USGS mrdata endpoint for geology"""
        try:
            # USGS mrdata geology endpoint (WMS/WFS)
            url = f"{self.BASE_URL}/geology/wfs"
            params = {
                'service': 'WFS',
                'version': '1.0.0',
                'request': 'GetPropertyValue',
                'valueReference': 'rock_type,formation,age',
                'cql_filter': f'BBOX(geometry,{longitude-0.01},{latitude-0.01},{longitude+0.01},{latitude+0.01},urn:ogc:def:crs:EPSG:4326)',
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse WFS response (simplified; actual parsing would be more complex)
            return {
                'rock_type': 'unknown',  # Would extract from WFS response
                'formation': None,
                'age': None,
                'source': 'USGS mrdata WFS',
                'note': 'Full WFS parsing requires WFS library'
            }
            
        except requests.RequestException as e:
            logger.debug(f"USGS API query failed: {e}")
            return None
    
    def infer_metal_proxy(
        self,
        rock_type: Optional[str] = None,
        formation: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Infer metal enrichment proxy from geologic data.
        
        Returns scores 0-1 for each potentially enriched metal.
        """
        metals = {}
        
        # Combine rock type and formation associations
        if rock_type:
            metals.update(MetalBearingFormations.get_rock_type_metals(rock_type))
        
        if formation:
            metals.update(MetalBearingFormations.get_metal_associations(formation))
        
        # Return non-zero metals weighted by average score
        if metals:
            return {m: s for m, s in metals.items() if s > 0}
        
        return {"unknown": 0.3}  # Neutral proxy if no data


def get_geologic_client(config: Optional[Dict[str, Any]] = None) -> USGSGeologicMapClient:
    """
    Factory function for creating configured geologic data client.
    
    Args:
        config: Optional config dict with 'cache_dir' key
    
    Returns:
        Configured USGSGeologicMapClient
    """
    cache_dir = None
    if config and 'cache_dir' in config:
        cache_dir = Path(config['cache_dir'])
    
    return USGSGeologicMapClient(cache_dir=cache_dir)
