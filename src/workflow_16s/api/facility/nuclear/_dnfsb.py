# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_dnfsb.py
# ==================================================================================== #

# Standard Imports
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urljoin

# Third Party Imports
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s.config import AppConfig
from workflow_16s.utils.dir_utils import Project

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class DNFSBFacilityDB:
    """
    Class for downloading and processing DOE nuclear facilities from the DNFSB 
    website. Extracts facility names, primary functions, status, state, and 
    coordinates.
    """
    BASE_URL = "https://www.dnfsb.gov"
    # UPDATED URL (Old: /doe-sites -> New: /doe-defense-nuclear-sites)
    SITE_LIST_URL = f"{BASE_URL}/doe-defense-nuclear-sites"

    # Browser-like headers to prevent 403/404 blocks
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    def __init__(
        self, output_path: Optional[Union[str, Path]] = None, verbose: bool = True
    ):
        self.output_path = Path(output_path) if output_path else None
        self.verbose = verbose
        self.result: pd.DataFrame = pd.DataFrame()

    def run(self) -> pd.DataFrame:
        """Scrape the facility list and fetch coordinates."""
        try:
            facilities = self._fetch_facility_list()
        except Exception as e:
            logger.error(f"DNFSB Scraper failed: {e}")
            return pd.DataFrame()

        if not facilities:
            logger.warning("No DNFSB facilities found. Returning empty dataset.")
            return pd.DataFrame()

        self.log(f"Found {len(facilities)} facilities. Fetching coordinates...")

        for f in facilities:
            lat, lon = self._fetch_facility_coordinates(f.get("facility_link"))
            f["latitude"] = lat
            f["longitude"] = lon
            time.sleep(0.5)  # be polite

        self.result = pd.DataFrame(facilities)
        self.log(f"Loaded {self.result.shape[0]} DNFSB facilities")
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.result.to_csv(self.output_path, sep="\t", index=False)
            self.log(f"Saved results to {self.output_path}")

        return self.result
    
    def log(self, msg):
        """Log message if verbose mode is enabled."""
        return (lambda msg: logger.debug(msg)) if self.verbose else (lambda *_: None)

    def _fetch_facility_list(self) -> List[dict]:
        """Scrape the main DOE sites table and extract facility info + links."""
        try:
            resp = requests.get(self.SITE_LIST_URL, headers=self.HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not access DNFSB site list ({self.SITE_LIST_URL}): {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        facilities = []

        if table:
            rows = table.find_all("tr")[1:]  # skip header # type: ignore
            for row in rows:
                cols = row.find_all("td") # type: ignore
                # Flexible parsing: older tables had 4 cols, newer might have 2-3
                if len(cols) >= 2: 
                    link_tag = cols[0].find("a") # type: ignore
                    name = link_tag.text.strip() if link_tag else cols[0].get_text(strip=True) # type: ignore
                    href = urljoin(self.BASE_URL, link_tag["href"]) if link_tag else None # type: ignore
                    
                    # Safe extraction of other columns if they exist
                    status = cols[1].get_text(strip=True) if len(cols) > 1 else "Unknown"
                    desc = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                    facilities.append({
                        "facility": name,
                        "facility_description": desc,
                        "facility_status": status,
                        "facility_country": "United States of America",
                        "facility_link": href,
                        "facility_data_source": "DNFSB"
                    })
        else:
            logger.warning("Could not find facility table on DNFSB page.")
            
        return facilities

    def _fetch_facility_coordinates(self, facility_url: Optional[str]) -> tuple:
        """Fetch latitude and longitude from a facility detail page."""
        if not facility_url:
            return None, None

        try:
            resp = requests.get(facility_url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Updated selector logic: Look for geolocation block
            geo_div = soup.find("div", class_="block-field-blocknodedoe-sitefield-site-geolocation")
            
            # Fallback selector if site changed
            if not geo_div:
                geo_div = soup.find("div", class_="geolocation-location")

            if not geo_div:
                return None, None

            # Try extracting from meta tags (common in Drupal sites)
            lat_tag = geo_div.find('meta', attrs={'property': 'latitude'}) # type: ignore
            lon_tag = geo_div.find('meta', attrs={'property': 'longitude'}) # type: ignore
            
            if lat_tag and lon_tag:
                return self._dms_to_decimal(lat_tag.get("content")), self._dms_to_decimal(lon_tag.get("content")) # type: ignore

            return None, None

        except Exception as e:
            # Low-level warning only if verbose
            # logger.debug(f"Failed to fetch coordinates for {facility_url}: {e}")
            return None, None

    @staticmethod
    def _dms_to_decimal(dms_str: str) -> Optional[float]:
        """Convert DMS (degrees° minutes' seconds") string to decimal degrees."""
        if not dms_str: return None
        try:
            # Check if it's already a float string
            return float(dms_str)
        except ValueError:
            pass
            
        try:
            match = re.match(r"(-?\d+)°\s*(\d+)'?\s*(\d+(?:\.\d+)?)", dms_str)
            if match:
                deg, minutes, seconds = map(float, match.groups())
                sign = -1 if deg < 0 else 1
                return sign * (abs(deg) + minutes / 60 + seconds / 3600)
        except:
            return None
        return None

# ==================================================================================== #

def load_facilities(config: AppConfig) -> pd.DataFrame:
    """
    Helper function to fetch DNFSB facilities into a DataFrame.

    Args:
        config: Application configuration object.

    Returns:
        Pandas DataFrame with all facilities and coordinates.
    """
    use_local = config.nfc_facilities.use_cache
    verbose = config.verbose
    # Setup project directory
    project_dir = Project(config)
    if project_dir:
        output_dir = project_dir.raw_data / "_nfc_facilities" 
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / "dnfsb.tsv"
        if use_local and output_path.exists():
            logger.info(f"Using local DNFSB data from {output_path}")
            return pd.read_csv(output_path, sep='\t')
    else: output_path = None
    
    # Run Scraper with error handling
    try:
        db = DNFSBFacilityDB(output_path, verbose)
        df = db.run()
        return df
    except Exception as e:
        logger.error(f"DNFSB module critical failure: {e}")
        return pd.DataFrame()