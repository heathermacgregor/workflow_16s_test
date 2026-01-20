# ==================================================================================== #

# Standard Imports
import logging
import time
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urljoin
import re

# Third Party Imports
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ==================================================================================== #

logger = logging.getLogger("workflow_16s")

# ==================================================================================== #

class DNFSBFacilityDB:
    """
    Class for downloading and processing DOE nuclear facilities from the DNFSB website.
    Extracts facility names, primary functions, status, state, and coordinates.
    """

    BASE_URL = "https://www.dnfsb.gov"
    SITE_LIST_URL = f"{BASE_URL}/doe-sites"

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.result: pd.DataFrame = pd.DataFrame()

    # -------------------------- Public Methods -------------------------- #
    def run(self) -> pd.DataFrame:
        """Scrape the facility list and fetch coordinates."""
        facilities = self._fetch_facility_list()
        logger.info(f"Found {len(facilities)} facilities. Fetching coordinates...")

        for f in facilities:
            lat, lon = self._fetch_facility_coordinates(f.get("link"))
            f["latitude"] = lat
            f["longitude"] = lon
            time.sleep(0.5)  # be polite

        self.result = pd.DataFrame(facilities)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.output_dir / "dnfsb_facilities_geolocated.csv"
            self.result.to_csv(csv_path, index=False)
            logger.info(f"Saved results to {csv_path}")

        return self.result

    # -------------------------- Internal Methods -------------------------- #
    def _fetch_facility_list(self) -> List[dict]:
        """Scrape the main DOE sites table and extract facility info + links."""
        resp = requests.get(self.SITE_LIST_URL)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        facilities = []

        if table:
            rows = table.find_all("tr")[1:]  # skip header
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    link_tag = cols[0].find("a")
                    name = link_tag.text.strip() if link_tag else cols[0].get_text(strip=True)
                    href = urljoin(self.BASE_URL, link_tag["href"]) if link_tag else None

                    facilities.append({
                        "facility": name,
                        "facility_description": cols[1].get_text(strip=True),
                        "state_or_province": cols[2].get_text(strip=True),
                        "country": "United States of America",
                        "status": cols[3].get_text(strip=True),
                        "link": href,
                        "data_source": "DNFSB"
                    })
        return facilities

    def _fetch_facility_coordinates(self, facility_url: Optional[str]) -> tuple:
        """Fetch latitude and longitude from a facility detail page."""
        if not facility_url:
            return None, None

        try:
            resp = requests.get(facility_url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            geo_div = soup.find("div", class_="block-field-blocknodedoe-sitefield-site-geolocation")
            if not geo_div:
                return None, None

            lat_tag = geo_div.find("meta", {"property": "latitude"})
            lon_tag = geo_div.find("meta", {"property": "longitude"})
            if not lat_tag or not lon_tag:
                return None, None

            return self._dms_to_decimal(lat_tag["content"]), self._dms_to_decimal(lon_tag["content"])

        except Exception as e:
            logger.warning(f"Failed to fetch coordinates for {facility_url}: {e}")
            return None, None

    @staticmethod
    def _dms_to_decimal(dms_str: str) -> Optional[float]:
        """Convert DMS (degrees° minutes' seconds") string to decimal degrees."""
        match = re.match(r"(-?\d+)°\s*(\d+)'?\s*(\d+(?:\.\d+)?)", dms_str)
        if not match:
            return None
        deg, minutes, seconds = map(float, match.groups())
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minutes / 60 + seconds / 3600)

# ==================================================================================== #

def load_dnfsb_facilities(output_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Helper function to fetch DNFSB facilities into a DataFrame.

    Args:
        output_dir: Optional path to save CSV output.

    Returns:
        Pandas DataFrame with all facilities and coordinates.
    """
    db = DNFSBFacilityDB(output_dir=output_dir)
    df = db.run()
    logger.info(f"Loaded {df.shape[0]} DNFSB facilities")
    return df
