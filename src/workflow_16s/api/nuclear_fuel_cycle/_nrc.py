# ==================================================================================== #
#                       api/nuclear_fuel_cycle/_nrc.py
# ==================================================================================== #

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger("workflow_16s")

class NRC:
    """Scrapes official US Nuclear Regulatory Commission (NRC) facility lists."""
    
    URLS = {
        "reactors": "https://www.nrc.gov/reactors/operating/list-power-reactor-units.html",
        "fuel_cycle": "https://www.nrc.gov/info-finder/fc/index.html"
    }

    def load(self) -> pd.DataFrame:
        logger.info("Scraping US NRC facility lists...")
        dfs = []
        
        # 1. Power Reactors
        try:
            tables = pd.read_html(self.URLS["reactors"])
            df_react = tables[0].copy()
            df_react = df_react.rename(columns={
                "Plant Name": "facility",
                "Location": "location_desc", 
                "Reactor Type": "facility_type"
            })
            df_react['facility_status'] = 'Operating'
            df_react['facility_type'] = "Nuclear Power Plant (" + df_react['facility_type'] + ")"
            dfs.append(df_react)
        except Exception as e:
            logger.error(f"Failed to load NRC Reactors: {e}")

        # 2. Fuel Cycle Facilities
        try:
            tables = pd.read_html(self.URLS["fuel_cycle"])
            # The page often has multiple tables; the main one usually lists licensees
            for t in tables:
                if "Location" in t.columns and "Licensee" in t.columns:
                    df_fc = t.copy()
                    df_fc = df_fc.rename(columns={"Licensee": "facility", "Location": "location_desc"})
                    df_fc['facility_type'] = "Fuel Cycle Facility"
                    df_fc['facility_status'] = 'Operating'
                    dfs.append(df_fc)
                    break
        except Exception as e:
            logger.error(f"Failed to load NRC Fuel Cycle facilities: {e}")

        if not dfs: return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        df['country'] = "United States of America"
        df['data_source'] = "US NRC"
        
        # Note: NRC provides "Location" as text (e.g., "6 miles W of Russellville, AR").
        # These will be geocoded by your existing `_geocode` logic in _nfcis.py
        return df