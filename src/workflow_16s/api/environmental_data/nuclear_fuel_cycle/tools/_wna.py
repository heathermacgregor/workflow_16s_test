# workflow_16s/api/environmental_data/nuclear_fuel_cycle/tools/_wna.py

import pandas as pd

from workflow_16s.utils.logger import get_logger, with_logger


@with_logger
class WNA:
    """Scrapes World Nuclear Association uranium mining data."""
    logger = get_logger("workflow_16s")
    # Stable page for world uranium mining production
    URL = "https://world-nuclear.org/information-library/nuclear-fuel-cycle/mining-of-uranium/world-uranium-mining-production"

    def load(self) -> pd.DataFrame:
        self.logger.info("Scraping WNA Uranium Mining data...")
        try:
            # WNA pages usually contain a table "Uranium production by company/mine"
            tables = pd.read_html(self.URL)
            
            # Look for the table that contains "Mine" and "Country"
            target_df = pd.DataFrame()
            for t in tables:
                if "Mine" in t.columns and "Country" in t.columns:
                    target_df = t
                    break
            
            if target_df.empty:
                self.logger.warning("Could not find mining table in WNA page.")
                return pd.DataFrame()

            # Clean up
            df = target_df.rename(columns={
                "Mine": "facility",
                "Country": "country",
                "Type": "facility_type",
                "Main owner": "owner"
            })
            
            df['facility_type'] = "Uranium Mine (" + df['facility_type'].astype(str) + ")"
            df['data_source'] = "WNA"
            
            # WNA lists production (tonnes U), good proxy for size/capacity
            if "Production (tonnes U)" in df.columns:
                 df['facility_capacity'] = df["Production (tonnes U)"]

            return df

        except Exception as e:
            self.logger.error(f"WNA Scraper failed: {e}")
            return pd.DataFrame()