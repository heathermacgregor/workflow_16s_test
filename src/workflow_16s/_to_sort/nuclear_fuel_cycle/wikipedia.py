# ===================================== IMPORTS ====================================== #

# Standard Imports
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Third-Party Imports
import pandas as pd
from bs4 import BeautifulSoup

# Local Imports
from workflow_16s.constants import REFERENCES_DIR, USER_AGENT 

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ====================================== CLASSES ===================================== #

class WikipediaScraper:
    """A web scraper for extracting nuclear facility information from Wikipedia tables.
    
    - Scrapes nuclear power stations and uranium mines data from Wikipedia
    - Combinines scraped data as a pandas DataFrame
    - Saves DataFrame to a TSV file
    
    Attributes:
        BaseURL:     Base URL for Wikipedia pages.
        output_path: Output file path.
        session:     HTTP session for making requests.
        data:        Combined dataset of scraped facilities.
    """
    BaseURL = "https://en.wikipedia.org/wiki/"
    def __init__(self, output_path: Union[str, Path] = REFERENCES_DIR):
        self.output_path = output_path
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.data = pd.DataFrame()
      
    def _get_soup(self, url):
        """Retrieve and parse HTML content from a URL."""
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')

    def _get_wikitables(self, soup):
        """Extract all wikitable elements from parsed HTML."""
        return soup.find_all('table', {'class': 'wikitable'})

    def _nuclear_power_stations(self, url: str = None):
        """Scrape nuclear power station data from Wikipedia."""
        if url is None:
            url = f"{self.BaseURL}List_of_nuclear_power_stations"
        soup = self._get_soup(url)
        tables = self._get_wikitables(soup)
    
        dfs = []
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
    
            data = []
            for row in rows[1:]:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                station_name = cells[0] if cells[0] else "Unknown Station"
                row_data = {
                    'facility': station_name,
                    'data_source': "Wikipedia",
                    'wikipedia': url,
                    'wikitable': f"Table {i+1}",
                    'last_updated': datetime.now().isoformat()
                }
                for n in range(1, len(headers)-1):
                    row_data[headers[n]] = cells[n] 
                data.append(row_data)
            dfs.append(pd.DataFrame(data))
        df = pd.concat(dfs)
        rename = {
            "# units[note 1]": "n_units_1",
            "Capacity(MWe)[note 2]": "capacity_mwe",
            "Country or territory": "country_1",
            "Location": "lat_lon",
            "Began operation": "facility_start_year",
            "No. ofunits": "n_units_2",
            "Net capacityunder construction(MW)": "capacity_mw_during_construction",
            "Constructionstart": "facility_start_construction_year",
            "Plannedconnection": "facility_planned_connection_year",
            "Country": "country_2",
            "Past capacity (MW)": "capacity_mw_past",
        }
        df = df.rename(columns=rename)
        try:
            # 69°42′35″N 170°18′22″E / 69.7097°N 170.3061°E / 69.7097; 170.3061 (Akademik Lomonosov)
            df['latitude'] = [str(i).strip().split('/')[2].split('(')[0].split(';')[0] for i in df['lat_lon']]
            df['longitude'] = [str(i).strip().split('/')[2].split('(')[0].split(';')[1] for i in df['lat_lon']]
            logger.info(df[['lat_lon', 'latitude', 'longitude']])
            df = df.drop('lat_lon', axis=1)
        except Exception as e:
            logger.error(str(e))
            
        better_facility = []
        for i in df['facility']:
            try:
                facility = i.split('/')[-1].split('(')[-1].split(')')[0]
                if facility:
                    better_facility.append(facility)
                else:
                    better_facility.append(i)
            except Exception as e:
                logger.info(i)
                better_facility.append(i)
        df['facility'] = better_facility
        df['country'] = df['country_1'].combine_first(df['country_2'])
        df['n_units'] = df['n_units_1'].combine_first(df['n_units_2'])
        return df

    def _uranium_mines(self, url: str = None):
        """Scrape uranium mine data from Wikipedia."""
        if url is None:
            url = f"{self.BaseURL}List_of_uranium_mines"
        soup = self._get_soup(url)
        tables = self._get_wikitables(soup)

        dfs = []
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            headers = [th.get_text().strip() for th in rows[1].find_all(['th', 'td'])]
            data = []
            for row in rows[2:]: # Skip header
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if len(cells) >=2:
                    mine_name = cells[0] if cells[0] else "Unknown Mine"
                    if mine_name == 'Mine':
                        continue
                    location = cells[1] if len(cells) > 1 else "Unknown"
                        
                    # Try to extract country from location
                    country = "Unknown"
                    if len(cells) > 2 and cells[2]:
                        country = cells[2]
                    elif location and ',' in location:
                        country = location.split(',')[-1].strip()

                row_data = {
                    'facility': mine_name,
                    'location': location,
                    'country': country,
                    'data_source': "Wikipedia",
                    'wikipedia': url,
                    'wikitable': f"Table {i+1}",
                    'last_updated': datetime.now().isoformat()
                }
                if len(cells) >= 3:
                    for n in range(3, len(cells)-1):
                        row_data[headers[n]] = cells[n] 
                data.append(row_data)
            dfs.append(pd.DataFrame(data))
        df = pd.concat(dfs)
        rename = {
            "Year discovered": "facility_discovered_year",	
            "Year commenced": "facility_start_year",	
            "Grade %U[2]": "grade_%_u_2",	
            "Grade %U": "grade_%_u",
            "Annual production (tOre)[3]": "annual_production_tons_ore",	
            "Annual production (tU)[4]": "annual_production_tons_u",	
            "Scheduled Commencement": "scheduled_commencement",	
            "Planned Annual production (tOre)": "planned_annual_production_tons_ore",	
            "Planned Annual production (tU)": "planned_annual_production_tons_u",	
            "Probable commencement": "probable_facility_start_year",	
            "Probable annual production (tOre)": "probable_annual_production_tons_ore",	
            "Probable annual production (tU)": "probable_annual_production_tons_u",	
            "Total production (tU)": "total_production_tons_u",	
            "Year closed": "facility_end_year",
        }
        df = df.rename(columns=rename)
        return df

    def _compile_and_sort(self):
        """Combine and sort all scraped facility data."""
        nps = self._nuclear_power_stations()
        um = self._uranium_mines()
        datasets = [nps, um]
        df = pd.concat(datasets)
        df = df.sort_values(by='facility')
        df.to_csv(self.output_path, sep="\t", index=False)
        self.data = df
        return df

# ======================================= API ======================================== #

def world_nfc_facilities(config: Dict, output_dir: Union[str, Path] = REFERENCES_DIR):
    """Public API function to retrieve worldwide nuclear facility data from Wikipedia.
    
    Returns:
        DataFrame containing combined nuclear facility information from Wikipedia.
    """
    output_path = Path(output_dir) / "wikipedia.tsv"
    if config.get("nfc_facilities", {}).get("use_local", False):
        if output_path.exists():
            return pd.read_csv(output_path, sep='\t')
    scraper = WikipediaScraper(output_path)
    return scraper._compile_and_sort()
