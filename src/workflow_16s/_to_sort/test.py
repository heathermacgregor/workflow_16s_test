#!/usr/bin/env python3
"""
Nuclear Facilities & Uranium Mines Data Compiler (No API Keys Required)

This script compiles nuclear facilities and uranium mining sites data from 
publicly available web sources without requiring API keys.

Data sources:
- Wikipedia lists of nuclear facilities
- Public uranium mine databases
- World Nuclear Association public data
- OpenStreetMap/Nominatim for geocoding (free)

Requirements:
pip install requests pandas beautifulsoup4 lxml geopy folium openpyxl
"""

import requests
import pandas as pd
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import folium

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NuclearFacilitiesCompiler:
    def __init__(self):
        self.facilities_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize geocoder (free service)
        self.geolocator = Nominatim(user_agent="nuclear_facilities_compiler_v2")
        
    def geocode_location(self, location: str, country: str = None) -> Tuple[Optional[float], Optional[float]]:
        """Geocode a location to get latitude and longitude"""
        try:
            if country:
                full_location = f"{location}, {country}"
            else:
                full_location = location
                
            result = self.geolocator.geocode(full_location, timeout=10)
            if result:
                return result.latitude, result.longitude
            else:
                logger.warning(f"Could not geocode: {full_location}")
                return None, None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding error for {location}: {e}")
            time.sleep(2)  # Rate limiting
            return None, None
    
    def scrape_wikipedia_nuclear_plants(self) -> List[Dict]:
        """Scrape nuclear power plants from Wikipedia"""
        facilities = []
        
        try:
            # Wikipedia list of nuclear power stations
            url = "https://en.wikipedia.org/wiki/List_of_nuclear_power_stations"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find tables with nuclear plant data
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                rows = table.find_all('tr')
                headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
                
                for row in rows[1:]:
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    
                    if len(cells) >= 3:  # Ensure we have enough data
                        plant_name = cells[0] if cells[0] else "Unknown"
                        country = cells[1] if len(cells) > 1 else "Unknown"
                        
                        # Extract capacity if available
                        capacity = None
                        for cell in cells:
                            if 'MW' in cell:
                                capacity_match = re.search(r'(\d+(?:,\d+)?)\s*MW', cell.replace(',', ''))
                                if capacity_match:
                                    capacity = int(capacity_match.group(1).replace(',', ''))
                                    break
                        
                        # Geocode location
                        lat, lon = self.geocode_location(f"{plant_name} nuclear power plant", country)
                        
                        facility = {
                            'name': plant_name,
                            'type': 'Nuclear Power Plant',
                            'country': country,
                            'latitude': lat,
                            'longitude': lon,
                            'capacity_mw': capacity,
                            'status': 'Unknown',
                            'data_source': 'Wikipedia',
                            'last_updated': datetime.now().isoformat()
                        }
                        facilities.append(facility)
                        
                        # Rate limiting for geocoding
                        time.sleep(1)
            
            logger.info(f"Scraped {len(facilities)} nuclear plants from Wikipedia")
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia nuclear plants: {e}")
        
        return facilities
    
    def scrape_uranium_mines_data(self) -> List[Dict]:
        """Scrape uranium mining facilities from public sources"""
        mines = []
        
        try:
            # Wikipedia list of uranium mines
            url = "https://en.wikipedia.org/wiki/List_of_uranium_mines"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find tables and lists with uranium mine data
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # Skip header
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    
                    if len(cells) >= 2:
                        mine_name = cells[0] if cells[0] else "Unknown Mine"
                        location = cells[1] if len(cells) > 1 else "Unknown"
                        
                        # Try to extract country from location
                        country = "Unknown"
                        if len(cells) > 2 and cells[2]:
                            country = cells[2]
                        elif location and ',' in location:
                            country = location.split(',')[-1].strip()
                        
                        # Extract production data if available
                        production = None
                        for cell in cells:
                            if any(unit in cell.lower() for unit in ['tonnes', 't/year', 'tons']):
                                prod_match = re.search(r'(\d+(?:,\d+)?)', cell.replace(',', ''))
                                if prod_match:
                                    production = int(prod_match.group(1).replace(',', ''))
                                    break
                        
                        # Geocode location
                        lat, lon = self.geocode_location(f"{mine_name} uranium mine", country if country != "Unknown" else None)
                        
                        mine = {
                            'name': mine_name,
                            'type': 'Uranium Mine',
                            'country': country,
                            'location': location,
                            'latitude': lat,
                            'longitude': lon,
                            'production_tonnes_year': production,
                            'status': 'Unknown',
                            'data_source': 'Wikipedia',
                            'last_updated': datetime.now().isoformat()
                        }
                        mines.append(mine)
                        
                        # Rate limiting
                        time.sleep(1)
            
            logger.info(f"Scraped {len(mines)} uranium mines from Wikipedia")
            
        except Exception as e:
            logger.error(f"Error scraping uranium mines: {e}")
        
        return mines
    
    def get_world_nuclear_org_data(self) -> List[Dict]:
        """Scrape publicly available data from World Nuclear Association"""
        facilities = []
        
        try:
            # World Nuclear Association reactor database (public info)
            url = "https://www.world-nuclear.org/information-library/facts-and-figures/world-nuclear-power-reactors-and-uranium-requirements.aspx"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for tables with reactor data
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has data rows
                    for row in rows[1:]:
                        cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                        
                        # Look for country and reactor count data
                        if len(cells) >= 2 and cells[0] and not cells[0].startswith('Total'):
                            country = cells[0]
                            
                            # This is aggregate country data, so we create summary entries
                            facility = {
                                'name': f"{country} Nuclear Program",
                                'type': 'National Nuclear Program',
                                'country': country,
                                'latitude': None,  # Will need geocoding for country center
                                'longitude': None,
                                'data_source': 'World Nuclear Association',
                                'last_updated': datetime.now().isoformat()
                            }
                            
                            # Try to extract reactor count or capacity
                            for i, cell in enumerate(cells[1:], 1):
                                if cell.isdigit():
                                    if i == 1:
                                        facility['reactor_count'] = int(cell)
                                    elif 'MW' in cells[0] or any('MW' in c for c in cells):
                                        facility['total_capacity_mw'] = int(cell) if cell.isdigit() else None
                            
                            facilities.append(facility)
            
            logger.info(f"Collected {len(facilities)} entries from World Nuclear Association")
            
        except Exception as e:
            logger.error(f"Error fetching World Nuclear Association data: {e}")
        
        return facilities
    
    def add_major_facilities_manual(self) -> List[Dict]:
        """Add manually curated major nuclear facilities and uranium mines"""
        facilities = [
            # Major Nuclear Power Plants
            {
                'name': 'Kashiwazaki-Kariwa Nuclear Power Plant',
                'type': 'Nuclear Power Plant',
                'country': 'Japan',
                'latitude': 37.4206,
                'longitude': 138.5944,
                'capacity_mw': 8212,
                'status': 'Shut down (post-Fukushima)',
                'data_source': 'Manual Entry',
                'notes': 'Largest nuclear power plant by capacity'
            },
            {
                'name': 'Bruce Nuclear Generating Station',
                'type': 'Nuclear Power Plant',
                'country': 'Canada',
                'latitude': 44.3147,
                'longitude': -81.6014,
                'capacity_mw': 6430,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Largest operating nuclear facility'
            },
            {
                'name': 'Zaporizhzhia Nuclear Power Plant',
                'type': 'Nuclear Power Plant',
                'country': 'Ukraine',
                'latitude': 47.5147,
                'longitude': 34.5856,
                'capacity_mw': 6000,
                'status': 'Occupied/Disputed',
                'data_source': 'Manual Entry',
                'notes': 'Largest nuclear plant in Europe'
            },
            
            # Major Uranium Mines
            {
                'name': 'McArthur River Uranium Mine',
                'type': 'Uranium Mine',
                'country': 'Canada',
                'latitude': 57.2500,
                'longitude': -105.0000,
                'production_tonnes_year': 7000,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Highest grade uranium mine in the world'
            },
            {
                'name': 'Cigar Lake Uranium Mine',
                'type': 'Uranium Mine',
                'country': 'Canada',
                'latitude': 57.8333,
                'longitude': -104.5000,
                'production_tonnes_year': 6900,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Second highest grade uranium mine'
            },
            {
                'name': 'Olympic Dam Mine',
                'type': 'Uranium Mine',
                'country': 'Australia',
                'latitude': -30.4406,
                'longitude': 136.8864,
                'production_tonnes_year': 3500,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Multi-commodity mine (copper, uranium, gold, silver)'
            },
            {
                'name': 'Ranger Uranium Mine',
                'type': 'Uranium Mine',
                'country': 'Australia',
                'latitude': -12.6667,
                'longitude': 132.9000,
                'production_tonnes_year': 2500,
                'status': 'Closing',
                'data_source': 'Manual Entry',
                'notes': 'Located in Kakadu National Park'
            },
            {
                'name': 'Tortkuduk Mine',
                'type': 'Uranium Mine',
                'country': 'Kazakhstan',
                'latitude': 49.2000,
                'longitude': 60.9000,
                'production_tonnes_year': 3000,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'In-situ leaching operation'
            },
            {
                'name': 'Akokan Mine',
                'type': 'Uranium Mine',
                'country': 'Niger',
                'latitude': 18.7333,
                'longitude': 7.3833,
                'production_tonnes_year': 1500,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Major African uranium producer'
            },
            
            # Nuclear Fuel Processing Facilities
            {
                'name': 'Hanford Site',
                'type': 'Nuclear Fuel Processing/Waste Storage',
                'country': 'United States',
                'latitude': 46.5500,
                'longitude': -119.4500,
                'status': 'Cleanup/Decommissioning',
                'data_source': 'Manual Entry',
                'notes': 'Major plutonium production facility, now cleanup site'
            },
            {
                'name': 'La Hague Reprocessing Plant',
                'type': 'Nuclear Fuel Reprocessing',
                'country': 'France',
                'latitude': 49.6781,
                'longitude': -1.8761,
                'status': 'Operational',
                'data_source': 'Manual Entry',
                'notes': 'Major spent fuel reprocessing facility'
            },
            {
                'name': 'Sellafield',
                'type': 'Nuclear Fuel Processing/Reprocessing',
                'country': 'United Kingdom',
                'latitude': 54.4167,
                'longitude': -3.4833,
                'status': 'Decommissioning/Processing',
                'data_source': 'Manual Entry',
                'notes': 'Historic reprocessing and waste storage site'
            }
        ]
        
        # Add timestamps
        for facility in facilities:
            facility['last_updated'] = datetime.now().isoformat()
        
        return facilities
    
    def compile_all_data(self) -> pd.DataFrame:
        """Compile data from all sources"""
        logger.info("Starting nuclear facilities and uranium mines data compilation...")
        
        all_facilities = []
        
        # Scrape Wikipedia nuclear plants
        logger.info("Collecting nuclear power plants from Wikipedia...")
        wiki_plants = self.scrape_wikipedia_nuclear_plants()
        all_facilities.extend(wiki_plants)
        
        # Scrape uranium mines
        logger.info("Collecting uranium mines from Wikipedia...")
        uranium_mines = self.scrape_uranium_mines_data()
        all_facilities.extend(uranium_mines)
        
        # Get World Nuclear Association data
        logger.info("Collecting data from World Nuclear Association...")
        wna_data = self.get_world_nuclear_org_data()
        all_facilities.extend(wna_data)
        
        # Add manual entries
        logger.info("Adding manually curated major facilities...")
        manual_data = self.add_major_facilities_manual()
        all_facilities.extend(manual_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_facilities)
        
        if not df.empty:
            # Clean data
            df = df.drop_duplicates(subset=['name', 'country'], keep='first')
            df = df.sort_values(['country', 'name'])
            
            # Fill missing values
            df['type'] = df['type'].fillna('Unknown')
            df['country'] = df['country'].fillna('Unknown')
            df['status'] = df['status'].fillna('Unknown')
            
            logger.info(f"Compiled {len(df)} unique nuclear facilities and uranium mines")
        else:
            logger.warning("No facility data was collected")
        
        return df
    
    def export_data(self, df: pd.DataFrame, formats: List[str] = ['csv', 'json']):
        """Export compiled data to various formats"""
        if df.empty:
            logger.warning("No data to export")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'csv' in formats:
            csv_filename = f'nuclear_uranium_facilities_{timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            logger.info(f"Data exported to {csv_filename}")
        
        if 'json' in formats:
            json_filename = f'nuclear_uranium_facilities_{timestamp}.json'
            df.to_json(json_filename, orient='records', indent=2)
            logger.info(f"Data exported to {json_filename}")
        
        if 'excel' in formats:
            excel_filename = f'nuclear_uranium_facilities_{timestamp}.xlsx'
            df.to_excel(excel_filename, index=False)
            logger.info(f"Data exported to {excel_filename}")
    
    def create_interactive_map(self, df: pd.DataFrame):
        """Create interactive map of facilities"""
        if df.empty:
            logger.warning("No data for mapping")
            return
        
        # Filter for facilities with coordinates
        mapped_df = df.dropna(subset=['latitude', 'longitude'])
        
        if mapped_df.empty:
            logger.warning("No facilities with coordinates for mapping")
            return
        
        # Create map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Color coding by facility type
        type_colors = {
            'Nuclear Power Plant': 'red',
            'Uranium Mine': 'blue',
            'Nuclear Fuel Processing/Waste Storage': 'orange',
            'Nuclear Fuel Reprocessing': 'purple',
            'Nuclear Fuel Processing/Reprocessing': 'darkpurple',
            'National Nuclear Program': 'green'
        }
        
        for _, facility in mapped_df.iterrows():
            facility_type = facility.get('type', 'Unknown')
            color = type_colors.get(facility_type, 'gray')
            
            # Create popup
            popup_content = f"""
            <b>{facility['name']}</b><br>
            Type: {facility_type}<br>
            Country: {facility.get('country', 'Unknown')}<br>
            Status: {facility.get('status', 'Unknown')}<br>
            """
            
            if 'capacity_mw' in facility and pd.notna(facility['capacity_mw']):
                popup_content += f"Capacity: {facility['capacity_mw']} MW<br>"
            
            if 'production_tonnes_year' in facility and pd.notna(facility['production_tonnes_year']):
                popup_content += f"Production: {facility['production_tonnes_year']} tonnes/year<br>"
            
            popup_content += f"Source: {facility.get('data_source', 'Unknown')}"
            
            # Add marker
            folium.Marker(
                location=[facility['latitude'], facility['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=facility['name'],
                icon=folium.Icon(color=color, icon='industry', prefix='fa')
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;">
        <h4>Facility Types</h4>
        <i class="fa fa-industry" style="color:red"></i> Nuclear Power Plant<br>
        <i class="fa fa-industry" style="color:blue"></i> Uranium Mine<br>
        <i class="fa fa-industry" style="color:orange"></i> Fuel Processing/Waste<br>
        <i class="fa fa-industry" style="color:purple"></i> Fuel Reprocessing<br>
        <i class="fa fa-industry" style="color:green"></i> National Program<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        map_filename = f'nuclear_uranium_facilities_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        m.save(map_filename)
        logger.info(f"Interactive map saved as {map_filename}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print comprehensive summary"""
        if df.empty:
            print("No data collected")
            return
        
        print("\n" + "="*70)
        print("NUCLEAR FACILITIES & URANIUM MINES DATA SUMMARY")
        print("="*70)
        
        print(f"Total facilities: {len(df)}")
        print(f"Facilities with coordinates: {df.dropna(subset=['latitude', 'longitude']).shape[0]}")
        
        if 'type' in df.columns:
            print(f"\nFacility types:")
            type_counts = df['type'].value_counts()
            for ftype, count in type_counts.items():
                print(f"  {ftype}: {count}")
        
        if 'country' in df.columns:
            print(f"\nCountries represented: {df['country'].nunique()}")
            print(f"\nTop countries by facility count:")
            country_counts = df['country'].value_counts().head(10)
            for country, count in country_counts.items():
                print(f"  {country}: {count}")
        
        if 'status' in df.columns:
            print(f"\nFacilities by status:")
            status_counts = df['status'].value_counts()
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        print(f"\nData sources:")
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        print("="*70)

def main():
    """Main execution function"""
    print("Nuclear Facilities & Uranium Mines Data Compiler")
    print("===============================================")
    print("No API keys required - using public web sources")
    print()
    
    compiler = NuclearFacilitiesCompiler()
    
    # Compile all data
    facilities_df = compiler.compile_all_data()
    
    if not facilities_df.empty:
        # Show summary
        compiler.print_summary(facilities_df)
        
        # Export options
        print("\nExport Options:")
        export_formats = input("Choose export formats (csv,json,excel) [default: csv,json]: ").strip()
        if not export_formats:
            export_formats = 'csv,json'
        formats = [f.strip() for f in export_formats.split(',') if f.strip()]
        
        compiler.export_data(facilities_df, formats)
        
        # Map option
        create_map = input("Create interactive map? (y/n) [default: y]: ").strip().lower()
        if create_map != 'n':
            print("Creating interactive map...")
            compiler.create_interactive_map(facilities_df)
        
        print("\n✓ Data compilation completed successfully!")
        print("Files have been saved in the current directory.")
        
    else:
        print("❌ No facility data was collected.")
        print("This may be due to network issues or changes in source websites.")

if __name__ == "__main__":
    main()
