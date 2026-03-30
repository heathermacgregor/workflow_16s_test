# workflow_16s/api/environmental_data/nuclear_fuel_cycle/constants.py

SOURCE_PRIORITY = {
    "NFCIS": 1, "NRC": 2, "JRC": 3, "WNA": 4, 
    "DNFSB": 5, "GEM": 6, "WIKIDATA": 7, "OSM": 8,  # 🚀 New: OSM is secondary to curated expert databases
    "MINDAT": 9, "WIKIDATA_ANALOG": 10
}

MINDAT_COLUMNS_TO_KEEP = [
    "facility", "country", "lat", "lon", "elements",
    "refs", "wikipedia", "data_source"
]
mindat_columns_to_keep = [
    "facility", "country", "lat", "lon", "elements",
    "refs", "wikipedia", "data_source"
]

TYPE_DEFINITIONS = {
            'Nuclear Power Plant': [
                r'(?i)pressuri[sz]ed|PWR', r'(?i)boiling|BWR', r'(?i)candu|heavy|PHWR', 
                r'(?i)reactor', r'(?i)npp', r'(?i)power plant', r'(?i)generating station'
            ],
            'Uranium Mine': [
                r'(?i)uranium.*mine', r'(?i)open.*pit', r'(?i)in-situ', r'(?i)isr', 
                r'(?i)deposit', r'(?i)prospect', r'(?i)project' # Common in Mindat names
            ],
            'Uranium Mill': [r'(?i)mill'],
            'Salt Mine': [r'(?i)salt'],
            'Potash Mine': [r'(?i)potash'],
            'Fertilizer Plant': [r'(?i)fertilizer|nitrate|ammonia'],
            'Acid Plant': [r'(?i)acid', r'(?i)sulfuric'],
            'Coal Power Plant': [r'(?i)coal'],
            'Gold Mine': [r'(?i)gold'],
            'Copper Mine': [r'(?i)copper'],
            'Smelter': [r'(?i)aluminium|smelter'],
            'Desalination Plant': [r'(?i)desalination'],
            'Geothermal Plant': [r'(?i)geothermal'],
            'Waste Storage': [r'(?i)storage', r'(?i)repository', r'(?i)disposal'],
            'Superfund Site': [r'(?i)superfund']
        }
TYPE_DEFINITIONS.update({
    'Landfill': [r'(?i)landfill', r'(?i)waste.*dump', r'(?i)refuse'],
    'Wastewater': [r'(?i)sewage', r'(?i)wastewater', r'(?i)water.*reclamation'],
    'Mine_General': [r'(?i)quarry', r'(?i)pit', r'(?i)extraction'],
    'Research_Lab': [r'(?i)laboratory', r'(?i)research.*center', r'(?i)institute']
})
# 2. Source Map: Default types based on the database origin
#    This catches "Rossing" (from MINDAT) even if the name lacks "Mine"
SOURCE_DEFAULTS = {
            'MINDAT': 'Uranium Mine',
            'WNA': 'Nuclear Power Plant', # Mostly NPPs
            'PRIS': 'Nuclear Power Plant'
        }