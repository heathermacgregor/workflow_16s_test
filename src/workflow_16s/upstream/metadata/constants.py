# ===================================== IMPORTS ====================================== #

# Standard Imports
import re
from typing import Callable, Dict, Tuple
import pandas as pd

# ============================== REGEX PATTERNS ================================== #

# Pre-compiled regex for efficiency
NUM_PATTERN = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
PH_PATTERN = re.compile(r'^ph[^a-zA-Z]|^ph$')

# =========================== COORDINATE DEFINITIONS ============================= #

DEFAULT_COORDINATE_SOURCES = {
    'lat': [
        'lat_study', 'lat_ena', 'lat.1', 'lat', 'biosample_geographic_location_(latitude)',
        'biosample_latitude', 'experiment_lat', 'run_lat', 'latitude'
    ],
    'lon': [
        'lon_study', 'lon.1', 'lon', 'biosample_geographic_location_(longitude)',
        'biosample_longitude', 'experiment_lon', 'run_lon', 'longitude'
    ],
    'pairs': [
        'location_ena', 'location_start', 'location_end', 'location_start_study',
        'location_end_study', 'lat_lon', 'location', 'biosample_lat_lon',
        'biosample_latitude_and_longitude', 'run_location', 'run_location_start',
        'run_location_end', 'experiment_location', 'experiment_location_start',
        'experiment_location_end'
    ]
}

# ============================ COLUMN/UNIT DEFINITIONS =========================== #

DEFAULT_COLUMN_MAPPINGS = {
    'env_biome': 'environment_biome', 'env_feature': 'environment_feature',
    'env_material': 'environment_material'
}

DEFAULT_UNIT_PATTERNS = {
    'celsius': re.compile(r'_(?:celsius|cel|c)$', re.IGNORECASE),
    'fahrenheit': re.compile(r'_(?:fahrenheit|far|f)$', re.IGNORECASE),
    'kelvin': re.compile(r'_(?:kelvin|k)$', re.IGNORECASE),
    'meters': re.compile(r'_(?:meters|meter|m)$', re.IGNORECASE),
    'feet': re.compile(r'_(?:feet|ft)$', re.IGNORECASE)
}

DEFAULT_CONVERSIONS: Dict[str, Tuple[str, Callable[[pd.Series], pd.Series]]] = {
    'fahrenheit': ('celsius', lambda f: (pd.to_numeric(f, errors='coerce') - 32) * 5 / 9),
    'kelvin': ('celsius', lambda k: pd.to_numeric(k, errors='coerce') - 273.15),
    'feet': ('meters', lambda ft: pd.to_numeric(ft, errors='coerce') * 0.3048),
}

DEFAULT_MEASUREMENT_STANDARDS = {
    'temp': 'celsius', 'depth': 'meters', 'altitude': 'meters'
}

# ============================= ONTOLOGY DEFINITIONS ============================= #

ONTOLOGY_MAP = {
    'empo_1': {
        'Host-associated': [
            'host', 'symbiont', 'microbiome', 'human', 'animal'
        ],
        'Free-living': [
            'free living', 'environmental', 'soil', 'water', 'sediment', 'air'
        ]
    },
    'empo_2': {
        'Animal': ['animal', 'human', 'insect', 'mammal', 'gut', 'feces', 'skin'],
        'Plant': ['plant', 'rhizosphere', 'root', 'leaf', 'flower'],
        'Fungus': ['fungus', 'fungal'],
        'Aquatic': ['aquatic', 'water', 'marine', 'freshwater', 'sediment', 'ocean'],
        'Terrestrial': ['terrestrial', 'soil', 'land', 'desert', 'forest']
    },
    'empo_3': {
        'Gut': ['gut', 'feces', 'fecal', 'intestinal'],
        'Soil': ['soil', 'rhizosphere', 'terrestrial'],
        'Water': ['water', 'aquatic', 'marine', 'freshwater'],
        'Sediment': ['sediment'], 'Skin': ['skin']
    },
    'env_biome': {
        'Urban': ['urban', 'city'],
        'Agricultural': ['agricultural', 'farm', 'crop'],
        'Forest': ['forest'],
        'Grassland': ['grassland', 'savanna'],
        'Aquatic': ['aquatic', 'marine', 'freshwater', 'lake', 'river', 'ocean']
    },
    'env_feature': {
        'Anthropogenic': ['anthropogenic', 'human-made', 'built environment'],
        'Natural': ['natural', 'wild']
    },
    'env_material': {
        'Soil': ['soil', 'loam', 'clay', 'silt'],
        'Water': ['water'],
        'Sediment': ['sediment', 'mud'],
        'Air': ['air']
    }
}

# workflow_16s/upstream/metadata/constants.py

"""
Stores static constant definitions for metadata partitioning.
"""

# Keywords used to filter out datasets that are likely host-associated
# and not relevant to the environmental focus.
exclusion_keywords = [
    # --- Host Organisms (Human & Animal) ---
    "human", "patient", "clinical",      # Human-specific
    "mouse", "mice", "murine",           # Mouse models
    "rat", "rattus",                     # Rat models
    "bovine", "cattle", "cow",           # Bovine hosts
    "porcine", "pig", "swine",           # Pig hosts
    "avian", "chicken", "poultry",       # Bird hosts
    "ovine", "sheep",                    # Sheep hosts
    "canine", "dog",                     # Dog hosts
    "feline", "cat",                     # Cat hosts
    "equine", "horse",                   # Horse hosts
    "primate", "monkey", "ape",          # Non-human primates
    "animal model", "host", "host-associated", # General host terms

    # --- Body Parts, Tissues & Organs ---
    "gut", "gastrointestinal", "intestinal",   # Digestive system
    "oral", "mouth", "dental", "plaque",       # Oral cavity
    "organ", "tissue", "biopsy",               # General tissues
    "skin", "dermal", "cutaneous",             # Skin
    "lung", "pulmonary", "respiratory",        # Respiratory system
    "vaginal", "urogenital",                   # Urogenital tract
    "nasal", "nasopharyngeal",                 # Nasal cavity
    "brain", "neural",                         # Nervous system
    "liver", "hepatic",                        # Liver
    "kidney", "renal",                         # Kidney
    
    # --- Bodily Fluids & Excreta ---
    "feces", "fecal", "stool", "scat",         # Excrement
    "blood", "serum", "plasma",                # Blood products
    "saliva", "sputum",                        # Oral/respiratory fluids
    "urine", "urinary",                        # Urine
    "milk", "mammary",                         # Milk
    "mucus", "mucosal",                        # Mucus
    "semen", "seminal",                        # Seminal fluid
    "bile",                                    # Bile

    # --- Health, Disease & Clinical States ---
    "disease", "disorder", "syndrome",         # General disease
    "infection", "infectious", "pathogen",     # Infection
    "immune", "immunity", "immunological",     # Immune system
    "inflammation", "inflammatory",            # Inflammatory response
    "lesion", "wound", "abscess",              # Tissue damage
    "cancer", "tumor", "carcinoma",            # Oncology
    "health", "healthy", "control",            # Health states
    "treatment", "therapy", "antibiotic",      # Medical intervention
    "probiotic", "prebiotic",                  # Supplements
    
    # --- Microbiome-Specific (often host-related) ---
    "vaginal microbiome", "microbiota", "dysbiosis",   # Microbiome terms
    "gut microbiome", "oral microbiome",                 # Specific microbiomes
    "holobiont"                                        # Host + microbes
]