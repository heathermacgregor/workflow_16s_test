# workflow_16s/api/ena/constants.py

ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
BIOSAMPLES_API_URL = "https://www.ebi.ac.uk/biosamples/samples/"

HOST_KEYWORDS = [
    'homo sapiens', 
    'mus musculus', 
    'rattus norvegicus', 
    'metazoa', 
    'neotoma'
]
PATTERNS_TO_EXCLUDE = [
    'gut metagenome', 'plant metagenome', 'pig metagenome', 'fecal metagenome',
    'bat metagenome', 'algae metagenome', 'insect'
]# workflow_16s/api/ena/metadata/constants.py

# ===================================== CONSTANTS ==================================== #

from pathlib import Path

# API Endpoints
ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
BIOSAMPLES_API_URL = "https://www.ebi.ac.uk/biosamples/samples/"

# Defaults
DEFAULT_EMAIL = "user@example.com"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ena_metadata_cache" # Specific name