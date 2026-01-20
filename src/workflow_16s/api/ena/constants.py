# workflow_16s/api/ena/constants.py

# ===================================== CONSTANTS ==================================== #

ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
BIOSAMPLES_API_URL = "https://www.ebi.ac.uk/biosamples/samples/"

PATTERNS_TO_EXCLUDE = [
    'gut metagenome', 'plant metagenome', 'pig metagenome', 'fecal metagenome',
    'bat metagenome', 'algae metagenome', 'insect'
]