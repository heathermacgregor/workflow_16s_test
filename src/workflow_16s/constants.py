from pathlib import Path

# ==================================================================================== #
# GENERAL CONSTANTS
# ==================================================================================== #
DEFAULT_EMAIL = "macgregor@berkeley.edu"
# Default config file path
DEFAULT_CONFIG = "/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml"
# Default output directory
OUTPUT_DIR = Path("/usr2/people/macgregor/amplicon/test/data/nfc")
# Default data directory
DATA_DIR = Path("/usr2/people/macgregor/amplicon/test/data/")
# Default resources directory
RESOURCES_DIR = Path("/usr2/people/macgregor/amplicon/workflow_16s/resources")
# Default mode ('genus' or 'ASV')
DEFAULT_MODE = 'genus'
# Default feature type ('genus' or 'ASV')
DEFAULT_FEATURE_TYPE = 'ASV'


# ==================================================================================== #
# 16S WORKFLOW CONSTANTS
# ==================================================================================== #

SAMPLE_ID_COLUMN = '#sampleid'
DATASET_COLUMN = 'dataset_name'

SET_SAMPLE_ID_COLUMN = 'sample_id'
SET_DATASET_ID_COLUMN = 'dataset_id'

DEFAULT_META_ID_COLUMN = '#sampleid'
DEFAULT_DATASET_COLUMN = 'dataset_name'

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_GROUP_COLUMNS = [
    {
        'name': "nuclear_contamination_status",
        'type': "bool",
        'values': [True, False]
    },
]
DEFAULT_PSEUDOCOUNT = 1.0

# ==================================================================================== #
# PROGRESS BARS
# ==================================================================================== #

# Width of progress bar description
DEFAULT_N: int = 65 


# ==================================================================================== #
# NUCLEAR FACILITIES DATA
# ==================================================================================== #

# GEM
# ------------------------------------------------------------------------------------ #
# Path to default GEM data file
DEFAULT_GEM_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/resources/nfc_facilities_data/gem_nuclearpower_2024-07.tsv'

# NFCIS
# ------------------------------------------------------------------------------------ #
# Path to default NFCIS data file
DEFAULT_NFCIS_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/resources/nfc_facilities_data/NFCISFacilityList.xlsx'


# ==================================================================================== #
# APIS
# ==================================================================================== #

# Default user agent for web requests
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# NCBI ENTREZ
# ------------------------------------------------------------------------------------ #
# Base URL for NCBI Entrez API
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# SILVA
# ------------------------------------------------------------------------------------ #
# Base URL for SILVA database
SILVA_BASE_URL = "https://www.arb-silva.de/fileadmin/silva_databases/current/Exports/"
# Base URL for SILVA database portal
SILVA_PORTAL_URL = "https://www.arb-silva.de/documentation/release-138/"

# GTDB
# ------------------------------------------------------------------------------------ #
# Base URL for GTDB database
GTDB_BASE_URL = "https://data.gtdb.ecogenomic.org/releases/release202/202.0/auxillary_files/"
# Base URL for GTDB database portal
GTDB_PORTAL_URL = "https://gtdb.ecogenomic.org/"   

# QIITA
# ------------------------------------------------------------------------------------ #
# Base URL for Qiita API
QIITA_BASE_URL = "https://api.qiita.ucsd.edu/"
# Base URL for Qiita portal
QIITA_PORTAL_URL = "https://qiita.ucsd.edu/study/"

# EBA ENA
# ------------------------------------------------------------------------------------ #
# Base URL for EBI ENA API
EBI_ENA_BASE_URL = "https://www.ebi.ac.uk/ena/portal/api/"
# Base URL for EBI ENA portal
EBI_ENA_PORTAL_URL = "https://www.ebi.ac.uk/ena/browser/view/"
# Base URL for ENA SRA portal
ENA_SRA_PORTAL_URL = "https://www.ebi.ac.uk/ena/browser/home"

# NCBI
# ------------------------------------------------------------------------------------ #
# Base URL for NCBI SRA portal
NCBI_SRA_PORTAL_URL = "https://www.ncbi.nlm.nih.gov/sra/"
# Base URL for NCBI Taxonomy database
NCBI_TAXONOMY_URL = "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id="
# Base URL for NCBI BioSample database
NCBI_BIOSAMPLE_URL = "https://www.ncbi.nlm.nih.gov/biosample/"
# Base URL for NCBI Assembly database
NCBI_ASSEMBLY_URL = "https://www.ncbi.nlm.nih.gov/assembly/"
# Base URL for NCBI Genome database
NCBI_GENOME_URL = "https://www.ncbi.nlm.nih.gov/genome/"

# PUBMED
# ------------------------------------------------------------------------------------ #
# Base URL for PubMed database
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/"

# MG-RAST
# ------------------------------------------------------------------------------------ #
# Base URL for MG-RAST API
MGRAST_BASE_URL = "https://api.mg-rast.org/"
# Base URL for MG-RAST portal
MGRAST_PORTAL_URL = "https://www.mg-rast.org/mgmain.html?mgpage=search&search="

# DDBJ SRA
# ------------------------------------------------------------------------------------ #
# Base URL for DDBJ SRA portal
DDBJ_SRA_PORTAL_URL = "https://ddbj.nig.ac.jp/search/sra/"

# GOOGLE EARTH ENGINE
# ------------------------------------------------------------------------------------ #
# Path to local copy of GEE catalog file
GEE_CATALOG_FILE_PATH = Path('/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google/resources/catalog.json')
# URL to download GEE catalog file from Zenodo
GEE_CATALOG_FILE_ZENODO_URL = "https://zenodo.org/records/17155067/files/catalog.json?download=1"


# OPEN-METEO
# ------------------------------------------------------------------------------------ #
# Base URL for Open-Meteo Air Quality API
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
# Base URL for Open-Meteo Weather Forecast API
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# INATURALIST
# ------------------------------------------------------------------------------------ #
# Base URL for iNaturalist API
INATURALIST_API_URL = "https://api.inaturalist.org/v1/observations"

# NOAA
# ------------------------------------------------------------------------------------ #
# Base URL for NOAA Tides and Currents API
NOAA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
# Base URL for NOAA Meteorological Data API
NOAA_STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

# NREL
# ------------------------------------------------------------------------------------ #
# Base URL for NREL Solar API (5-minute data)
NREL_SOLAR_API_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
# Base URL for NREL Solar Batch API (yearly data)
NREL_SOLAR_BATCH_API_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-batch-download.json"

# NWS
# ------------------------------------------------------------------------------------ #
# Base URL for NWS National Weather Service API
NWS_API_URL = "https://api.weather.gov"

# SOILGRIDS
# ------------------------------------------------------------------------------------ #
# Base URL for SoilGrids API
SOILGRIDS_API_URL = "https://rest.isric.org/soilgrids/v 2.0/properties/query"

# USGS
# ------------------------------------------------------------------------------------ #
# Base URL for USGS Earthquake API
USGS_EARTHQUAKE_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
# Base URL for USGS Water Services API
USGS_WATER_SERVICES_API_URL = "https://waterservices.usgs.gov/nwis"


# ==================================================================================== #
# TAXONOMY
# ==================================================================================== #

# Mapping of taxonomic levels to their corresponding indices in a taxonomy string
# (0-based index where 0=kingdom, 1=phylum, etc.)
TAXONOMIC_LEVELS = {'phylum': 2, 'class': 3, 'order': 4, 'family': 5, 'genus': 6}
# List of prefixes used in taxonomy strings
TAXONOMY_PREFIXES = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']


# ==================================================================================== #
# FILTERING CONSTANTS
# ==================================================================================== #

# Minimum relative abundance (%) threshold for filtering features
# Relative to 1 (100% = 1, 10% = 0.1, etc.)
MIN_REL_ABUNDANCE: float = 1
# Minimum number of samples a feature must be present in to be retained
MIN_SAMPLES: int = 10
# Minimum total counts a feature must have to be retained
MIN_COUNTS: int = 1000
# Pseudocount to add to zero counts to avoid issues with log transformations
PSEUDOCOUNT: float = 1e-5
# Thresholds for filtering features based on abundance and prevalence
# Threshold for group-level filtering
GROUP_THRESHOLD: float = 0.05
# Threshold for overall prevalence filtering
PREVALENCE_THRESHOLD: float = 0.05


# ==================================================================================== #
# ALPHA DIVERSITY 
# ==================================================================================== #

DEFAULT_ALPHA_METRICS = [
    'ace', 'chao1', 'dominance', 'gini_index', 'goods_coverage', 'heip_evenness', 
    'observed_features', 'pielou_evenness', 'shannon', 'simpson'        
]
PHYLO_METRICS = ['faith_pd', 'pd_whole_tree']
DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0


# ==================================================================================== #
# BETA DIVERSITY
# ==================================================================================== #

# Default beta diversity metric
DEFAULT_METRIC = 'braycurtis'
# Default number of dimensions for PCA
DEFAULT_N_PCA = 20
# Default number of dimensions for PCoA
DEFAULT_N_PCOA = None
# Default number of dimensions for t-SNE
DEFAULT_N_TSNE = 3
# Default number of dimensions for UMAP
DEFAULT_N_UMAP = 3
# Default number of dimensions for MDS
DEFAULT_N_MDS = 20
# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 0
# Default number of CPU cores to use for parallel processing
DEFAULT_CPU_LIMIT = 4


# ==================================================================================== #
# FIGURES
# ==================================================================================== #

DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1100

DEFAULT_COLOR_COL = 'dataset_name'
DEFAULT_SYMBOL_COL = DEFAULT_GROUP_COLUMN

#   MAPS
# ------------------------------------------------------------------------------------ #
# Default projection for geographic maps
DEFAULT_PROJECTION = 'natural earth'
# Default latitude column name
DEFAULT_LATITUDE_COL = 'latitude_deg'
# Default longitude column name
DEFAULT_LONGITUDE_COL = 'longitude_deg'
# Default zoom level for maps
DEFAULT_SIZE_MAP = 5
# Default opacity for map layers
DEFAULT_OPACITY_MAP = 0.3

# ANCOM
# ------------------------------------------------------------------------------------ #
# Default feature type for ANCOM ('l2' for phylum, 'l6' for genus, etc.)
DEFAULT_FEATURE_TYPE_ANCOM = 'l6'
# Default color column for ANCOM plots ('p' for phylum, 'g' for genus, etc.)
DEFAULT_COLOR_COL_ANCOM = 'p'