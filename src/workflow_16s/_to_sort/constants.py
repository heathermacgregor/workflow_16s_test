import os
import re
from pathlib import Path
from typing import Dict, List

# ==================================================================================== #
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
MODE = 'genus'
FEATURE_TYPE = 'ASV'

SAMPLE_ID_COLUMN = '#sampleid'
GROUP_COLUMN: str = "nuclear_contamination_status"
GROUP_COLUMNS: List[Dict] = [
    {
        'name': "nuclear_contamination_status",
        'type': "bool",
        'values': [True, False]
    },
]

GROUP_THRESHOLD: float = 0.05
PREVALENCE_THRESHOLD: float = 0.05

TAXONOMIC_LEVELS = {'phylum': 2, 'class': 3, 'order': 4, 'family': 5, 'genus': 6}
TAXONOMIC_LEVELS_MAPPING = {
    'phylum': 2, 
    'class': 3, 
    'order': 4, 
    'family': 5, 
    'genus': 6
}

SET_SAMPLE_ID_COLUMN = 'sample_id'
SET_DATASET_ID_COLUMN = 'dataset_id'

# ==================================================================================== #
# PROGRESS BAR
# ==================================================================================== #
# See supported colors at: https://www.w3schools.com/colors/colors_x11.asp
PROGRESS_BAR_SETTINGS: Dict = {
    'text_width': 65,
    'description_color': 'white',
    'bar_width': 40,
    'bar_column_complete_color': 'honeydew2',
    'finished_color': 'dark_cyan',
    'percentage_color': 'honeydew2',
    'm_of_n_complete_color': 'honeydew2',
    'time_elapsed_color': 'light_sky_blue1',
    'time_remaining_color': 'thistle1'
}

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
CONFIG_PATH = REFERENCES_DIR / "config.yaml"


ENA_EMAIL = "macgregor@berkeley.edu"
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

MINDAT_API_KEY = "ee6cec6c0c15e4e2960871477dbfb072"


DEFAULT_GROUP_COLUMN: str = "nuclear_contamination_status"
# ==================================================================================== #
# PROGRESS BAR
# ==================================================================================== #
# See supported colors at: https://www.w3schools.com/colors/colors_x11.asp

# Total character width of the progress bar text
DEFAULT_PROGRESS_TEXT_N: int = 65
DEFAULT_N: int = 65 
# Color of the progress bar description text
DEFAULT_DESCRIPTION_STYLE: str = "white"
# Width of the progress bar
DEFAULT_BAR_WIDTH: int = 40
# Color of the filled/complete portion of the progress bar
DEFAULT_BAR_COLUMN_COMPLETE_STYLE: str = "honeydew2"
# Color used when the progress bar is finished
DEFAULT_FINISHED_STYLE: str = "dark_cyan" 
# Color of the percentage complete text (e.g., "85%")
DEFAULT_PROGRESS_PERCENTAGE_STYLE: str = "honeydew2"
# Color of the "X of Y complete" text (e.g., "42 of 65")
DEFAULT_M_OF_N_COMPLETE_STYLE: str = "honeydew2"
# Color of the time elapsed display (e.g., "E: 00:01:25")
DEFAULT_TIME_ELAPSED_STYLE: str = "light_sky_blue1"
# Color of the estimated time remaining display (e.g., "R: 00:00:34")
DEFAULT_TIME_REMAINING_STYLE: str = "thistle1"

# ==================================================================================== #
# SETTINGS
# ==================================================================================== #
# Go up two levels 
REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "references" / "config.yaml"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG
DEFAULT_MODE = 'genus'
DEFAULT_FEATURE_TYPE = 'ASV'

DEFAULT_EMAIL = "macgregor@berkeley.edu"
MINDAT_API_KEY = "ee6cec6c0c15e4e2960871477dbfb072"

# ==================================================================================== #
# ENA
# ==================================================================================== #
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

# ==================================================================================== #
# METADATA
# ==================================================================================== #
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

# ==================================================================================== #
# SEQUENCES
# ==================================================================================== #
DEFAULT_MAX_WORKERS_ENA = 16
DEFAULT_MAX_WORKERS_SEQKIT = 8

levels = {
    'phylum': 2, 
    'class': 3, 
    'order': 4, 
    'family': 5, 
    'genus': 6
}

DEFAULT_REGIONS = {
    'V1-V2': (100, 400),
    'V2-V3': (200, 500),
    'V3-V4': (350, 800),
    'V4': (515, 806),
    'V4-V5': (515, 950),
    'V5-V7': (830, 1190),
    'V6-V8': (900, 1400),
    'V7-V9': (1100, 1500)
}

DEFAULT_PRIMER_REGIONS = {
    "V1-V2": ("AGAGTTTGATCMTGGCTCAG", "TGCTGCCTCCCGTAGGAGT"),
    "V2-V3": ("ACTCCTACGGGAGGCAGCAG", "TTACCGCGGCTGCTGGCAC"),
    "V3-V4": ("CCTACGGGNGGCWGCAG", "GACTACHVGGGTATCTAATCC"),
    "V4": ("GTGCCAGCMGCCGCGGTAA", "GGACTACHVGGGTWTCTAAT"),
    "V4-V5": ("GTGYCAGCMGCCGCGGTAA", "CCGYCAATTYMTTTRAGTTT"),
    "V6-V8": ("AAACTYAAAKGAATTGACGG", "ACGGGCGGTGTGTACAAG")
}

DEFAULT_16S_PRIMERS = {
    "V1-V2": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "AGAGTTTGATCMTGGCTCAG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "TGCTGCCTCCCGTAGGAGT",
            "ref": None,
        },
    },
    "V2-V3": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "ACTCCTACGGGAGGCAGCAG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "TTACCGCGGCTGCTGGCAC",
            "ref": None,
        },
    },
    "V3-V4": {
        "fwd": {
            "name": "Bakt_341F",
            "full_name": "S-D-Bact-0341-b-S-17",
            "position": (341, 357),
            "seq": "CCTACGGGNGGCWGCAG",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
        "rev": {
            "name": "Bakt_805R",
            "full_name": "S-D-Bact-0785-a-A-21",
            "position": (785, 805),
            "seq": "GACTACHVGGGTATCTAATCC",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
    },
    "V4": {
        "fwd": {
            "name": "U515F",
            "full_name": "S-*-Univ-0515-a-S-19",
            "position": (515, 533),
            "seq": "GTGCCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
        "rev": {
            "name": "806R",
            "full_name": "S-D-Bact-0787-b-A-20",
            "position": (787, 808),
            "seq": "GGACTACHVGGGTWTCTAAT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
    },
    "V4-V5": {
        "fwd": {
            "name": "515F-Y",
            "full_name": None,
            "position": (515, 533),
            "seq": "GTGYCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
        "rev": {
            "name": "926R",
            "full_name": "S-D-Bact-0907-a-A-19",
            "position": (907, 926),
            "seq": "CCGYCAATTYMTTTRAGTTT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
    },
    "V6-V8": {
        "fwd": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "AAACTYAAAKGAATTGACGG",
            "ref": None,
        },
        "rev": {
            "name": None,
            "full_name": None,
            "position": (0, 0),
            "seq": "ACGGGCGGTGTGTACAAG",
            "ref": None,
        },
    },
}

# ==================================================================================== #
# QIIME
# ==================================================================================== #
DEFAULT_PER_DATASET = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))) 
    / "src" / "workflow_16s" / "qiime" / "workflows" / "per_dataset_run.py"
)
DEFAULT_SAVE_INTERMEDIATES = True

DEFAULT_N_THREADS = 32
DEFAULT_PADDING_WIDTH = 15

DEFAULT_LIBRARY_LAYOUT = "paired"
DEFAULT_MINIMUM_LENGTH = 100
DEFAULT_MIN_READS = 1000
DEFAULT_TRUNC_LEN_F = 250
DEFAULT_TRUNC_LEN_R = 250
DEFAULT_TRIM_LENGTH = 250
DEFAULT_TRIM_LEFT_F = 0
DEFAULT_TRIM_LEFT_R = 0
DEFAULT_TRUNC_Q = 2
DEFAULT_MAX_EE = 2
DEFAULT_MAX_EE_F = 2
DEFAULT_MAX_EE_R = 10
DEFAULT_CHIMERA_METHOD = "consensus"
DEFAULT_DENOISE_ALGORITHM = "DADA2"
DEFAULT_CLASSIFIER = "silva-138-99-515-806"
DEFAULT_CLASSIFIER_DIR = (
    "/usr2/people/macgregor/mtv_project/references/"
    "hrm_workflow/classifier/silva-138-99-515-806"
)
DEFAULT_CLASSIFY_METHOD = "sklearn"
DEFAULT_MAXACCEPTS = 50
DEFAULT_PERC_IDENTITY = 0.99
DEFAULT_QUERY_COV = 0.9
DEFAULT_CONFIDENCE = 0.7
DEFAULT_MSA_N_SEQUENCES = 1000000

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
debug_mode = False

# ==================================================================================== #
# BETA DIVERSITY
# ==================================================================================== #
DEFAULT_METRIC = 'braycurtis'
DEFAULT_N_PCA = 20
DEFAULT_N_PCOA = None
DEFAULT_N_TSNE = 3
DEFAULT_N_UMAP = 3
DEFAULT_N_MDS = 20
DEFAULT_RANDOM_STATE = 0
DEFAULT_CPU_LIMIT = 4

# ==================================================================================== #
# STATISTICS
# ==================================================================================== #
DEFAULT_MIN_REL_ABUNDANCE: float = 1
DEFAULT_MIN_SAMPLES: int = 10
DEFAULT_MIN_COUNTS: int = 1000
DEFAULT_PSEUDOCOUNT: float = 1e-5

DEFAULT_PREVALENCE_THRESHOLD: float = 0.05
DEFAULT_GROUP_THRESHOLD: float = 0.05

# ==================================================================================== #
# CATBOOST FEATURE SELECTION
# ==================================================================================== #
DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42

DEFAULT_METHOD = 'rfe'
DEFAULT_USE_PERMUTATION_IMPORTANCE = True
DEFAULT_THREAD_COUNT = 4
DEFAULT_STEP_SIZE = 1000
DEFAULT_NUM_FEATURES = 500

DEFAULT_ITERATIONS_RFE = 500
DEFAULT_LEARNING_RATE_RFE = 0.1
DEFAULT_DEPTH_RFE = 4

DEFAULT_PENALTY_LASSO = 'l1'
DEFAULT_SOLVER_LASSO = 'liblinear'
DEFAULT_MAX_ITER_LASSO = 1000

DEFAULT_ITERATIONS_SHAP = 1000
DEFAULT_LEARNING_RATE_SHAP = 0.1
DEFAULT_DEPTH_SHAP = 4

DEFAULT_PARAM_GRID = {
    'iterations': [500],#, 1000, 1500],
    'learning_rate': [0.01],#, 0.05, 0.1],
    'depth': [4],#, 6, 8],
    'l2_leaf_reg': [1],#, 3, 5, 7],
    'border_count': [32]#, 64, 128]
}
DEFAULT_LOSS_FUNCTION = 'Logloss'
DEFAULT_THREAD_COUNT = 8

# ==================================================================================== #
# FIGURES
# ==================================================================================== #
DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1100

DEFAULT_COLOR_COL = 'dataset_name'
DEFAULT_SYMBOL_COL = DEFAULT_GROUP_COLUMN

DEFAULT_METRIC = 'braycurtis'

DEFAULT_PROJECTION = 'natural earth'
DEFAULT_LATITUDE_COL = 'latitude_deg'
DEFAULT_LONGITUDE_COL = 'longitude_deg'
DEFAULT_SIZE_MAP = 5
DEFAULT_OPACITY_MAP = 0.3

DEFAULT_FEATURE_TYPE_ANCOM = 'l6'
DEFAULT_COLOR_COL_ANCOM = 'p'

# ==================================================================================== #
# NFC FACILITIES
# ==================================================================================== #
DEFAULT_NFCIS_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/references/nfc_facilities/NFCISFacilityList.xlsx'
DEFAULT_NFCIS_COLUMNS = {
    'country': "Country",
    'facility': "Facility Name",
    'facility_type': "Facility Type",
    'facility_capacity': "Design Capacity",
    'facility_status': "Facility Status",
    'facility_start_year': "Start of Operation",
    'facility_end_year': "End of Operation"
}

DEFAULT_GEM_PATH = '/usr2/people/macgregor/amplicon/workflow_16s/references/nfc_facilities/gem_nuclearpower_2024-07.tsv'
DEFAULT_GEM_COLUMNS = {
    'country': "Country/Area",
    'facility': "Project Name",
    'facility_type': "Reactor Type",
    'facility_capacity': " Capacity (MW) ",
    'facility_status': "Status",
    'facility_start_year': "Start Year",
    'facility_end_year': "Retirement Year"
}

DEFAULT_USER_AGENT = "workflow_16s/1.0"
