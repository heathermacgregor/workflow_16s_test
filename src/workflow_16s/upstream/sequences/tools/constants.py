# workflow_16s/upstream/sequences/constants.py

PRIMER_PRESENCE_THRESHOLD = 0.50
VSEARCH_COVERAGE_THRESHOLD = 75.0
PROCESSING_BATCH_SIZE = 150
YAML_OUTPUT_DIR_NAME = "estimations"
TSV_OUTPUT_FILENAME = "region_results.tsv"
PRIMER_DB_NAME = "primer_data.db"
REQUIRED_TOOLS = ["vsearch", "seqtk", "gzip"]

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