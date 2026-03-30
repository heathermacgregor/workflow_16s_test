# src/workflow_16s/downstream/machine_learning/constants.py

"""
Global constants for the 16S ML discovery suite.
Defines taxonomic markers for feature recovery and environmental target mapping.
"""

# --- TAXONOMY RECOVERY CONSTANTS ---
# Used by resolve_feature_names to skip technical columns and find biological names.
EXPECTED_VAR_COLUMNS = {
    'Taxon', 'Confidence', 'sequence', 'feature_id', 
    'asv_id', 'otu_id', 'id', 'Taxonomy', 'Lineage'
}

# --- ENVIRONMENTAL TARGET GROUPS ---
# Used to automatically identify targets for the SoilGrids & Environmental suites.

PH_TARGETS = [
    'ph', 'pH', 'ph_avg', 'biosample_ph', 'soil_ph',
    'SoilGrids_phh2o_0-5cm', 'SoilGrids_phh2o_5-15cm', 
    'SoilGrids_phh2o_15-30cm', 'SoilGrids_phh2o_30-60cm'
]

ORGANIC_CARBON_TARGETS = [
    'soc', 'organic_carbon', 'carbon_content',
    'SoilGrids_soc_0-5cm', 'SoilGrids_soc_5-15cm',
    'SoilGrids_soc_15-30cm'
]

BULK_DENSITY_TARGETS = [
    'bdod', 'bulk_density',
    'SoilGrids_bdod_0-5cm', 'SoilGrids_bdod_5-15cm',
    'SoilGrids_bdod_15-30cm'
]

# --- FORENSIC & FACILITY CONSTANTS ---
FACILITY_TARGETS = [
    'facility_match', 'facility_type', 'distance_to_facility_km',
    'is_analog_site', 'nuclear_fuel_cycle_stage'
]

# --- BATCH EFFECT MARKERS ---
# Keywords used to identify technical metadata in batch-impact plots.
BATCH_KEYWORDS = [
    'batch', 'study', 'accession', 'dataset', 'center', 
    'sequencing', 'instrument', 'run', 'library', 'pcr'
]

# Summary list for general ML eligibility checks
MANDATORY_METADATA = PH_TARGETS + ORGANIC_CARBON_TARGETS + BULK_DENSITY_TARGETS + FACILITY_TARGETS