# file: config_schema.py
# ==================================================================================== #

# Standard Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-Party Imports
from pydantic import BaseModel, Field, DirectoryPath, FilePath

# Local Imports
from workflow_16s.utils.logger import get_logger

# ==================================================================================== #

logger = get_logger()

# ============================ NESTED CONFIG MODELS ================================== #

# --- Paths & Datasets ---
class PathsConfig(BaseModel):
    base: DirectoryPath
    project: Path
    manual_metadata: DirectoryPath
    blast_db: Path
    vsearch_db: FilePath
    phylogeny: DirectoryPath
    classifier: DirectoryPath
    dataset_list: FilePath
    dataset_info: FilePath
    primer_db: Path = Field(..., description="Path to the primer SQLite database (primer_data.db).")

class DatasetConfig(BaseModel):
    dataset_list: FilePath
    dataset_info: FilePath

# --- Sequence Processing (Upstream) ---
class ENAConfig(BaseModel):
    max_concurrent: int = Field(..., gt=0)

class PCRPrimersConfig(BaseModel):
    mode: str
    n_threads: int

class FastQCConfig(BaseModel):
    enabled: bool
    cleanup: bool

class SeqKitConfig(BaseModel):
    enabled: bool
    max_workers: int = Field(..., gt=0)

class QualityControlConfig(BaseModel):
    fastqc: FastQCConfig
    seqkit: SeqKitConfig

class Validate16SConfig(BaseModel):
    enabled: bool
    n_threads: int
    concurrent_jobs: int
    n_runs: int
    run_targets: List[str]
    pident: float = Field(..., ge=0.0, le=1.0)

class CutAdaptConfig(BaseModel):
    enabled: bool
    n_cores: int = Field(..., gt=0)
    min_seq_length: int = Field(..., gt=0)
    start_trim: int
    end_trim: int
    start_q_cutoff: int
    end_q_cutoff: int
    
class TrimConfig(BaseModel):
    cutadapt: CutAdaptConfig

class SequenceProcessingConfig(BaseModel):
    target_subfragment: str
    ena: ENAConfig
    pcr_primers: PCRPrimersConfig
    quality_control: QualityControlConfig
    max_file_size_gb: float = Field(5.0, description="Maximum allowed size (GB) for raw FASTQ files before skipping a dataset (assumed metagenomic).")
    validate_16s: Validate16SConfig
    trim: TrimConfig
    cleanup_raw_files: bool = Field(True, description="Whether to delete raw FASTQ files after successful processing.")
    
# --- QIIME 2 Processing ---
class QIIMETrimConfig(BaseModel):
    enabled: bool

class QIIIMEDenoiseConfig(BaseModel):
    chimera_method: str
    denoise_algorithm: str

class QIIIMETaxonomyConfig(BaseModel):
    classifier_dir: Path
    classifier: str
    classify_method: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class QIIIMEFilterConfig(BaseModel):
    retain_threshold: int

class QIIMEPerDatasetConfig(BaseModel):
    script_path: FilePath
    hard_rerun: bool
    trim: QIIMETrimConfig
    denoise: QIIIMEDenoiseConfig
    taxonomy: QIIIMETaxonomyConfig
    filter: QIIIMEFilterConfig
    collapse_level: int = 6

class QIIMEConfig(BaseModel):
    hard_rerun: bool
    per_dataset: QIIMEPerDatasetConfig

# --- Metadata & Execution ---
class MetadataFiltersConfig(BaseModel):
    amplicon: bool
    no_host: bool

class MetadataColumnsConfig(BaseModel):
    sample_id: str
    dataset: str
    groups: List[Dict[str, Any]]
    
class MetadataConfig(BaseModel):
    filters: MetadataFiltersConfig
    mappings: Dict[str, Any]
    columns: MetadataColumnsConfig
    columns_to_drop: List[str]
    force_numeric_columns: List[str]
    categorical_mappings: Dict[str, Any]
    suffixes_to_collapse: List[str]

class ExecutionConfig(BaseModel):
    threads: int = Field(..., gt=0)
    cpu_limit: int
    max_concurrency: int

# --- API Credentials & Web ---
class CredentialsConfig(BaseModel):
    email: str
    ena_email: str
    mindat_api_key: Optional[str] = None
    airnow_api_key: Optional[str] = None
    usgs_api_key: Optional[str] = None
    ncbi_api_key: Optional[str] = None
    nrel_api_key: Optional[str] = None
    openaq_api_key: Optional[str] = None
    google_earth_engine_project: Optional[str] = None
    springer_api_key: Optional[str] = None
    ieee_api_key: Optional[str] = None
    mendeley_api_key: Optional[str] = None
    dimensions_api_key: Optional[str] = None

class WebConfig(BaseModel):
    user_agent: str

# --- Downstream Analysis ---
class GroupColumnConfig(BaseModel):
    name: str
    type: str
    values: List[Union[bool, str, int, float]]

class AlphaDiversityPlotsConfig(BaseModel):
    enabled: bool
    add_points: bool
    add_stat_annot: bool
    effect_size_threshold: float

class AlphaDiversityCorrelationConfig(BaseModel):
    enabled: bool
    max_categories: int
    min_group_size: int
    top_n_correlations: int

class AlphaDiversityTableConfig(BaseModel):
    enabled: bool
    levels: List[str]

class AlphaDiversityConfig(BaseModel):
    enabled: bool
    plots: AlphaDiversityPlotsConfig
    parametric: bool
    correlation_analysis: AlphaDiversityCorrelationConfig
    tables: Dict[str, AlphaDiversityTableConfig]
    metrics: List[str]

class BetaDiversityTableConfig(BaseModel):
    enabled: bool
    pcoa_metric: str
    methods: List[str]
    levels: List[str]

class BetaDiversityConfig(BaseModel):
    enabled: bool
    load_existing: bool
    max_workers: int
    cpu_limit: int
    tables: Dict[str, BetaDiversityTableConfig]

class MLPermutationImportanceConfig(BaseModel):
    enabled: bool

class MLPlotsConfig(BaseModel):
    enabled: bool

class MLTableConfig(BaseModel):
    enabled: bool
    levels: List[str]
    methods: List[str]

class MLConfig(BaseModel):
    enabled: bool
    load_existing: bool
    n_threads: int
    num_features: int
    step_size: int
    permutation_importance: MLPermutationImportanceConfig
    plots: MLPlotsConfig
    tables: Dict[str, MLTableConfig]

class DownstreamConfig(BaseModel):
    enabled: bool
    find_subsets: bool

# --- Misc Top-Level Sections ---
class NFCConfig(BaseModel):
    enabled: bool
    use_cache: bool
    use_local: bool
    databases: List[str]
    match_existing_samples: bool
    distance_threshold_km: int
    fetch_nearby_samples: bool
    max_distance_km: int
    maps: bool

class CPUConfig(BaseModel):
    limit: int

class FaprotaxConfig(BaseModel):
    enabled: bool

class Picrust2Config(BaseModel):
    enabled: bool
    
class FunctionalConfig(BaseModel):
    faprotax: FaprotaxConfig
    picrust2: Picrust2Config
    

class CleanMetadataConfig(BaseModel):
    enabled: bool
    key_cols: List[str]
    cols_to_keep: List[str]

class RebuildTreeConfig(BaseModel):
    enabled: bool

class FilterConfig(BaseModel):
    enabled: bool
    min_sequencing_depth: int
    min_sample_prevalence: int
    terms_to_exclude: List[str]
    contaminant_terms: List[str]
    admin_noise_columns: List[str]
    #min_sample_depth: int
    #min_feature_prevalence: float
    
class PreprocessingConfig(BaseModel):
    clean_metadata: CleanMetadataConfig
    rebuild_tree: RebuildTreeConfig
    filter: FilterConfig

# ======================== THE MAIN APP CONFIG MODEL =============================== #
# This class brings all the nested models together and models the top-level keys
# in your config.yaml file.
# ==================================================================================== #
class AppConfig(BaseModel):
    # Core sections
    paths: PathsConfig
    datasets: DatasetConfig
    sequences: SequenceProcessingConfig
    qiime2: QIIMEConfig
    metadata: MetadataConfig
    execution: ExecutionConfig
    credentials: CredentialsConfig
    functional: FunctionalConfig
    preprocessing: PreprocessingConfig
    # Other primary tools and sections
    nfc_facilities: NFCConfig
    web: WebConfig
    cpu: CPUConfig
    upstream: Dict[str, Any]
    downstream: DownstreamConfig
    
    # Downstream analysis configurations
    group_column: str
    group_column_type: str
    group_column_values: List[Union[bool, str, int, float]]
    dashboard: Dict[str, Any]
    maps: Dict[str, Any]
    features: Dict[str, Any]
    alpha_diversity: AlphaDiversityConfig
    stats: Dict[str, Any]
    beta_diversity: BetaDiversityConfig
    ml: MLConfig
    faprotax: Dict[str, Any]
    top_features: Dict[str, Any]
    feature_maps: Dict[str, Any]
    
    # Renamed key for consistency
    clean_fastq: Dict[str, Any]
    
    # General flag
    verbose: bool = False
    
# ==================================================================================== #

from pathlib import Path
import yaml
from pydantic import ValidationError

def load_config(path: Path) -> AppConfig:
    """Loads and validates the configuration from a YAML file."""
    with open(path, 'r') as f: data = yaml.safe_load(f)
    
    try:
        # Pydantic automatically validates the data against the class structure
        validated_config = AppConfig(**data)
        return validated_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    