# ==================================================================================== #

import re
from pathlib import Path
from typing import Any, Dict, List, Union

from workflow_16s.config_schema import AppConfig

# ==================================================================================== #

class Project:
    def __init__(self, config: AppConfig) -> None:
        self.project = Path(config.paths.project)
        self.main = self.project 
        self.classifier = Path(config.paths.classifier)
        self.raw_data = self.project / "01_raw_data"
        self.qiime = self.project / "02_qiime"
        self.processed_data = self.project / "03_processed_data"
        self.analysis = self.project / "04_analysis"
        self.reports = self.project / "05_reports"
        self.figures = self.project / "06_figures"
        self.logs = self.project / "07_logs"
        self.tmp = self.project / "08_tmp"
        self.cache = self.project / ".cache"
        
        all_dirs = [self.project, self.raw_data, self.qiime,
                    self.processed_data, self.analysis,
                    self.reports, self.figures, self.logs,
                    self.tmp, self.cache]
        for d in all_dirs: d.mkdir(parents=True, exist_ok=True)
            
        self.subsets: List[SubSet] = []
            
# ==================================================================================== #

class Analysis:
    def __init__(self, project: Project, analysis_id: str) -> None:
        self.project = project
        self.analysis_id = analysis_id
        self.main = project.analysis / analysis_id
        self.alpha = self.main / "alpha"
        self.beta = self.main / "beta"
        self.taxonomy = self.main / "taxonomy"
        self.differential_abundance = self.main / "differential_abundance"
        
        for dir_path in [
            self.main, self.alpha, self.beta, self.taxonomy, self.differential_abundance
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

class SubSet:
    """
    01_raw_data/
    └── subset_id/
    │   ├── seqs/
    │   │   ├── raw/
    │   │   └── trimmed/
    │   ├── sample-metadata.tsv
    """
    def __init__(self, project: Project, subset: Union[str, Dict[str, Any]]) -> None:
        if isinstance(subset, str):
            subset_id = subset
        elif isinstance(subset, dict):
            # Create a sanitized subset ID from the subset dictionary
            # Replace non-alphanumeric characters in primer sequences with underscores
            subset_id = (
                subset["dataset"] + '.' 
                + subset["instrument_platform"] + '.' 
                + subset["library_layout"] + '.' 
                + subset["target_subfragment"] + '.' 
                + f"FWD_{re.sub(r'[^a-zA-Z0-9-]', '_', subset['pcr_primer_fwd_seq'])}" + '.' 
                + f"REV_{re.sub(r'[^a-zA-Z0-9-]', '_', subset['pcr_primer_rev_seq'])}"
            ).upper()
        else:
            raise ValueError("Subset must be either a string or a dictionary.")
        self.project = project
        # Raw data directory for this subset
        self.raw_data = project.raw_data / subset_id
        # QIIME2 directory for this subset
        self.qiime = project.qiime / subset_id
        # Create directories if they don't exist
        for dir_path in [self.raw_data, self.qiime]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
# ==================================================================================== #
        
class RawData:
    def __init__(self, subset: SubSet):
        self.main = subset.raw_data
        self.seqs = subset.raw_data / "seqs"
        self.raw_seqs = subset.raw_data / "seqs" / "raw"
        self.trimmed_seqs =  subset.raw_data / "seqs" / "trimmed"
        self.metadata_tsv = subset.raw_data / "metadata.tsv"   
        
# ==================================================================================== #   
    
class QIIME:
    def __init__(self, subset: SubSet):
        self.main = subset.qiime
        self.manifest_tsv = subset.qiime / "manifest.tsv"
        self.metadata_tsv = subset.raw_data / "metadata.tsv"
        self.classifier = subset.project.classifier
        
        self.demux_stats = subset.qiime / 'demux-stats' 
        self.demux_stats_raw = subset.qiime / 'demux-stats' / 'raw'
        self.demux_stats_trimmed = subset.qiime / 'demux-stats' / 'trimmed'
        
        self.rep_seqs = subset.qiime / 'rep_seqs'
        self.stats = subset.qiime / 'stats'
        self.table = subset.qiime / 'table'
        self.table_6 = subset.qiime / 'table_6'
        self.taxonomy = subset.qiime / 'taxonomy'
        
        for dir_path in [
            subset.qiime, self.demux_stats, self.demux_stats_raw,
            self.demux_stats_trimmed, self.rep_seqs, self.stats, self.table, 
            self.table_6, self.taxonomy
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.rep_seqs_artifact = subset.qiime / 'rep_seqs'
        self.stats_artifact = subset.qiime / 'stats.qza'
        self.table_artifact = subset.qiime / 'table.qza'
        self.table_6_artifact = subset.qiime / 'table_6.qza'
        self.taxonomy_artifact = subset.qiime / 'taxonomy.qza'