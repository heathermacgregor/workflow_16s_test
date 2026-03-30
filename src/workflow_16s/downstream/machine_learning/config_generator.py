# src/workflow_16s/downstream/machine_learning/config_generator.py

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from workflow_16s.utils.logger import get_logger
from .constants import MANDATORY_METADATA
from .validation.check_study_eligibility import StudyEligibilityManager


class MLConfigGenerator:
    """
    Scans an AnnData object and generates a tailored ML configuration.
    Ensures that only targets meeting the 'Forensic Bar' are scheduled for discovery.
    """
    def __init__(self, adata, min_samples: int = 20):
        self.adata = adata
        self.min_samples = min_samples
        self.eligible_targets = []
        self.target_metadata = {}

    def _assess_target_eligibility(self):
        """Filters metadata columns based on volume and variance."""
        # Focus only on known forensic/environmental targets or user-defined priorities
        potential_targets = [c for c in self.adata.obs.columns if c in MANDATORY_METADATA or 'facility' in c.lower()]
        
        for col in potential_targets:
            # Use the Eligibility Manager logic
            auditor = StudyEligibilityManager(self.adata, target_col=col, min_n=self.min_samples)
            report = auditor.diagnose_studies()
            
            passing_studies = report[report['Status'] == "✅ PASS"]
            if not passing_studies.empty:
                total_n = passing_studies['N'].sum()
                n_studies = len(passing_studies)
                
                # Determine task type
                y = self.adata.obs[col].dropna()
                is_numeric = pd.api.types.is_numeric_dtype(y)
                task = "regression" if is_numeric and y.nunique() > 10 else "classification"
                
                self.eligible_targets.append(col)
                self.target_metadata[col] = {
                    "task": task,
                    "samples": int(total_n),
                    "studies": int(n_studies)
                }
                get_logger("workflow_16s").info(f"✨ Target Found: {col} ({task}) - {total_n} samples across {n_studies} studies.")

    def generate_config(self, output_path: Path):
        """Writes the suggested configuration to a YAML file."""
        self._assess_target_eligibility()
        
        config = {
            "ml": {
                "enabled": True,
                "eligibility_mode": "filter",
                "strict_targets": True,
                "targets": self.eligible_targets,
                "grid_settings": {
                    "levels": ["Genus", "Family"],
                    "transformations": ["clr"],
                    "fs_strategies": ["agnostic", "lopocv", "meta_analysis"]
                },
                "model_params": {
                    "iterations": 1000,
                    "depth": 6,
                    "early_stopping_rounds": 50
                },
                "target_profiles": self.target_metadata
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        get_logger("workflow_16s").info(f"✅ ML Configuration generated at: {output_path}")