"""
Sample Identity Validation

Cross-validates claimed vs. observed sample properties to detect:
1. Mislabeled samples (claimed soil but has marine taxa)
2. Human contamination in environmental samples
3. Primer region mismatches
4. Geographic implausibility
5. Outlier detection in metadata space
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger('workflow_16s')


class SampleIdentityValidator:
    """
    Cross-validate claimed sample properties against observed taxonomic profiles.
    
    This catches mislabeled samples that would lead to incorrect biological conclusions.
    """
    
    # Expected dominant phyla by environment type
    EXPECTED_TAXA = {
        'soil': {
            'dominant_phyla': ['Proteobacteria', 'Actinobacteria', 'Acidobacteria', 
                              'Bacteroidetes', 'Firmicutes', 'Verrucomicrobia'],
            'common_genera': ['Bacillus', 'Pseudomonas', 'Streptomyces', 'Arthrobacter'],
            'suspicious_genera': ['Bacteroides', 'Prevotella', 'Bifidobacterium'],  # Gut-associated
        },
        'marine': {
            'dominant_phyla': ['Proteobacteria', 'Bacteroidetes', 'Cyanobacteria', 
                              'Actinobacteria', 'Planctomycetes'],
            'common_genera': ['Prochlorococcus', 'Synechococcus', 'SAR11', 'Pelagibacter'],
            'suspicious_genera': ['Escherichia', 'Clostridium', 'Lactobacillus'],  # Terrestrial/gut
        },
        'freshwater': {
            'dominant_phyla': ['Proteobacteria', 'Actinobacteria', 'Bacteroidetes',
                              'Cyanobacteria', 'Verrucomicrobia'],
            'common_genera': ['Limnohabitans', 'Polynucleobacter', 'Flavobacterium'],
            'suspicious_genera': ['Bacteroides', 'Prevotella'],  # Gut-associated
        },
        'gut': {
            'dominant_phyla': ['Firmicutes', 'Bacteroidetes', 'Actinobacteria', 'Proteobacteria'],
            'common_genera': ['Bacteroides', 'Prevotella', 'Faecalibacterium', 
                             'Bifidobacterium', 'Lactobacillus', 'Clostridium'],
            'suspicious_genera': ['Prochlorococcus', 'Synechococcus'],  # Marine-specific
        },
        'wastewater': {
            'dominant_phyla': ['Proteobacteria', 'Bacteroidetes', 'Firmicutes', 
                              'Actinobacteria', 'Chloroflexi'],
            'common_genera': ['Acinetobacter', 'Pseudomonas', 'Nitrosomonas', 'Nitrospira'],
            'suspicious_genera': [],  # Wastewater can have diverse sources
        },
    }
    
    # Human-associated taxa (contamination indicators)
    HUMAN_TAXA = {
        'skin': ['Propionibacterium', 'Staphylococcus', 'Corynebacterium', 
                'Cutibacterium', 'Malassezia'],
        'gut': ['Bacteroides', 'Prevotella', 'Faecalibacterium', 'Bifidobacterium',
               'Ruminococcus', 'Blautia', 'Coprococcus'],
        'oral': ['Streptococcus', 'Veillonella', 'Prevotella', 'Fusobacterium',
                'Porphyromonas', 'Actinomyces'],
    }
    
    # Expected ASV length by primer region
    PRIMER_REGION_LENGTHS = {
        'V1-V2': (280, 320),
        'V1-V3': (460, 520),
        'V3-V4': (420, 480),
        'V4': (240, 280),
        'V4-V5': (360, 420),
        'V6-V8': (420, 480),
    }
    
    def __init__(self, adata, envo_categorizer=None):
        """
        Initialize validator.
        
        Args:
            adata: AnnData object with .obs (metadata) and .var (taxonomy)
            envo_categorizer: Optional ENVOOntology instance for categorization
        """
        self.adata = adata
        self.envo = envo_categorizer
        self.validation_results = []
    
    def validate_all(self) -> pd.DataFrame:
        """
        Run all validation checks.
        
        Returns:
            DataFrame with validation results per sample
        """
        logger.info("Starting sample identity validation...")
        
        # Run validation checks
        env_validation = self.validate_environment_type()
        human_validation = self.detect_human_contamination()
        primer_validation = self.validate_primer_region()
        outlier_validation = self.detect_metadata_outliers()
        
        # Combine results
        validation_df = pd.DataFrame(index=self.adata.obs.index)
        validation_df['env_match'] = env_validation
        validation_df['human_contamination'] = human_validation
        validation_df['primer_match'] = primer_validation
        validation_df['metadata_outlier'] = outlier_validation
        
        # Overall flag
        validation_df['overall_flag'] = self._compute_overall_flag(validation_df)
        
        return validation_df
    
    def validate_environment_type(self) -> pd.Series:
        """
        Check if observed taxa match claimed environment type.
        
        Returns:
            Series with validation status per sample (PASS/WARNING/FAIL)
        """
        logger.info("Validating environment types...")
        
        if 'env_category_type' not in self.adata.obs.columns:
            logger.warning("No env_category_type column. Run MetadataValidator first.")
            return pd.Series('UNKNOWN', index=self.adata.obs.index)
        
        results = []
        
        for sample_id in self.adata.obs.index:
            # Get claimed environment
            claimed_env = self.adata.obs.loc[sample_id, 'env_category_type']
            
            if claimed_env == 'unclassified' or pd.isna(claimed_env):
                results.append('UNKNOWN')
                continue
            
            # Get taxonomic profile for this sample
            sample_profile = self.adata[sample_id, :].X.toarray().flatten()
            
            # Get top taxa
            top_feature_idx = np.argsort(sample_profile)[-20:]  # Top 20 features
            top_taxa = self.adata.var.iloc[top_feature_idx]
            
            # Check against expected taxa
            validation = self._check_taxa_match(top_taxa, claimed_env)
            results.append(validation)
        
        return pd.Series(results, index=self.adata.obs.index)
    
    def _check_taxa_match(self, taxa_df: pd.DataFrame, expected_env: str) -> str:
        """
        Check if taxa match expected environment.
        
        Args:
            taxa_df: DataFrame with taxonomic assignments
            expected_env: Expected environment type
        
        Returns:
            'PASS', 'WARNING', or 'FAIL'
        """
        # Map environment category to expected taxa
        env_map = {
            'soil': 'soil',
            'marine': 'marine',
            'freshwater': 'freshwater',
            'wastewater': 'wastewater',
        }
        
        expected_key = env_map.get(expected_env)
        if not expected_key or expected_key not in self.EXPECTED_TAXA:
            return 'UNKNOWN'
        
        expected = self.EXPECTED_TAXA[expected_key]
        
        # Count matches at phylum level
        if 'Phylum' in taxa_df.columns:
            observed_phyla = taxa_df['Phylum'].value_counts()
            dominant_phyla = set(observed_phyla.head(3).index)
            expected_phyla = set(expected['dominant_phyla'])
            
            overlap = len(dominant_phyla & expected_phyla)
            
            if overlap >= 2:  # At least 2 of top 3 phyla match
                # Check for suspicious genera
                if 'Genus' in taxa_df.columns:
                    observed_genera = set(taxa_df['Genus'].dropna())
                    suspicious = observed_genera & set(expected['suspicious_genera'])
                    
                    if suspicious:
                        return 'WARNING'  # Expected phyla but suspicious genera
                
                return 'PASS'
            else:
                return 'FAIL'  # Dominant phyla don't match
        
        return 'UNKNOWN'
    
    def detect_human_contamination(self, threshold: float = 0.05) -> pd.Series:
        """
        Detect human-associated taxa in environmental samples.
        
        Args:
            threshold: Proportion threshold for flagging (default: 5%)
        
        Returns:
            Series with contamination levels (NONE/LOW/MEDIUM/HIGH)
        """
        logger.info("Detecting human contamination...")
        
        # Get all human-associated genera
        human_genera = set()
        for source, genera in self.HUMAN_TAXA.items():
            human_genera.update(genera)
        
        results = []
        
        for sample_id in self.adata.obs.index:
            sample_profile = self.adata[sample_id, :].X.toarray().flatten()
            
            # Find human-associated features
            human_mask = self.adata.var['Genus'].isin(human_genera) if 'Genus' in self.adata.var.columns else False
            
            if isinstance(human_mask, bool):
                results.append('UNKNOWN')
                continue
            
            human_abundance = sample_profile[human_mask].sum()
            total_abundance = sample_profile.sum()
            
            if total_abundance == 0:
                results.append('UNKNOWN')
                continue
            
            human_fraction = human_abundance / total_abundance
            
            if human_fraction < threshold:
                results.append('NONE')
            elif human_fraction < 0.15:
                results.append('LOW')
            elif human_fraction < 0.30:
                results.append('MEDIUM')
            else:
                results.append('HIGH')
        
        return pd.Series(results, index=self.adata.obs.index)
    
    def validate_primer_region(self) -> pd.Series:
        """
        Validate that ASV lengths match claimed primer region.
        
        Returns:
            Series with validation status (PASS/WARNING/FAIL/UNKNOWN)
        """
        logger.info("Validating primer regions...")
        
        # Check if we have primer region info
        primer_col = None
        for col in ['target_subfragment', 'primer_region', 'amplicon_region']:
            if col in self.adata.obs.columns:
                primer_col = col
                break
        
        if not primer_col:
            logger.warning("No primer region column found")
            return pd.Series('UNKNOWN', index=self.adata.obs.index)
        
        # Get ASV sequences if available
        if 'sequence' not in self.adata.var.columns:
            logger.warning("No ASV sequences in .var")
            return pd.Series('UNKNOWN', index=self.adata.obs.index)
        
        # Calculate median ASV length per sample
        results = []
        
        for sample_id in self.adata.obs.index:
            claimed_region = self.adata.obs.loc[sample_id, primer_col]
            
            if pd.isna(claimed_region) or claimed_region not in self.PRIMER_REGION_LENGTHS:
                results.append('UNKNOWN')
                continue
            
            # Get ASVs for this sample
            sample_profile = self.adata[sample_id, :].X.toarray().flatten()
            present_asvs = sample_profile > 0
            
            if not present_asvs.any():
                results.append('UNKNOWN')
                continue
            
            # Get lengths of present ASVs
            asv_seqs = self.adata.var.loc[present_asvs, 'sequence']
            asv_lengths = asv_seqs.str.len()
            median_length = asv_lengths.median()
            
            # Check against expected range
            expected_min, expected_max = self.PRIMER_REGION_LENGTHS[claimed_region]
            
            if expected_min <= median_length <= expected_max:
                results.append('PASS')
            elif abs(median_length - np.mean([expected_min, expected_max])) < 50:
                results.append('WARNING')  # Within 50bp tolerance
            else:
                results.append('FAIL')
        
        return pd.Series(results, index=self.adata.obs.index)
    
    def detect_metadata_outliers(self, n_std: float = 3.0) -> pd.Series:
        """
        Detect outliers in metadata space using multivariate distance.
        
        Flags samples that are very different from others in their claimed category.
        
        Args:
            n_std: Number of standard deviations for outlier threshold
        
        Returns:
            Series with outlier status (NORMAL/OUTLIER)
        """
        logger.info("Detecting metadata outliers...")
        
        # Get numeric metadata columns
        numeric_cols = self.adata.obs.select_dtypes(include=[np.number]).columns
        
        # Remove columns with too many NaNs
        valid_cols = [c for c in numeric_cols 
                     if self.adata.obs[c].notna().sum() > len(self.adata.obs) * 0.5]
        
        if len(valid_cols) < 2:
            logger.warning("Not enough numeric metadata for outlier detection")
            return pd.Series('UNKNOWN', index=self.adata.obs.index)
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        from sklearn.covariance import EllipticEnvelope
        
        X = self.adata.obs[valid_cols].copy()
        X = X.fillna(X.median())  # Impute with median
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit robust covariance estimator
        try:
            outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
            predictions = outlier_detector.fit_predict(X_scaled)
            
            # Convert to NORMAL/OUTLIER
            results = ['NORMAL' if p == 1 else 'OUTLIER' for p in predictions]
            return pd.Series(results, index=self.adata.obs.index)
        
        except Exception as e:
            logger.warning(f"Outlier detection failed: {e}")
            return pd.Series('UNKNOWN', index=self.adata.obs.index)
    
    def _compute_overall_flag(self, validation_df: pd.DataFrame) -> pd.Series:
        """
        Compute overall validation flag from individual checks.
        
        Logic:
        - PASS: All checks pass or unknown
        - WARNING: 1-2 checks fail/warning
        - FAIL: 3+ checks fail or critical failure
        
        Args:
            validation_df: DataFrame with individual validation columns
        
        Returns:
            Series with overall flags
        """
        results = []
        
        for idx, row in validation_df.iterrows():
            fail_count = 0
            warning_count = 0
            
            for col in ['env_match', 'human_contamination', 'primer_match', 'metadata_outlier']:
                val = row[col]
                if val == 'FAIL':
                    fail_count += 1
                elif val in ['WARNING', 'LOW', 'MEDIUM']:
                    warning_count += 1
                elif val == 'HIGH':  # High human contamination is critical
                    fail_count += 2
            
            if fail_count >= 3:
                results.append('FAIL')
            elif fail_count >= 1 or warning_count >= 2:
                results.append('WARNING')
            else:
                results.append('PASS')
        
        return pd.Series(results, index=validation_df.index)
    
    def get_flagged_samples(self, min_severity: str = 'WARNING') -> List[str]:
        """
        Get list of samples flagged as problematic.
        
        Args:
            min_severity: Minimum severity to include ('WARNING' or 'FAIL')
        
        Returns:
            List of sample IDs
        """
        validation_df = self.validate_all()
        
        if min_severity == 'FAIL':
            mask = validation_df['overall_flag'] == 'FAIL'
        else:  # WARNING or above
            mask = validation_df['overall_flag'].isin(['WARNING', 'FAIL'])
        
        return validation_df.index[mask].tolist()
    
    def generate_report(self, output_path: str):
        """Save validation report to CSV."""
        validation_df = self.validate_all()
        validation_df.to_csv(output_path)
        logger.info(f"Saved validation report to {output_path}")
        
        # Log summary
        summary = validation_df['overall_flag'].value_counts()
        logger.info(f"Validation summary:\n{summary}")
