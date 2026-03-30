"""ANCOM-BC Analysis Wrapper

Wraps R ANCOMBC package for differential abundance analysis with compositional bias correction.
Uses rpy2 to call R backend.

ANCOM-BC detects differentially abundant taxa while accounting for:
- Zero inflation
- Library size variation  
- Sample composition effects
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import anndata
from dataclasses import dataclass

logger = logging.getLogger("workflow_16s")

try:
    import rpy2
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro
    RLIBS_AVAILABLE = True
except ImportError:
    RLIBS_AVAILABLE = False


@dataclass
class ANCAMBCResult:
    """Results from ANCOM-BC analysis"""
    da_table: pd.DataFrame  # Differentially abundant features + stats
    bias_table: Optional[pd.DataFrame] = None  # Estimated biases
    n_significant: int = 0  # Count of significant features
    log_folder: Optional[Path] = None


class ANCAMBCWrapper:
    """Wrapper for R ANCOMBC differential abundance detection"""
    
    def __init__(self, logger_obj=None):
        """
        Initialize wrapper.
        
        Args:
            logger_obj: Logger instance
            
        Raises:
            ImportError: If rpy2 or ANCOMBC not available
        """
        self.logger_obj = logger_obj or logger
        
        if not RLIBS_AVAILABLE:
            raise ImportError(
                "rpy2 not installed. Install with: pip install rpy2\n"
                "Also requires R package: install.packages('ANCOMBC')"
            )
        
        # Enable R-Python conversion
        pandas2ri.activate()
        
        try:
            self.ancombc = importr('ANCOMBC')
            self.stats = importr('stats')
            self.logger_obj.info("✓ Loaded R ANCOMBC package")
        except Exception as e:
            raise ImportError(f"Failed to load ANCOMBC R package: {e}")
    
    def run_ancombc(
        self,
        adata: anndata.AnnData,
        formula: str,
        outcome: str,
        fix_formula: Optional[str] = None,
        rand_formula: Optional[str] = None,
        p_adj_method: str = "BH",
        significance_level: float = 0.05,
        zero_cut: float = 0.90,
        logger_obj=None,
    ) -> ANCAMBCResult:
        """
        Run ANCOM-BC differential abundance analysis.
        
        Args:
            adata: AnnData object with KO abundance in .obsm['KO_counts']
            formula: R formula for full model (e.g., "~ grouping_variable")
            outcome: Column name for outcome of interest
            fix_formula: Fixed effects formula (subset of formula)
            rand_formula: Random effects formula for mixed models
            p_adj_method: p-value adjustment method ("BH", "BY", "bonferroni")
            significance_level: alpha for significance testing
            zero_cut: Exclude features with >X fraction of zeros
            logger_obj: Logger instance
            
        Returns:
            ANCAMBCResult with differential abundance table
        """
        logger_obj = logger_obj or self.logger_obj
        
        # Get KO matrix from AnnData
        if 'KO_CLR' in adata.obsm:
            ko_matrix = adata.obsm['KO_CLR'].T  # Features × samples for ANCOMBC
            logger_obj.debug("Using CLR-transformed KO matrix")
        elif 'KO_counts' in adata.obsm:
            ko_matrix = adata.obsm['KO_counts'].T
            logger_obj.debug("Using raw KO count matrix")
        else:
            raise ValueError("KO matrix not found in adata.obsm")
        
        # Build metadata
        metadata = adata.obs.copy()
        
        # Filter features with high zero fraction
        zero_fracs = (ko_matrix == 0).sum(axis=1) / ko_matrix.shape[1]
        keep_features = zero_fracs < zero_cut
        ko_matrix = ko_matrix[keep_features]
        
        logger_obj.info(
            f"📊 Running ANCOM-BC analysis with formula: {formula}\n"
            f"   Kept {keep_features.sum()}/{len(keep_features)} features "
            f"(zero_cut={zero_cut})"
        )
        
        try:
            # Create R expression for ANCOMBC
            ro.globalenv['count_matrix'] = ko_matrix
            ro.globalenv['metadata'] = pandas2ri.py2r(metadata)
            
            # Build ANCOMBC call
            fix_formula = fix_formula or formula
            
            r_code = f'''
            library(ANCOMBC)
            result <- ancombc(
                phyloseq = list(
                    otu_table = count_matrix,
                    sample_data = metadata
                ),
                formula = "{formula}",
                p_adj_method = "{p_adj_method}",
                zero_cut = {zero_cut},
                test = "Wald",
                alpha = {significance_level}
            )
            result_table <- result$res
            '''
            
            ro.r(r_code)
            result_table_r = ro.r('result_table')
            result_table = pandas2ri.r2py(result_table_r)
            
            n_sig = (result_table['q_value'] <= significance_level).sum()
            logger_obj.info(f"✅ Found {n_sig} significant features (q <= {significance_level})")
            
            return ANCAMBCResult(
                da_table=result_table,
                n_significant=n_sig
            )
            
        except Exception as e:
            logger_obj.error(f"❌ ANCOM-BC analysis failed: {e}")
            raise


class ElasticNetCV:
    """
    Elastic Net with Leave-One-Study-Out Cross-Validation.
    
    Performs feature selection while respecting study boundaries to avoid
    overfitting from multi-study datasets.
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, logger_obj=None):
        """
        Initialize ElasticNetCV.
        
        Args:
            alpha: Regularization strength (0-1, higher = more regularization)
            l1_ratio: Ratio between L1 and L2 (0 = Ridge, 1 = Lasso)
            logger_obj: Logger instance
        """
        try:
            from sklearn.linear_model import ElasticNetCV as SklearnElasticNetCV
            from sklearn.preprocessing import StandardScaler
            self.ElasticNetCV = SklearnElasticNetCV
            self.StandardScaler = StandardScaler
        except ImportError:
            raise ImportError("scikit-learn required for ElasticNetCV")
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.logger_obj = logger_obj or logger
        self.model = None
        self.feature_importances = None
    
    def fit_loocv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        study_ids: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_alphas: int = 100,
    ) -> Dict:
        """
        Fit Elastic Net with Leave-One-Study-Out CV.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            y: Target vector (n_samples,)
            study_ids: Study identifier per sample
            feature_names: Optional feature names
            n_alphas: Number of alpha values to test
            
        Returns:
            Dictionary with model, coefficients, CV scores
        """
        unique_studies = np.unique(study_ids)
        self.logger_obj.info(
            f"🔧 Fitting ElasticNet with LOOCV ({len(unique_studies)} studies)"
        )
        
        cv_scores = []
        coefficients = np.zeros((len(unique_studies), X.shape[1]))
        
        # Leave-one-study-out loop
        for fold, test_study in enumerate(unique_studies):
            train_mask = study_ids != test_study
            test_mask = study_ids == test_study
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Standardize
            scaler = self.StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model = self.ElasticNetCV(
                alphas=np.logspace(-4, 0, n_alphas),
                l1_ratio=[self.l1_ratio],
                cv=5,
                random_state=42,
                max_iter=2000
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            score = model.score(X_test_scaled, y_test)
            cv_scores.append(score)
            coefficients[fold] = model.coef_
            
            self.logger_obj.debug(f"  Study {fold+1}/{len(unique_studies)}: R² = {score:.3f}")
        
        # Average coefficients
        mean_coef = coefficients.mean(axis=0)
        coef_std = coefficients.std(axis=0)
        
        # Feature importance: coefficient magnitude weighted by CV performance
        mean_score = np.mean(cv_scores)
        feature_importances = np.abs(mean_coef) * mean_score
        
        self.logger_obj.info(
            f"✅ LOOCV complete: mean R² = {mean_score:.3f} ± {np.std(cv_scores):.3f}"
        )
        
        return {
            'model': self,
            'coefficients': mean_coef,
            'coefficients_std': coef_std,
            'cv_scores': cv_scores,
            'feature_importances': feature_importances,
            'feature_names': feature_names,
        }


class CandidateFeaturesSelector:
    """
    Select consensus candidate features from multiple methods.
    
    Intersects significant features from ANCOM-BC and ElasticNet
    to get high-confidence candidates.
    """
    
    def __init__(self, logger_obj=None):
        """Initialize selector."""
        self.logger_obj = logger_obj or logger
    
    def select_consensus(
        self,
        ancombc_result: ANCAMBCResult,
        elasticnet_result: Dict,
        ancombc_threshold: float = 0.05,
        elasticnet_percentile: float = 95.0,
        min_consensus: int = 2,
    ) -> Dict:
        """
        Select features identified by multiple methods.
        
        Args:
            ancombc_result: Results from ANCOM-BC
            elasticnet_result: Results from ElasticNet CV
            ancombc_threshold: q-value threshold for ANCOM-BC
            elasticnet_percentile: Percentile threshold for feature importance
            min_consensus: Minimum methods agreeing on feature
            
        Returns:
            Dictionary with consensus features and method-specific rankings
        """
        # Features from ANCOM-BC
        ancombc_features = set(
            ancombc_result.da_table[ancombc_result.da_table['q_value'] <= ancombc_threshold].index
        )
        
        # Features from ElasticNet
        elasticnet_threshold = np.percentile(
            elasticnet_result['feature_importances'],
            elasticnet_percentile
        )
        elasticnet_features = set(
            np.array(elasticnet_result['feature_names'])[
                elasticnet_result['feature_importances'] > elasticnet_threshold
            ]
        )
        
        # Consensus (intersection)
        consensus = ancombc_features & elasticnet_features
        
        # Ranking by combined score (normalized)
        ancombc_scores = -np.log10(ancombc_result.da_table.loc[list(ancombc_features), 'q_value'])
        elasticnet_scores = elasticnet_result['feature_importances']
        
        combined_score = {}
        for feature in consensus:
            score = 0.5 * (ancombc_scores[feature] if feature in ancombc_features else 0) + \
                    0.5 * elasticnet_scores[
                        list(elasticnet_result['feature_names']).index(feature)
                    ] if feature in elasticnet_features else 0
            combined_score[feature] = score
        
        self.logger_obj.info(
            f"✅ Selected {len(consensus)} consensus features from:\n"
            f"   • ANCOM-BC: {len(ancombc_features)} features\n"
            f"   • ElasticNet: {len(elasticnet_features)} features"
        )
        
        return {
            'consensus_features': sorted(
                consensus, 
                key=lambda f: combined_score[f],
                reverse=True
            ),
            'ancombc_features': ancombc_features,
            'elasticnet_features': elasticnet_features,
            'combined_scores': combined_score,
        }
