from workflow_16s.utils.analysis import AnalysisUtils
"""
🔬 16S Machine Learning Discovery Architecture
Tiered Forensic Integrity Suite

THE FORENSIC STRATEGY MATRIX
-----------------------------------------------------------------------------------------
1. ENSEMBLE DISCOVERY (Triple Consensus / Cassandra)
   - Logic:   Runs CatBoost (Non-linear), Elastic Net (Linear), and Random Forest (Cassandra).
   - Purpose: Features are kept ONLY if verified by all three mathematical families.
              Reduces false positives from model-specific biases.

2. BATCH_AWARE & CONQUR (Technical Correction & BPS-RF Audit)
   - Input:   Microbiome (Batch-Centered CLR or ConQuR) + Batch ID (Encoded).
   - Logic:   Subtracts the "Lab Effect". Validated strictly via Batch Prediction Score (BPS-RF).

3. META_AWARE (Omniscient Context)
   - Input:   Microbiome + Environmental Metadata (Encoded).
   - Logic:   Checks if environment (pH, Temp) explains the target better than biology.

4. LOPOCV (Generalization Audit)
   - Input:   Microbiome (Global CLR).
   - Logic:   Train on N-1 studies, Test on 1 unseen study.

5. SPATIAL_CV (Geographic Audit)
   - Input:   Microbiome (Global CLR).
   - Logic:   Clusters samples by Lat/Lon; Test on unseen geographic blocks.
-----------------------------------------------------------------------------------------
"""

import json
import warnings
import pandas as pd
import numpy as np
import scipy.sparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- SYSTEM UTILS ---
from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.utils import AnalysisUtils, filter_by_prevalence

# --- ML CORE ENGINES ---
from workflow_16s.downstream.machine_learning.feature_selection.core import catboost_feature_selection
from workflow_16s.downstream.machine_learning.utils import (
    clean_feature_names, resolve_feature_names, align_data_robust, 
    apply_batch_centered_clr, verify_model_outputs
)
from workflow_16s.downstream.machine_learning.constants import MANDATORY_METADATA

# --- BATCH CONTROL & COVARIATES ---
from workflow_16s.downstream.machine_learning.batch_control.batch_control import (
    audit_biomarker_confidence, 
    create_confounding_heatmap
)
from workflow_16s.downstream.machine_learning.batch_control.covariates import (
    prepare_batch_covariates, 
    calculate_batch_importance
)
from workflow_16s.downstream.stats.batch_correction import conqur_batch_correction as apply_conqur_correction

# --- ELIGIBILITY & PREPROCESSING ---
from workflow_16s.downstream.machine_learning.feature_selection.validation import (
    EligibilityManager,
    filter_data
)
from workflow_16s.downstream.machine_learning.modular_preprocessing import (
    compose_feature_matrix,
    validate_numeric_dtype
)

# --- VALIDATION TIER ---
from workflow_16s.downstream.machine_learning.validation.check_study_eligibility import StudyEligibilityManager
from workflow_16s.downstream.machine_learning.validation.overfitting_prevention import run_comprehensive_validation
from workflow_16s.downstream.machine_learning.validation.validation import (
    run_shuffle_baseline, 
    validate_consensus_panel
)
from workflow_16s.downstream.machine_learning.validation.quality_audit import (
    BiomarkerAuditor, 
    verify_run
)

# --- VISUALIZATION TIER ---
from workflow_16s.downstream.machine_learning.visualization.features import generate_comprehensive_ml_report
from workflow_16s.downstream.machine_learning.visualization.batch_dependency import plot_batch_dependency

# --- PHASE 8: COMPREHENSIVE VISUALIZATION PIPELINE ---
from workflow_16s.downstream.visualization.integration_guide import VisualizationPipeline
from workflow_16s.downstream.visualization.feature_plots import (
    plot_feature_correlation_heatmap,
    plot_feature_importance_bars,
)

# --- META-ANALYSIS ---
from workflow_16s.downstream.machine_learning.meta_analysis import (
    perform_meta_analysis, 
    apply_meta_consensus_weighting
)

logger = get_logger("workflow_16s")


# ============================================================================
# PIPELINE VARIANT: ASV-FIRST (No Prior Aggregation)
# ============================================================================

def run_asv_first_pipeline(
    adata: Any,
    output_base_dir: Union[str, Path],
    targets: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    cv_strategy_threshold: int = 10,
    auto_fix_compositionality: bool = False,
    eligibility_mode: str = 'raw',
    **kwargs: Any
) -> Dict[str, Any]:
    """
    ASV-FIRST PIPELINE:
    Analyzes at ASV (native) level without prior aggregation.
    QC filtering happens at the finest (ASV) granularity.
    
    Workflow:
    1. Input: ASV-level data
    2. QC Filtering: At ASV level (removes low-quality sequences)
    3. Aggregation: NONE (keeps native resolution)
    4. Transformation: CLR
    5. Analysis: ML at ASV level
    
    Returns: Dict with results under "asv_first" key
    """
    logger.info("🔬 Starting ASV-FIRST PIPELINE (No aggregation)")
    
    output_dir = Path(output_base_dir) / "asv_first" if kwargs.get('separate_output_dirs', True) else Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"   Input data: {adata.n_obs} samples × {adata.n_vars} features (ASV level)")
    logger.info(f"   Pipeline strategy: No aggregation, analyze native ASV resolution")
    logger.info(f"   Output directory: {output_dir}")
    
    # Filter kwargs: remove ALL orchestration-level parameters that run_catboost_selection doesn't accept
    orchestration_params = {'separate_output_dirs', 'pipeline_variants', 'telemetry', 'grid_cfg', 'n_cpus', 'output_base_dir', 'ml_config'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in orchestration_params}
    
    # Call main orchestrator with level="ASV"
    results = run_catboost_selection(
        adata=adata,
        catboost_output_dir=output_dir,
        level='ASV',  # Force ASV level (native, no aggregation)
        priority_targets=targets,
        strategies=strategies,
        cv_strategy_threshold=cv_strategy_threshold,
        auto_fix_compositionality=auto_fix_compositionality,
        eligibility_mode=eligibility_mode,
        **filtered_kwargs
    )
    
    return {"asv_first": results, "pipeline_variant": "asv_first", "output_dir": str(output_dir)}


# ============================================================================
# PIPELINE VARIANT: AGGREGATE-FIRST (Pre-Aggregation)
# ============================================================================

def run_aggregate_first_pipeline(
    adata: Any,
    output_base_dir: Union[str, Path],
    aggregation_level: str = 'Genus',
    skip_prevalence_filter: bool = False,
    targets: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    cv_strategy_threshold: int = 10,
    auto_fix_compositionality: bool = False,
    eligibility_mode: str = 'raw',
    **kwargs: Any
) -> Dict[str, Any]:
    """
    AGGREGATE-FIRST PIPELINE:
    Aggregates to configured taxonomy level BEFORE QC filtering.
    Reduces feature space early (10K ASVs → ~500 Genus).
    
    Workflow:
    1. Input: ASV-level data
    2. Aggregation: ASVs → Taxonomy level (Genus, Family, etc.)
    3. QC Filtering: At aggregated level (removes low-abundance taxa)
    4. Transformation: CLR
    5. Analysis: ML at aggregated level
    
    Advantages:
    - Faster QC (smaller feature space)
    - Biologically grouped features
    
    Disadvantages:
    - Loses fine-grained sequence resolution
    - May miss strain-level effects
    
    Returns: Dict with results under "aggregate_first" key
    """
    logger.info(f"📊 Starting AGGREGATE-FIRST PIPELINE (Pre-aggregate to {aggregation_level})")
    
    output_dir = Path(output_base_dir) / "aggregate_first" if kwargs.get('separate_output_dirs', True) else Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"   Input data: {adata.n_obs} samples × {adata.n_vars} features (ASV level)")
    logger.info(f"   Step 1: Aggregate to {aggregation_level} level...")
    
    # Aggregate to target level FIRST
    adata_agg = AnalysisUtils.get_analysis_adata(adata, level=aggregation_level)
    
    if adata_agg is None or adata_agg.n_vars == 0:
        logger.error(f"❌ Aggregation to {aggregation_level} failed or returned 0 features")
        return {"aggregate_first": [], "pipeline_variant": "aggregate_first", "error": "Aggregation failed"}
    
    logger.info(f"   Step 2: After aggregation: {adata_agg.n_obs} samples × {adata_agg.n_vars} features ({aggregation_level} level)")
    logger.info(f"   Step 3: Running ML analysis on aggregated data...")
    logger.info(f"   Output directory: {output_dir}")
    
    # Filter kwargs: remove ALL orchestration-level parameters that run_catboost_selection doesn't accept
    orchestration_params = {'separate_output_dirs', 'pipeline_variants', 'telemetry', 'grid_cfg', 'n_cpus', 'output_base_dir', 'ml_config'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in orchestration_params}
    
    # Call main orchestrator with pre-aggregated data
    results = run_catboost_selection(
        adata=adata_agg,
        catboost_output_dir=output_dir,
        level=aggregation_level,
        priority_targets=targets,
        strategies=strategies,
        cv_strategy_threshold=cv_strategy_threshold,
        auto_fix_compositionality=auto_fix_compositionality,
        eligibility_mode=eligibility_mode,
        **filtered_kwargs
    )
    
    return {"aggregate_first": results, "pipeline_variant": "aggregate_first", "output_dir": str(output_dir), "aggregation_level": aggregation_level}


# ============================================================================
# MAIN ORCHESTRATOR: Respects Config Choices
# ============================================================================

def run_selection_with_config(
    adata: Any,
    output_base_dir: Union[str, Path],
    ml_config: Any,  # MLConfig object from config_schema
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Main entry point that respects all ML config parameters.
    
    Branches to appropriate pipeline(s) based on:
    - ml.preprocessing.pipeline_variants (asv_first, aggregate_first, or both)
    - ml.preprocessing.separate_output_dirs (organize outputs)
    - ml.grid_settings.levels (aggregation levels for aggregate_first)
    - ml.targets, ml.validation, ml.models (passed to both pipelines)
    
    This ensures ALL config parameters are respected at every step.
    """
    logger.info("="*70)
    logger.info("🎯 UNIFIED ML PIPELINE ORCHESTRATOR")
    logger.info("="*70)
    
    # Extract config settings
    preproc_cfg = getattr(ml_config, 'preprocessing', None)
    pipeline_variants = getattr(preproc_cfg, 'pipeline_variants', ['asv_first']) if preproc_cfg else ['asv_first']
    separate_dirs = getattr(preproc_cfg, 'separate_output_dirs', True) if preproc_cfg else True
    grid_cfg = getattr(ml_config, 'grid_settings', None)
    levels = getattr(grid_cfg, 'levels', ['Genus']) if grid_cfg else ['Genus']
    targets = getattr(ml_config, 'targets', kwargs.get('targets', []))
    strategies = getattr(grid_cfg, 'fs_strategies', []) if grid_cfg else []
    cv_strategy_threshold = getattr(grid_cfg, 'cv_strategy_threshold', 10) if grid_cfg else 10
    
    # Extract validation config (applies to all pipelines)
    val_cfg = getattr(ml_config, 'validation', None)
    auto_fix_compositionality = getattr(val_cfg, 'auto_fix_compositionality', False) if val_cfg else False
    eligibility_mode = getattr(ml_config, 'eligibility_mode', 'raw')  # NEW: Read from config (default 'raw' to preserve training data)
    
    logger.info(f"\n📋 CONFIG PARAMETERS BEING RESPECTED:")
    logger.info(f"   • Pipeline variants: {pipeline_variants}")
    logger.info(f"   • Separate output dirs: {separate_dirs}")
    logger.info(f"   • Aggregation levels (for aggregate_first): {levels}")
    logger.info(f"   • ML targets: {targets}")
    logger.info(f"   • Feature selection strategies: {strategies}")
    logger.info(f"   • CV strategy threshold: {cv_strategy_threshold} projects (LOPO→GroupKFold switch point)")
    logger.info(f"   • Auto-fix compositionality: {auto_fix_compositionality}")
    logger.info(f"   • Eligibility filtering mode: {eligibility_mode}")  # NEW: Log eligibility mode
    logger.info(f"\n🔄 EXECUTION FLOW:")
    
    # Filter orchestration-level kwargs that shouldn't be passed to pipeline functions
    # Keep only ML-relevant parameters
    orchestration_level_params = {'n_cpus', 'telemetry', 'ml_config', 'output_base_dir', 'targets'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in orchestration_level_params}
    
    all_results = {}
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Run selected pipeline(s)
    if 'asv_first' in pipeline_variants:
        logger.info("\n[1/2] Executing ASV-FIRST pipeline...")
        result = run_asv_first_pipeline(
            adata=adata,
            output_base_dir=output_base,
            targets=targets,
            strategies=strategies,
            separate_output_dirs=separate_dirs,
            cv_strategy_threshold=cv_strategy_threshold,
            auto_fix_compositionality=auto_fix_compositionality,
            eligibility_mode=eligibility_mode,
            **filtered_kwargs
        )
        all_results['asv_first'] = result
        logger.info(f"   ✅ ASV-FIRST complete. Results: {len(result.get('asv_first', []))} discoveries")
    
    if 'aggregate_first' in pipeline_variants:
        # For aggregate_first, run once per level
        for level in levels:
            logger.info(f"\n[2/2] Executing AGGREGATE-FIRST pipeline (level: {level})...")
            result = run_aggregate_first_pipeline(
                adata=adata,
                output_base_dir=output_base,
                aggregation_level=level,
                targets=targets,
                strategies=strategies,
                separate_output_dirs=separate_dirs,
                cv_strategy_threshold=cv_strategy_threshold,
                auto_fix_compositionality=auto_fix_compositionality,
                eligibility_mode=eligibility_mode,
                **filtered_kwargs
            )
            all_results[f'aggregate_first_{level}'] = result
            logger.info(f"   ✅ AGGREGATE-FIRST ({level}) complete. Results: {len(result.get('aggregate_first', []))} discoveries")
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ UNIFIED PIPELINE EXECUTION COMPLETE")
    logger.info(f"   Pipelines run: {', '.join(pipeline_variants)}")
    logger.info(f"   Total results: {len(all_results)}")
    logger.info("="*70)
    
    return all_results


def run_catboost_selection(
    adata: Any, 
    catboost_output_dir: Union[str, Path], 
    level: str = 'Genus', 
    priority_targets: Optional[List[str]] = None, 
    strict_targets: bool = False,
    eligibility_mode: str = 'filter', 
    strategies: Optional[List[str]] = None, 
    test_size: float = 0.3, 
    random_state: int = 42, 
    num_features: int = 50,  # Max features to keep from each discovery method
    n_top_final: int = 20,   # Final refined model feature count
    method: Literal['rfe', 'shap', 'select_k_best'] = 'shap', 
    use_permutation: bool = False, 
    n_cpus: int = 4,
    batch_col: str = 'batch_original',
    meta_cols: Optional[List[str]] = None,  
    X_custom: Optional[pd.DataFrame] = None,
    param_grid: Optional[Dict[str, Any]] = None,
    cv_strategy_threshold: int = 10,  # Switch LOPO→GroupKFold above this many projects
    auto_fix_compositionality: bool = False  # Auto-correct CLR if data is warped
) -> List[Dict[str, Any]]:
    """
    Orchestrates the 16S Forensic Pipeline: Ensemble Discovery -> Refinement -> Audit.
    
    Parameters
    ----------
    eligibility_mode : str
        How to handle study eligibility filtering:
        - 'filter' (default): Completely remove ineligible studies (single class, small N)
        - 'two_tier': Use ALL studies for training, but only PASS studies for testing
                      (maximizes training data, keeps test set clean)
        - 'audit_only': Print report but don't filter data
    """
    logger.info(f" 🎬 Starting Forensic Discovery Engine ({level})")
    
    out_dir = Path(catboost_output_dir)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # PHASE 8: INITIALIZE VISUALIZATION PIPELINE (NEW)
    # ═════════════════════════════════════════════════════════════════════════════
    viz_pipeline = VisualizationPipeline(
        output_dir=out_dir / 'visualizations',
        create_png=False  # Set to True if Kaleido available
    )
    logger.info(f" 📊 Visualization pipeline initialized: {out_dir / 'visualizations'}")
    
    # CRITICAL: Perform eligibility check BEFORE aggregation to avoid index mismatch
    # This ensures filtered data at sample level can be properly aggregated
    logger.info(f" 🔍 Performing pre-aggregation eligibility check...")
    
    # Identify valid targets first
    defaults = ['baseline', 'batch_aware', 'conqur', 'meta_aware', 'lopocv', 'spatial_cv']
    if strategies:
        mapped_strategies = []
        for s in strategies:
            if s == 'agnostic': mapped_strategies.append('baseline')
            elif s == 'batch_adjusted': mapped_strategies.append('batch_aware')
            elif s == 'spatial': mapped_strategies.append('spatial_cv')
            else: mapped_strategies.append(s)
        strategies_to_run = [s for s in mapped_strategies if s in defaults]
        strategies_to_run = list(dict.fromkeys(strategies_to_run))
    else:
        strategies_to_run = defaults
    
    valid_targets = _generate_facility_targets(adata, priority_targets, strict_targets)
    logger.info(f" CHECKPOINT: About to check {len(valid_targets)} targets for eligibility")
    
    # Pre-filter data based on target eligibility (at sample level, before aggregation)
    adata_filtered = adata.copy()
    for target_col in valid_targets:
        eligibility_pre = StudyEligibilityManager(adata, target_col=target_col)
        eligibility_pre.diagnose_studies()
        
        # Apply eligibility filtering at the sample level BEFORE aggregation
        if eligibility_mode == 'filter':
            adata_filtered = eligibility_pre.get_filtered_adata()
            logger.info(f" 🔍 Pre-aggregation filtering for {target_col}: {len(adata_filtered)} eligible samples")
            break  # Just use the first target for initial filtering (will be re-checked per-target later)
    
    # 1. Feature Engineering (Global Baseline) - NOW FROM FILTERED DATA
    adata_agg = AnalysisUtils.get_analysis_adata(adata_filtered, level=level)
    if adata_agg is None or adata_agg.n_obs < 20: 
        n_samples = adata_agg.n_obs if adata_agg is not None else 0
        logger.warning(f"⚠️  Insufficient data for ML: {level} aggregation requires ≥20 samples after filtering, have {n_samples}. Skipping ML.")
        return []
        
    if X_custom is None:
        if scipy.sparse.issparse(adata_agg.X):
            raw_counts = np.asarray(adata_agg.X.toarray()) # type: ignore
        elif hasattr(adata_agg.X, 'toarray'):
            raw_counts = np.asarray(adata_agg.X.toarray()) # type: ignore
        else:
            raw_counts = np.asarray(adata_agg.X) if isinstance(adata_agg.X, np.ndarray) else np.array(adata_agg.X)
        feature_names = resolve_feature_names(adata_agg, level)
        raw_counts_df = pd.DataFrame(raw_counts, index=adata_agg.obs_names, columns=feature_names)
        
        # Standard CLR
        X_df = AnalysisUtils.clr_transform_from_df(raw_counts_df, pseudocount=1.0)
    else:
        X_df = X_custom.copy()
        
    X_df = clean_feature_names(X_df)
    logger.info(f" ✅ Feature matrix created from aggregated & filtered data: {X_df.shape[0]} samples × {X_df.shape[1]} features")
    
    # Now that we have X_df from filtered aggregated data, re-validate targets
    valid_targets = _generate_facility_targets(adata_filtered, priority_targets, strict_targets)
    logger.info(f" CRITICAL: valid_targets generated - count={len(valid_targets)}, values={valid_targets}, priority={priority_targets}")
    
    all_results = []
    logger.info(f" 🔍 CRITICAL DEBUG: About to enter target loop. valid_targets={valid_targets}, len={len(valid_targets)}")
    
    for target_col in valid_targets:
        logger.info(f" 📍 Processing Forensic Target: {target_col}")
        
        # Eligibility check (already pre-filtered at sample level, now re-aggregate at target level)
        eligibility = StudyEligibilityManager(adata_filtered, target_col=target_col)
        eligibility.diagnose_studies()
        logger.info(f" 🔍 [DEBUG] After diagnose_studies on pre-filtered data. eligibility_mode={eligibility_mode}")
        
        # Re-aggregate the pre-filtered sample-level data to the aggregated level for this target
        # This ensures X_df (aggregated) and adata_working (aggregated) have matching indices
        adata_working = AnalysisUtils.get_analysis_adata(adata_filtered, level=level)
        if adata_working is None or adata_working.n_obs < 20:
            logger.warning(f" ⚠️ After aggregation, insufficient samples for target '{target_col}': {adata_working.n_obs if adata_working else 0} < 20. Skipping.")
            continue
        
        logger.info(f" ✅ Re-aggregated data for {target_col}: {adata_working.n_obs} samples × {adata_working.n_vars} features")
        adata_test_filtered = None  # For two-tier filtering
        
        # Skip eligibility filtering if no suitable batch column exists
        has_batch_col = any(col in adata_filtered.obs.columns for col in ['batch_original', 'study_accession', 'Project', 'dataset', 'study', 'batch', 'project_id', 'study_id'])
        
        # TWO-TIER ELIGIBILITY FILTERING (NEW)
        # Use all studies for training, only PASS studies for test/validation
        if eligibility_mode == 'two_tier' and has_batch_col:
            logger.info(f" 🔄 Two-tier eligibility filtering: input {len(adata_filtered)} samples")
            adata_working, adata_test_filtered = eligibility.get_two_tier_split()
            logger.info(f" 🔄 Two-tier split complete:")
            logger.info(f"    └─ Train (all studies):  {len(adata_working)} samples")
            logger.info(f"    └─ Test (PASS only):    {len(adata_test_filtered)} samples")
            if len(adata_test_filtered) < 20:
                logger.warning(f" ⚠️ Insufficient test samples after eligibility filtering: {len(adata_test_filtered)} < 20. Skipping target '{target_col}'.")
                continue
        
        # STANDARD FILTERING (existing behavior)
        elif eligibility_mode == 'filter' and has_batch_col:
            logger.info(f" 🔍 Standard eligibility filtering: input {len(adata_filtered)} samples")
            adata_working = eligibility.get_filtered_adata()
            logger.info(f" 🔍 Standard filtering: output {len(adata_working)} samples")
            if len(adata_working) < 20:
                logger.warning(f" ⚠️ Insufficient samples for ML after eligibility filtering: {len(adata_working)} < 20. Skipping target '{target_col}'.")
                continue
        elif eligibility_mode == 'filter' and not has_batch_col:
            logger.info(f" ℹ️ Eligibility filtering skipped: no batch column found in metadata. Proceeding with all {len(adata_filtered)} samples.")
        
        # CRITICAL FIX: Re-aggregate adata_working to the same level as X_df
        # This ensures indices match perfectly after filtering
        adata_working_agg = AnalysisUtils.get_analysis_adata(adata_working, level=level)
        if adata_working_agg is None or adata_working_agg.n_obs < 10:
            logger.warning(f" ⚠️ After eligibility filtering and re-aggregation: {level} has {adata_working_agg.n_obs if adata_working_agg else 0} samples. Insufficient for ML (need ≥10). Skipping target '{target_col}'.")
            continue
            
        # Re-create X_df from filtered aggregated data
        if scipy.sparse.issparse(adata_working_agg.X):
            raw_counts = np.asarray(adata_working_agg.X.toarray())
        elif hasattr(adata_working_agg.X, 'toarray'):
            raw_counts = np.asarray(adata_working_agg.X.toarray())
        else:
            raw_counts = np.asarray(adata_working_agg.X) if isinstance(adata_working_agg.X, np.ndarray) else np.array(adata_working_agg.X)
        
        feature_names = resolve_feature_names(adata_working_agg, level)
        raw_counts_df = pd.DataFrame(raw_counts, index=adata_working_agg.obs_names, columns=feature_names)
        X_df_aligned = AnalysisUtils.clr_transform_from_df(raw_counts_df, pseudocount=1.0)
        X_df_aligned = clean_feature_names(X_df_aligned)
        logger.info(f" 🔄 Re-aggregated X_df: {X_df_aligned.shape[0]} samples × {X_df_aligned.shape[1]} features from {level} level")
        
        # Use aggregated data for metadata alignment
        adata_working = adata_working_agg
        
        # Consensus Weighting Setup
        meta_weight_map = {}
        logger.info(f" 🔍 [DEBUG] Before meta_analysis check. 'meta_analysis' in strategies={('meta_analysis' in strategies_to_run)}")
        if 'meta_analysis' in strategies_to_run:
            meta_out = out_dir / "meta_analysis" / f"{level}_{target_col}"
            X_meta, y_meta, meta_aligned = align_data_robust(X_df_aligned, adata_working.obs, target_col)
            meta_res = perform_meta_analysis(
                X_meta, 
                y_meta, 
                meta_aligned['study_accession'].values, 
                meta_out
            )
            meta_weight_map = meta_res.get('feature_importance_map', {})
        logger.info(f" 🔍 [DEBUG] After meta_analysis block.")

        # Strategy Loop
        logger.info(f" 🔍 [DEBUG] About to create modeling_strats. strategies_to_run={strategies_to_run}")
        modeling_strats = [s for s in strategies_to_run if s != 'meta_analysis']
        logger.info(f" 🔄 Starting strategy loop with {len(modeling_strats)} strategies: {modeling_strats}")
        
        # ===================================================================
        # CRITICAL FIX: ELIGIBILITY FILTERING (BEFORE strategy loop)
        # ===================================================================
        # Apply EligibilityManager to get eligible sample indices
        # This ensures training data is NEVER filtered to zero samples
        # (Problem: old code filtered AFTER train/test split, breaking training set)
        
        logger.info(f" 🔍 ELIGIBILITY CHECK: Applying dynamic threshold reduction for target '{target_col}'")
        eligibility_mgr = EligibilityManager(
            data_df=adata_working.obs,
            target_col=target_col,
            study_col=batch_col if batch_col in adata_working.obs.columns else 'Project',
            start_threshold=None,  # Auto-detect max classes
            min_threshold=2,
            min_samples_for_training=50,
            test_size=0.2,
            task_type='Regression' if pd.api.types.is_numeric_dtype(adata_working.obs[target_col]) else 'Classification'
        )
        
        eligible_indices = eligibility_mgr.get_eligible_samples()
        
        if len(eligible_indices) == 0:
            logger.error(f" ❌ No eligible samples found for target '{target_col}'. Skipping.")
            continue
        
        # Filter X_final, y_final, meta_final to eligible samples (BEFORE train/test split)
        X_eligible = X_df_aligned.loc[eligible_indices].copy()
        y_eligible = adata_working.obs.loc[eligible_indices, target_col].copy()
        meta_eligible = adata_working.obs.loc[eligible_indices].copy()
        
        logger.info(f" ✅ ELIGIBLE SAMPLES: {len(eligible_indices)}/{len(adata_working)} samples passed criteria")
        
        # Validate training adequacy after filtering
        train_size = int(len(eligible_indices) * 0.8)  # Rough estimate
        if train_size < 50:
            logger.warning(
                f" ⚠️ WARNING: Estimated training set too small ({train_size} samples after split). "
                f"Consider lowering min_samples_for_training threshold."
            )
        
        # ═════════════════════════════════════════════════════════════════════════════
        # VISUALIZATION: SAMPLE GEOGRAPHY & METADATA (NEW - Phase 8)
        # ═════════════════════════════════════════════════════════════════════════════
        try:
            if 'lat' in meta_eligible.columns and 'lon' in meta_eligible.columns:
                logger.info(f" 🗺️  Generating sample geography visualization for {target_col}...")
                
                # Approximate train/test split
                train_indices = np.array(sorted(np.random.choice(eligible_indices, size=train_size, replace=False)))
                test_indices = np.array([idx for idx in eligible_indices if idx not in train_indices])
                
                viz_pipeline.visualize_samples(
                    metadata=meta_eligible.copy(),
                    lat_col='lat',
                    lon_col='lon',
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
        except Exception as e:
            logger.warning(f" ⚠️ Sample visualization failed for {target_col}: {e}")
        
        for strategy in modeling_strats:
            logger.info(f" ➡️ Entering strategy loop iteration for strategy: {strategy}")
            target_out = out_dir / strategy / f"{level}_{target_col}"
            target_out.mkdir(exist_ok=True, parents=True)

            # ===================================================================
            # STRATEGY-SPECIFIC FEATURE COMPOSITION (NEW)
            # ===================================================================
            # Use modular preprocessing to compose features per strategy
            # instead of hardcoded branches
            
            try:
                # Get curated environment columns for strategies that need them
                env_cols = None
                if strategy in ['meta_aware', 'spatial_cv']:
                    env_cols = ['elevation', 'temperature', 'ph']  # Curated minimal set
                
                # Compose features per strategy
                X_composed = compose_feature_matrix(
                    X_base=X_eligible.copy(),
                    metadata=meta_eligible,
                    strategy_name=strategy,
                    study_col=batch_col if batch_col in meta_eligible.columns else 'Project',
                    env_columns=env_cols
                )
                
                # Validate numeric dtypes for CatBoost
                X_composed = validate_numeric_dtype(X_composed, allow_categories=False)
                
                logger.info(f" ✅ Feature composition ({strategy}): {X_composed.shape[1]} features total")
                
            except Exception as e:
                logger.error(f" ❌ Feature composition failed for strategy '{strategy}': {e}")
                continue
            
            # ===================================================================
            # ALIGNMENT WITH ELIGIBLE DATA (BEFORE train/test split)
            # ===================================================================
            X_run = X_composed.copy()
            X_final, y_final, meta_final = align_data_robust(X_run, meta_eligible, target_col)
            
            # Setup batch column for CV strategies
            possible_batch_cols = ['batch_original', 'study_accession', 'Project', 'dataset', 'study', 'batch', 'project_id', 'study_id']
            current_batch_col = batch_col if batch_col in meta_eligible.columns else None
            
            if current_batch_col is None:
                for col in possible_batch_cols:
                    if col in meta_eligible.columns:
                        current_batch_col = col
                        break
            
            if current_batch_col not in meta_final.columns and current_batch_col in meta_eligible.columns:
                meta_final[current_batch_col] = meta_eligible.loc[X_final.index, current_batch_col]
            
            if current_batch_col in meta_final.columns:
                meta_final[current_batch_col] = meta_final[current_batch_col].astype(object).fillna("unknown").astype(str)
            
            logger.info(f" 🎯 Strategy: {strategy.upper()} - {X_final.shape[0]} eligible samples, {X_final.shape[1]} features")
            
            # --- RARE CLASS FILTERING ---
            # Additional safety check for strategies that require multiple classes per batch
            # CRITICAL FIX: Check if filtering would remove >80% of samples BEFORE applying
            if strategy in ['lopocv', 'batch_aware', 'conqur', 'spatial_cv'] and current_batch_col in meta_final.columns:
                batch_source = meta_final[current_batch_col]
                temp_df = pd.DataFrame({'target': y_final.values, 'batch': batch_source.values}, index=y_final.index)
                class_batch_counts = temp_df.groupby('target')['batch'].nunique()
                valid_classes = class_batch_counts[class_batch_counts >= 2].index.tolist()
                
                if len(valid_classes) < len(class_batch_counts):
                    dropped = set(class_batch_counts.index) - set(valid_classes)
                    
                    # PHASE 1 FIX: Calculate removal percentage BEFORE filtering
                    samples_after_filtering = y_final.isin(valid_classes).sum()
                    removal_percentage = 1.0 - (samples_after_filtering / len(y_final))
                    
                    # GUARDRAIL: Skip filtering if it would remove >80% or leave <20 samples
                    if removal_percentage > 0.80:
                        logger.warning(
                            f" ⚠️ RARE CLASS FILTER SKIPPED: Would remove {100*removal_percentage:.1f}% of samples "
                            f"({len(y_final) - samples_after_filtering}/{len(y_final)}). "
                            f"Threshold: 80% max removal. Classes {dropped} are kept despite single-batch origin."
                        )
                    elif samples_after_filtering < 20:
                        logger.warning(
                            f" ⚠️ RARE CLASS FILTER SKIPPED: Would leave only {samples_after_filtering} samples (<20 minimum). "
                            f"Classes {dropped} are kept to preserve training data."
                        )
                    else:
                        logger.warning(f" ⚠️ Dropping rare classes (unique to single batch): {dropped}")
                        valid_mask = y_final.isin(valid_classes)
                        X_final = X_final[valid_mask].copy()
                        y_final = y_final[valid_mask].copy()
                        meta_final = meta_final[valid_mask].copy()
                    
                    # Check if anything remains after filtering (for strategies that WERE filtered)
                    if len(X_final) < 20:
                        if strategy == 'lopocv':
                            logger.warning(f" ⏭️ SKIPPING LOPOCV: Requires ≥2 batches per class, but only {len(X_final)} samples remain (min: 20). Data insufficient for Leave-One-Project-Out CV.")
                        else:
                            logger.warning(f" ⏭️ SKIPPING {strategy.upper()}: Only {len(X_final)} samples available. Minimum required: 20.")
                        continue
            
            # Log what we're processing
            logger.info(f" 🎯 Strategy: {strategy.upper()} - {X_final.shape[0]} samples, {X_final.shape[1]} features")
            
            if strategy == 'batch_aware':
                logger.info(f" 🎯 Strategy: {strategy.upper()} - Centering CLR + Injecting Batch ID")
                X_final = apply_batch_centered_clr(X_final, meta_final[current_batch_col].astype(str))
                y_final = y_final.loc[X_final.index]
                meta_final = meta_final.loc[X_final.index]
                
                if current_batch_col in meta_final.columns:
                    # Forensic Audit: BPS-RF
                    _calculate_bps_rf(X_final, meta_final[current_batch_col].astype(str))
                    batch_codes, _ = pd.factorize(meta_final[current_batch_col].astype(str))
                    batch_df = pd.DataFrame({current_batch_col: batch_codes}, index=meta_final.index)
                    X_final = pd.concat([X_final, batch_df], axis=1)

            elif strategy == 'conqur':
                logger.info(f" 🎯 Strategy: CONQUR - Conditional Quantile Regression Batch Correction")
                
                # ConQuR requires raw counts. We extract the exact subset of samples that passed filtering.
                adata_sub = adata_working[X_final.index].copy()
                
                try:
                    # Pass raw counts into the R interoperability layer
                    adata_corrected = apply_conqur_correction(
                        adata=adata_sub,
                        batch_col=current_batch_col,
                        covariate_cols=None, 
                        output_dir=target_out
                    )
                    
                    # Convert corrected outputs back into Euclidean space (CLR) for ML pipelines
                    corrected_counts = np.asarray(adata_corrected.X.toarray() if hasattr(adata_corrected.X, 'toarray') else adata_corrected.X)
                    corrected_df = pd.DataFrame(corrected_counts, index=adata_corrected.obs_names, columns=resolve_feature_names(adata_corrected, level))
                    
                    X_final = AnalysisUtils.clr_transform_from_df(corrected_df, pseudocount=1.0)
                    X_final = clean_feature_names(X_final)
                    
                    # Align one final time
                    X_final, y_final, meta_final = align_data_robust(X_final, adata_sub.obs, target_col)
                    
                    if current_batch_col in meta_final.columns:
                        logger.info(" 🛡️ Auditing ConQuR Correction via BPS-RF...")
                        _calculate_bps_rf(X_final, meta_final[current_batch_col].astype(str))
                        
                except Exception as e:
                    logger.warning(f" ⚠️ ConQuR R-Bridge failed: {e}. Falling back to robust batch CLR centering.")
                    X_final = apply_batch_centered_clr(X_final, meta_final[current_batch_col].astype(str))

            elif strategy == 'meta_aware':
                logger.info(" 💉 Strategy: META_AWARE - Injecting Environmental Context")
                
                allowed_exact = [
                    'env_biome', 'env_feature', 'env_material', 
                    'ph', 'temperature', 'salinity', 'elevation', 'depth',
                    'water_content', 'organic_carbon_percent', 'clay_percent', 
                    'sand_percent', 'silt_percent'
                ]
                allowed_prefixes = ('SoilGrids_', 'Meteostat_', 'OpenMeteo_')
                
                meta_cols_to_add = [
                    col for col in adata_working.obs.columns 
                    if (col in allowed_exact or col.startswith(allowed_prefixes)) 
                    and col != target_col
                ]
                
                logger.info(f" ✅ Injecting {len(meta_cols_to_add)} strictly filtered environmental features...")
                X_meta = adata_working.obs.loc[X_final.index, meta_cols_to_add].copy()
                
                for col in X_meta.columns:
                    if X_meta[col].dtype == 'object' or isinstance(X_meta[col].dtype, pd.CategoricalDtype):
                        # 🚨 FIX: Convert to string first, fill NaN, then convert to category
                        # This avoids pandas categorical constraint errors
                        X_meta[col] = X_meta[col].astype(str).fillna("unknown").astype('category')
                    else:
                        X_meta[col] = pd.to_numeric(X_meta[col], errors='coerce').fillna(0)
                        
                X_final = pd.concat([X_final, X_meta], axis=1)
            
            elif strategy == 'spatial_cv':
                logger.info(f" 🗺️ Strategy: {strategy.upper()} - Spatial Clustering")
                if 'latitude' in meta_final and 'longitude' in meta_final:
                    coords = meta_final[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce').fillna(0)
                    n_samples = len(coords)
                    n_folds = max(2, min(10, int(n_samples / 50))) 
                    
                    try:
                        kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init='auto')
                        labels = kmeans.fit_predict(coords)
                        meta_final['spatial_block'] = [f"Block_{i}" for i in labels]
                        current_batch_col = 'spatial_block' 
                        logger.info(f"   -> Created {n_folds} spatial blocks.")
                    except Exception as e:
                        logger.error(f" ❌ Spatial clustering failed: {e}")
            
            # --- Cross-Validation Setup ---
            cv_groups, cv_strat, cv_name = _setup_cv_strategy(
                strategy, meta_final, current_batch_col, 
                cv_strategy_threshold=cv_strategy_threshold
            )
            if strategy == 'baseline': cv_groups = None
            
            task_type = 'Regression' if pd.api.types.is_numeric_dtype(y_final) and y_final.nunique() > 10 else 'Classification'
            
            try:
                # =========================================================
                # PHASE 1: ENSEMBLE DISCOVERY (Triple Consensus / Cassandra)
                # =========================================================
                logger.info(f" 🔍 [Phase 01] Ensemble Discovery: Scanning {X_final.shape[1]} features...")
                
                # Prepare test_indices for two-tier filtering if applicable
                test_indices_for_split = None
                if adata_test_filtered is not None:
                    test_indices_for_split = adata_test_filtered.obs_names.tolist()
                
                # A. CatBoost Discovery
                cb_grid = param_grid or {'depth': [4, 6, 8], 'learning_rate': [0.05, 0.1], 'l2_leaf_reg': [3, 5]}
                cb_res = catboost_feature_selection(
                    metadata=meta_final, features=X_final, output_dir=target_out / "discovery_catboost",
                    group_col=target_col, cv_groups=cv_groups, cv_strategy=cv_strat,
                    method=method, num_features=num_features, n_top_final=num_features, 
                    task_type=task_type, random_state=random_state, thread_count=n_cpus, param_grid=cb_grid,
                    test_indices=test_indices_for_split,
                    auto_fix_compositionality=auto_fix_compositionality
                )
                cb_top = set(cb_res['top_features'])
                
                # B. Elastic Net Consensus
                logger.info(" 🥅 Running Linear Consensus (Elastic Net)...")
                linear_top = _run_linear_discovery(X_final, y_final, task_type, num_features)

                # C. Cassandra Consensus (Random Forest)
                logger.info(" 🌲 Running Cassandra Consensus (Random Forest)...")
                rf_top = _run_rf_discovery(X_final, y_final, task_type, num_features)
                
                # D. Consensus Logic
                intersection = list(cb_top.intersection(linear_top).intersection(rf_top))
                
                if len(intersection) >= 5:
                    final_candidates = intersection[:n_top_final]
                    logger.info(f" ✅ Triple Consensus: Found {len(intersection)} features across CatBoost, ElasticNet, and RF.")
                else:
                    dual_intersection = list(cb_top.intersection(rf_top))
                    if len(dual_intersection) >= 5:
                         final_candidates = dual_intersection[:n_top_final]
                         logger.warning(f" ⚠️ Strict Triple Consensus failed. Fallback to CatBoost + Cassandra dual consensus ({len(dual_intersection)} features).")
                    else:
                         logger.warning(f" ⚠️ Low Consensus. Fallback: Using CatBoost top features.")
                         final_candidates = cb_res['top_features'][:n_top_final]

                # =========================================================
                # PHASE 2: REFINEMENT
                # =========================================================
                logger.info(f" 🛠️ [Phase 02] Refinement: Training Final Model on {len(final_candidates)} features...")
                
                X_refined = X_final[final_candidates].copy()
                refinement_grid = {
                    'depth': [2, 3, 4], 
                    'learning_rate': [0.03, 0.05, 0.1], 
                    'l2_leaf_reg': [3, 5, 9], 
                    'iterations': [500]
                }
                
                final_res = catboost_feature_selection(
                    metadata=meta_final, 
                    features=X_refined, 
                    output_dir=target_out / "refined_model",
                    group_col=target_col, 
                    cv_groups=cv_groups, 
                    cv_strategy=cv_strat,
                    num_features=len(final_candidates), 
                    n_top_final=len(final_candidates),
                    method='shap', 
                    task_type=task_type, 
                    random_state=random_state, 
                    thread_count=n_cpus, 
                    param_grid=refinement_grid,
                    test_indices=test_indices_for_split,
                    auto_fix_compositionality=auto_fix_compositionality
                )
                
                # =========================================================
                # PHASE 3: FORENSIC AUDITS
                # =========================================================
                # Check if model was successfully created
                if 'model' not in final_res:
                    logger.error(f" ❌ Strategy {strategy} failed: No valid model created. Skipping validation steps.")
                    continue
                
                # 3A. Information Leakage Audit
                _audit_information_leakage(X_refined, adata_working, final_candidates)

                # 3B. Covariate / Confounding Audit
                logger.info(" 🛡️ Running Biomarker Confidence Audit against technical covariates...")
                batch_covs, _ = prepare_batch_covariates(adata_working, [current_batch_col], X_refined.index)
                
                if not batch_covs.empty:
                    conf_df, exclusions = audit_biomarker_confidence(X_refined, batch_covs, final_candidates)
                    conf_df.to_csv(target_out / "biomarker_confidence_audit.csv", index=False)
                    create_confounding_heatmap(X_refined, batch_covs, final_candidates, target_out, target_col, level)

                if meta_weight_map and 'top_features' in final_res and 'feature_importances' in final_res:
                    global_imp = pd.DataFrame({'feature': final_res['top_features'], 'importance': final_res['feature_importances']})
                    weighted_df = apply_meta_consensus_weighting(
                        global_imp, 
                        {'feature_importance_map': meta_weight_map}
                    )
                    weighted_df.to_csv(target_out / "robustness_weighted_biomarkers.csv", index=False)
                    plot_batch_dependency(
                        global_imp, 
                        target_out / "batch_dependency_donut.html",
                        target_col
                    )

                if 'model' in final_res:
                    run_comprehensive_validation(
                        final_res['model'], 
                        X_refined, 
                        y_final, 
                        output_dir=target_out / "audit", 
                        target_name=target_col,
                        groups=cv_groups.values if cv_groups is not None else None, # type: ignore
                        task_type=task_type
                    )
                
                run_shuffle_baseline(
                    adata_working, 
                    target_col, 
                    target_out / "significance", 
                    real_score=final_res.get('best_score', 0.0), 
                    level=level
                )

                final_res['strategy'] = strategy
                
                # ═════════════════════════════════════════════════════════════════════════════
                # VISUALIZATION: FEATURE ANALYSIS (NEW - Phase 8)
                # ═════════════════════════════════════════════════════════════════════════════
                try:
                    if 'feature_importances' in final_res and 'top_features' in final_res:
                        logger.info(f" 🔬 Generating feature analysis visualization for {strategy}...")
                        
                        # Create feature importance Series
                        feature_imp = pd.Series(
                            final_res.get('feature_importances', []),
                            index=final_res.get('top_features', [])
                        ).sort_values(ascending=False)
                        
                        # Determine if target is categorical or continuous
                        is_categorical = not pd.api.types.is_numeric_dtype(y_final) or \
                                       len(np.unique(y_final)) < 10
                        
                        # Visualize features
                        viz_pipeline.visualize_features(
                            X_features=X_refined,
                            y_target=y_final,
                            feature_importance=feature_imp if not feature_imp.empty else None,
                            target_is_categorical=is_categorical,
                        )
                        
                        logger.info(f" ✅ Feature visualization complete for {strategy}")
                        
                        # CRITICAL: Close figures to prevent "too many open files" error
                        import matplotlib.pyplot as plt
                        plt.close('all')
                
                except Exception as e:
                    logger.warning(f" ⚠️ Feature visualization failed for {strategy}: {e}")
                
                all_results.append(final_res)

            except Exception as e:
                logger.error(f" ❌ Strategy {strategy} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        verify_run(out_dir, [target_col])
    
    generate_comprehensive_ml_report(
        out_dir, 
        str(out_dir / "summary_report.html"), 
        ml_targets=valid_targets, 
        strategies=strategies_to_run
    )
    
    # ═════════════════════════════════════════════════════════════════════════════
    # VISUALIZATION: FINAL SUMMARY (Phase 8)
    # ═════════════════════════════════════════════════════════════════════════════
    try:
        logger.info(viz_pipeline.generate_summary())
    except Exception as e:
        logger.warning(f" ⚠️ Visualization summary generation failed: {e}")
    
    return all_results

def _run_linear_discovery(
    X: pd.DataFrame, 
    y: pd.Series, 
    task_type: str, 
    n_top: int = 50
) -> set:
    """Runs a robust linear scan for consensus using Elastic Net."""
    logger = get_logger("workflow_16s")
    try:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns).fillna(0)
        
        if task_type == 'Classification':
            model = LogisticRegression(
                penalty='elasticnet', solver='saga', 
                l1_ratio=0.7, C=1.0, max_iter=2000,
                class_weight='balanced', n_jobs=-1, random_state=42
            )
        else:
            model = ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95, .99], cv=3, 
                max_iter=2000, n_jobs=-1, random_state=42
            )
            
        model.fit(X_scaled, y)
        
        if task_type == 'Classification':
            coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_.flatten())
        else:
            coefs = np.abs(model.coef_)
            
        non_zero_mask = coefs > 1e-5
        if non_zero_mask.sum() > 0:
            top_features = X.columns[non_zero_mask].tolist()
            if len(top_features) > n_top:
                top_idx = np.argsort(coefs)[::-1][:n_top]
                return set(X.columns[top_idx])
            return set(top_features)
        
        top_idx = np.argsort(coefs)[::-1][:n_top]
        return set(X.columns[top_idx])
        
    except Exception as e:
        logger.warning(f" ⚠️ Linear discovery failed: {e}. Returning empty set.")
        return set()

def _run_rf_discovery(
    X: pd.DataFrame, 
    y: pd.Series, 
    task_type: str, 
    n_top: int = 50
) -> set:
    """Runs a robust Cassandra (Random Forest) scan for consensus."""
    logger = get_logger("workflow_16s")
    try:
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['object', 'category']).columns:
            X_clean[col] = X_clean[col].astype('category').cat.codes

        if task_type == 'Classification':
            model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        
        model.fit(X_clean, y)
        importances = model.feature_importances_
        
        top_idx = np.argsort(importances)[::-1][:n_top]
        return set(X_clean.columns[top_idx])
    except Exception as e:
        logger.warning(f" ⚠️ Cassandra (RF) discovery failed: {e}. Returning empty set.")
        return set()

def _calculate_bps_rf(X_corrected: pd.DataFrame, batches: pd.Series) -> float:
    """Calculates the Batch Prediction Score (BPS-RF). Low score = Good."""
    logger = get_logger("workflow_16s")
    if batches.nunique() <= 1: 
        return 0.0
    
    X_num = X_corrected.select_dtypes(include=[np.number])
    if X_num.empty: return 0.0

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=min(3, batches.value_counts().min())) if batches.value_counts().min() >= 2 else 2
        scores = cross_val_score(rf, X_num, batches, cv=cv, scoring='accuracy')
    
    bps_score = np.mean(scores)
    
    if bps_score > 0.5:
        logger.error(f" ❌ BPS-RF FAILED: Model can still predict batch with {bps_score:.2f} accuracy after correction.")
    else:
        logger.info(f" ✅ BPS-RF PASSED: Batch prediction accuracy is low ({bps_score:.2f}).")
    return bps_score

def _audit_information_leakage(X_refined: pd.DataFrame, adata: Any, top_features: List[str]):
    """Audits final features against alpha diversity to detect CLR leakage."""
    logger = get_logger("workflow_16s")
    
    shannon_col = next((col for col in adata.obs.columns if 'shannon' in col.lower()), None)
    
    if shannon_col:
        shannon = adata.obs.loc[X_refined.index, shannon_col]
        shannon = pd.to_numeric(shannon, errors='coerce').fillna(0)
        
        for feat in top_features:
            if feat in X_refined.columns:
                feat_data = pd.to_numeric(X_refined[feat], errors='coerce').fillna(0)
                corr = np.abs(feat_data.corr(shannon))
                if corr > 0.80:
                    logger.warning(f" ⚠️ LEAKAGE DETECTED: Feature '{feat}' has r={corr:.2f} with {shannon_col}. This may be a CLR mathematical artifact!")
    else:
        logger.debug("No Shannon diversity column found. Skipping information leakage audit.")


def _generate_facility_targets(
    adata: Any, 
    priority_targets: Optional[List[str]], 
    strict_targets: bool = False
) -> List[str]:
    """Generates boolean match columns for facility-type metadata."""
    targets_to_use = list(priority_targets) if priority_targets else []
    type_col, match_col = 'facility_type', 'Env_Level_1'
    
    if type_col in adata.obs.columns and match_col in adata.obs.columns:
        if pd.api.types.is_bool_dtype(adata.obs[match_col]):
            matched_mask = adata.obs[match_col]
        else:
            matched_mask = adata.obs[match_col].astype(str).str.lower() == 'true'

        unique_types = adata.obs.loc[matched_mask, type_col].unique()
        for f_type in unique_types:
            if pd.isna(f_type) or f_type in ['None', 'Analog', 'unknown']: continue
            col_name = f"facility_match_{str(f_type).replace(' ', '_').lower()}"
            is_match = matched_mask
            is_type = adata.obs[type_col] == f_type
            adata.obs[col_name] = (is_match & is_type).astype(int).astype(str)
            if col_name not in targets_to_use: targets_to_use.append(col_name)
    
    if not targets_to_use and not strict_targets:
        defaults = ['Env_Level_1', 'Env_Level_2']
        targets_to_use = [t for t in defaults if t in adata.obs.columns]
        
    return [t for t in targets_to_use if t in adata.obs.columns]

def _setup_cv_strategy(
    strategy: str, 
    meta: pd.DataFrame, 
    batch_col: str,
    cv_strategy_threshold: int = 10
) -> tuple[Optional[pd.Series], Optional[Any], str]:
    """
    Configures CV strategy with automatic switching based on group count.
    
    Parameters:
    -----------
    strategy : str
        Requested strategy ('lopocv', 'batch_aware', etc.)
    meta : pd.DataFrame
        Metadata with grouping column
    batch_col : str
        Column name defining groups (projects/studies)
    cv_strategy_threshold : int
        Number of groups above which we switch from LeaveOneGroupOut to GroupKFold.
        Default: 10 (use LOPO with ≤10 groups, GroupKFold with >10 groups)
    
    Returns:
    --------
    tuple: (cv_groups, cv_strategy_obj, actual_cv_name)
        - cv_groups: Series of group labels
        - cv_strategy_obj: sklearn CV splitter object (LeaveOneGroupOut or GroupKFold)
        - actual_cv_name: str describing which CV was chosen
    """
    logger = get_logger("workflow_16s")
    cv_groups, cv_strategy_obj = None, None
    actual_cv_name = strategy
    
    if batch_col in meta.columns:
        groups = meta[batch_col].astype(str)
        n_groups = groups.nunique()
        
        if strategy in ['lopocv', 'batch_aware', 'conqur', 'meta_aware', 'spatial_cv']:
            if n_groups < 2:
                logger.warning(f" ⚠️ Strategy {strategy} needs >=2 groups ({n_groups} found). Reverting to baseline.")
                return None, None, strategy
            
            cv_groups = groups
            
            # DYNAMIC SWITCHING: Switch CV strategy based on group count
            if n_groups <= cv_strategy_threshold:
                # Few groups: Use strict Leave-One-Group-Out
                cv_strategy_obj = LeaveOneGroupOut()
                actual_cv_name = f"{strategy} (LOPO)"
                logger.info(f" 🗒️ CV Strategy: Leave-One-Group-Out on '{batch_col}' ({n_groups} groups ≤ {cv_strategy_threshold} threshold)")
            else:
                # Many groups: Use stable GroupKFold
                n_splits = min(5, max(3, n_groups // 2))  # Adaptive n_splits: 3-5 folds
                cv_strategy_obj = GroupKFold(n_splits=n_splits)
                actual_cv_name = f"{strategy} (GroupKFold-{n_splits})"
                logger.info(f" 🗒️ CV Strategy: GroupKFold ({n_splits} folds) on '{batch_col}' ({n_groups} groups > {cv_strategy_threshold} threshold)")
                logger.info(f"    → Switched from LOPO to GroupKFold for stability with many projects")
            
    return cv_groups, cv_strategy_obj, actual_cv_name