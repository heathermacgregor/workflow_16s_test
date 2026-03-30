"""
Integration orchestration for Modules 2-5: ASV-to-MAG, Functional Profiling, Statistics, Validation.

This file provides the wrapper functions to integrate new modules into the main analysis.py pipeline.

Author: GitHub Copilot
Date: 2026-03-20
"""

import sys
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np
import scipy.sparse as sp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from workflow_16s.downstream.asv_mag_mapping import (
    GTDBClient,
    VSEARCHWrapper,
    AssignmentEngine
)
from workflow_16s.downstream.functional_profiling import (
    DRAMParser,
    build_sample_ko_matrix,
    clr_transform
)
from workflow_16s.downstream.statistical_analysis import (
    ANCAMBCWrapper,
    ElasticNetCV,
    CandidateFeaturesSelector,
    VariancePartitioningAnalyzer
)
from workflow_16s.downstream.validation import (
    MeasuredMetalValidator,
    MetatranscriptomeValidator
)
from workflow_16s.utils.logger import get_logger


def run_asv_mag_mapping_module(workflow: Any) -> bool:
    """
    Module 2a: ASV-to-MAG Alignment & Weight Assignment
    
    Orchestrates:
    1. GTDB 16S download (cached)
    2. VSEARCH alignment
    3. Weight matrix assignment
    
    Output: Sparse ASV×MAG weight matrix stored in adata.obsm['asv_mag_weights']
    """
    logger = get_logger("workflow_16s")
    
    mag_cfg = getattr(workflow.config, 'asv_mag_mapping', None)
    if not (mag_cfg and getattr(mag_cfg, 'enabled', True)):
        logger.info("⊘ ASV-to-MAG Mapping disabled in config")
        return False
    
    logger.info("="*70)
    logger.info("MODULE 2a: ASV-to-MAG Alignment & Weight Assignment")
    logger.info("="*70)
    
    try:
        # Get config parameters
        gtdb_db_path = getattr(mag_cfg, 'gtdb_database_path', None)
        if not gtdb_db_path:
            logger.warning("⚠️ No GTDB database path configured. Skipping Module 2a.")
            return False
        
        gtdb_db_path = Path(gtdb_db_path)
        if not gtdb_db_path.exists():
            logger.warning(f"⚠️ GTDB database not found at {gtdb_db_path}. Skipping Module 2a.")
            return False
        
        # 1. Initialize GTDB client with caching
        cache_dir = Path(getattr(mag_cfg, 'gtdb_cache_dir', '/tmp/gtdb_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        client = GTDBClient(database_path=gtdb_db_path, cache_dir=cache_dir)
        logger.info(f"✓ GTDB client initialized (db: {gtdb_db_path.name})")
        
        # 2. Initialize VSEARCH wrapper
        work_dir = Path(getattr(workflow, 'output_dir', '/tmp')) / "asv_mag_alignment"
        work_dir.mkdir(parents=True, exist_ok=True)
        wrapper = VSEARCHWrapper(work_dir=work_dir)
        logger.info(f"✓ VSEARCH wrapper initialized")
        
        # 3. Extract ASV sequences from adata
        asvs_16s = {}
        if 'sequence' in workflow.adata.var.columns:
            for asv_id in workflow.adata.var_names:
                seq = workflow.adata.var.loc[asv_id, 'sequence']
                if isinstance(seq, str) and len(seq) >= 50:  # Minimum 50bp
                    asvs_16s[asv_id] = seq
        
        if not asvs_16s:
            logger.warning("⚠️ No sequences found in adata.var. Skipping Module 2a.")
            return False
        logger.info(f"✓ Extracted {len(asvs_16s)} ASV sequences")
        
        # 4. Build ASV FASTA file
        asv_fasta = work_dir / "asvs.fasta"
        with open(asv_fasta, 'w') as f:
            for asv_id, seq in asvs_16s.items():
                f.write(f">{asv_id}\n{seq}\n")
        
        # 5. Get GTDB 16S reference
        gtdb_16s = client.get_16s_sequences()
        logger.info(f"✓ Retrieved GTDB 16S (n={len(gtdb_16s)})")
        
        # 6. Run VSEARCH alignment
        align_params = {
            'method': getattr(mag_cfg, 'alignment_method', 'usearch_global'),
            'id': getattr(mag_cfg, 'identity_threshold', 0.97),
            'query_cov': getattr(mag_cfg, 'coverage_threshold', 0.50)
        }
        
        alignments = wrapper.align(asv_fasta, gtdb_16s, **align_params)
        logger.info(f"✓ VSEARCH alignment complete ({len(alignments)} hits)")
        
        # 7. Assign weights
        engine = AssignmentEngine(method=getattr(mag_cfg, 'assignment_method', 'best_hit'))
        weights = engine.assign_weights(
            alignments=alignments,
            top_matches=getattr(mag_cfg, 'top_matches', 5)
        )
        logger.info(f"✓ Assigned weights to {len(weights)} ASVs")
        
        # 8. Build sparse matrix
        mag_ids = sorted(set(hit.mag_id for hits in alignments.values() for hit in hits))
        asv_ids = sorted(asvs_16s.keys())
        
        asv_mag_matrix = engine.to_sparse_matrix(weights, asv_ids, mag_ids)
        
        # 9. Store in adata
        workflow.adata.obsm['asv_mag_weights'] = asv_mag_matrix
        workflow.adata.uns['asv_mag_mapping'] = {
            'n_asvs': asv_mag_matrix.shape[0],
            'n_mags': asv_mag_matrix.shape[1],
            'assignment_method': align_params['method'],
            'identity_threshold': align_params['id'],
            'coverage_threshold': align_params['query_cov'],
            'sparsity': 1.0 - (asv_mag_matrix.nnz / (asv_mag_matrix.shape[0] * asv_mag_matrix.shape[1]))
        }
        
        logger.info(f"✓ ASV×MAG matrix: {asv_mag_matrix.shape[0]} ASVs × {asv_mag_matrix.shape[1]} MAGs")
        logger.info(f"✓ Sparsity: {(1.0 - asv_mag_matrix.nnz / (asv_mag_matrix.shape[0] * asv_mag_matrix.shape[1])):.1%}")
        logger.info("✅ Module 2a Complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ ASV-to-MAG Mapping failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_functional_profiling_module(workflow: Any) -> bool:
    """
    Module 3: Functional KO Profile Construction
    
    Orchestrates:
    1. DRAM output parsing
    2. Matrix multiplication chain: samples×ASVs @ ASVs×MAGs @ MAGs×KOs
    3. CLR transformation for compositional data
    
    Output: Sparse samples×KO matrix with CLR transformation in adata.obsm['KO_CLR']
    """
    logger = get_logger("workflow_16s")
    
    ko_cfg = getattr(workflow.config, 'functional_profiling', None)
    if not (ko_cfg and getattr(ko_cfg, 'enabled', True)):
        logger.info("⊘ Functional Profiling disabled in config")
        return False
    
    # Require Module 2a
    if 'asv_mag_weights' not in workflow.adata.obsm:
        logger.warning("⚠️ Module 3 requires Module 2a output. Skipping.")
        return False
    
    logger.info("="*70)
    logger.info("MODULE 3: Functional KO Profile Construction")
    logger.info("="*70)
    
    try:
        # Get config parameters
        dram_dir = getattr(ko_cfg, 'dram_workspace', None)
        if not dram_dir:
            logger.warning("⚠️ DRAM workspace path not provided. Skipping functional profiling.")
            return False
        
        dram_dir = Path(dram_dir)
        if not dram_dir.exists():
            logger.warning(f"⚠️ DRAM directory not found: {dram_dir}. Skipping functional profiling.")
            return False
        
        # 1. Parse DRAM output
        logger.info(f"Loading DRAM annotations from {dram_dir}...")
        parser = DRAMParser(workspace=dram_dir)
        
        # Load MAG-to-KO relationships
        mag_ko_matrix, mag_ids, ko_ids = parser.build_mag_ko_matrix(
            annotation_sources=getattr(ko_cfg, 'annotation_sources', ['ko_id'])
        )
        logger.info(f"✓ Loaded {len(mag_ids)} MAGs with {len(ko_ids)} KEGG orthologs")
        
        # Apply quality filtering if configured
        if getattr(ko_cfg, 'normalize_by_mag_quality', False):
            min_completeness = getattr(ko_cfg, 'min_completeness', 50.0)
            quality_filter = parser.get_completeness_matrix() >= min_completeness
            mag_ko_matrix = mag_ko_matrix[quality_filter]
            logger.info(f"✓ Filtered to {quality_filter.sum()} MAGs (≥{min_completeness}% complete)")
        
        # 2. Extract matrices
        X_samples = workflow.adata.X  # samples × ASVs
        X_asv_mag = workflow.adata.obsm['asv_mag_weights']  # ASVs × MAGs
        
        # Ensure sparse matrices for efficiency
        if not sp.issparse(X_samples):
            X_samples = sp.csr_matrix(X_samples)
        if not sp.issparse(X_asv_mag):
            X_asv_mag = sp.csr_matrix(X_asv_mag)
        if not sp.issparse(mag_ko_matrix):
            mag_ko_matrix = sp.csr_matrix(mag_ko_matrix)
        
        logger.info(f"Input shapes: samples={X_samples.shape}, ASV-MAG={X_asv_mag.shape}, MAG-KO={mag_ko_matrix.shape}")
        
        # 3. Stage 1: samples × ASVs @ ASVs × MAGs = samples × MAGs
        logger.info("Computing sample-MAG profile (Stage 1)...")
        X_sample_mag = X_samples @ X_asv_mag
        logger.info(f"✓ Sample-MAG shape: {X_sample_mag.shape}")
        
        # 4. Stage 2: samples × MAGs @ MAGs × KOs = samples × KOs
        logger.info("Computing sample-KO profile (Stage 2)...")
        X_ko_counts = X_sample_mag @ mag_ko_matrix
        logger.info(f"✓ Sample-KO shape: {X_ko_counts.shape}")
        
        # 5. CLR transformation (for compositional data)
        logger.info("Applying Centered Log-Ratio transformation...")
        pseudocount = getattr(ko_cfg, 'pseudocount', 0.5)
        X_ko_clr = clr_transform(
            X_ko_counts,
            pseudocount=pseudocount,
            sparse_output=True
        )
        logger.info(f"✓ CLR transformation complete (pseudocount={pseudocount})")
        
        # 6. Store results in adata
        workflow.adata.obsm['KO_counts'] = X_ko_counts
        workflow.adata.obsm['KO_CLR'] = X_ko_clr
        
        # Store metadata
        workflow.adata.uns['functional_profiling'] = {
            'n_samples': X_ko_counts.shape[0],
            'n_kos': X_ko_counts.shape[1],
            'n_mags': mag_ko_matrix.shape[0],
            'ko_ids': ko_ids,
            'dram_dir': str(dram_dir),
            'pseudocount': pseudocount,
            'normalization': 'CLR',
            'sparsity_ko_clr': 1.0 - (X_ko_clr.nnz / (X_ko_clr.shape[0] * X_ko_clr.shape[1]))
        }
        
        # Export matrices if configured
        if getattr(ko_cfg, 'export_ko_matrix', False):
            output_dir = Path(getattr(ko_cfg, 'export_ko_matrix_path', '/tmp/ko_matrix.npz')).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            export_path = output_dir / 'ko_matrix.npz'
            sp.save_npz(export_path, X_ko_counts)
            logger.info(f"✓ Exported KO counts to {export_path}")
            
            export_clr_path = output_dir / 'ko_clr_matrix.npz'
            sp.save_npz(export_clr_path, X_ko_clr)
            logger.info(f"✓ Exported KO CLR to {export_clr_path}")
        
        logger.info(f"✓ KO CLR matrix: {X_ko_clr.shape[0]} samples × {X_ko_clr.shape[1]} KOs")
        logger.info("✅ Module 3 Complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Functional Profiling failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_statistical_analysis_module(workflow: Any) -> bool:
    """
    Module 4: Statistical Analysis (Differential Abundance, Variance Partitioning)
    
    Orchestrates:
    1. ANCOM-BC differential abundance
    2. ElasticNet feature selection (LOOCV)
    3. Consensus feature selection
    4. RDA variance partitioning
    
    Output: Results stored in adata.uns['statistical_analysis']
    """
    logger = get_logger("workflow_16s")
    
    stat_cfg = getattr(workflow.config, 'statistical_analysis', None)
    if not (stat_cfg and getattr(stat_cfg, 'enabled', True)):
        logger.info("⊘ Statistical Analysis disabled in config")
        return False
    
    # Require Module 3
    if 'KO_CLR' not in workflow.adata.obsm:
        logger.warning("⚠️ Module 4 requires Module 3 output. Skipping.")
        return False
    
    logger.info("="*70)
    logger.info("MODULE 4: Statistical Analysis")
    logger.info("="*70)
    
    try:
        output_dir = Path(getattr(workflow, 'output_dir', '/tmp')) / "statistical_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract KO profiles and metadata
        X_ko_clr = workflow.adata.obsm['KO_CLR']
        y_outcome = workflow.adata.obs.get(
            getattr(stat_cfg, 'outcome_variable', None),
            None
        )
        batch = workflow.adata.obs.get(
            getattr(stat_cfg, 'batch_column', None),
            None
        )
        
        if y_outcome is None or batch is None:
            logger.warning("⚠️ Missing outcome_variable or batch_column in obs. Skipping Module 4.")
            return False
        
        results = {}
        selected_features = {}
        
        # 1. ANCOM-BC (if configured)
        if getattr(stat_cfg, 'run_ancombc', True):
            logger.info("Running ANCOM-BC differential abundance...")
            try:
                ancombc = ANCAMBCWrapper(
                    formula=getattr(stat_cfg, 'ancombc_formula', 'outcome_variable ~ batch_column'),
                    lib_size_correction=getattr(stat_cfg, 'ancombc_lib_size_correction', 'CSS')
                )
                
                ancombc_results = ancombc.run(
                    X=X_ko_clr,
                    metadata=workflow.adata.obs[[y_outcome.name, batch.name]],
                    effect_size_threshold=getattr(stat_cfg, 'ancombc_effect_size_threshold', 0.5),
                    pvalue_threshold=getattr(stat_cfg, 'ancombc_pvalue_threshold', 0.05)
                )
                
                features_ancombc = ancombc_results.get('significant_features', [])
                results['ancombc'] = ancombc_results
                selected_features['ancombc'] = features_ancombc
                logger.info(f"✓ ANCOM-BC identified {len(features_ancombc)} significant features")
                
            except Exception as e:
                logger.warning(f"⚠️ ANCOM-BC failed: {e}. Continuing...")
                results['ancombc_error'] = str(e)
        
        # 2. ElasticNet with LOOCV (if configured)
        if getattr(stat_cfg, 'run_elasticnet', True):
            logger.info("Running ElasticNet with Leave-One-Study-Out CV...")
            try:
                elnet = ElasticNetCV(
                    l1_ratio=getattr(stat_cfg, 'elasticnet_l1_ratio', 0.5),
                    alpha=getattr(stat_cfg, 'elasticnet_alpha', 0.1),
                    cv_folds=getattr(stat_cfg, 'elasticnet_cv_folds', 5)
                )
                
                loocv_group = None
                if getattr(stat_cfg, 'use_loocv', False):
                    loocv_group = workflow.adata.obs.get(
                        getattr(stat_cfg, 'loocv_group_column', None),
                        None
                    )
                
                elnet_results = elnet.run(
                    X=X_ko_clr,
                    y=y_outcome,
                    cv_groups=loocv_group,
                    max_features=getattr(stat_cfg, 'elasticnet_max_features', 100)
                )
                
                features_elnet = elnet_results.get('selected_features', [])
                results['elasticnet'] = elnet_results
                selected_features['elasticnet'] = features_elnet
                logger.info(f"✓ ElasticNet selected {len(features_elnet)} features")
                
            except Exception as e:
                logger.warning(f"⚠️ ElasticNet failed: {e}. Continuing...")
                results['elasticnet_error'] = str(e)
        
        # 3. Consensus selection (if multiple methods ran)
        if len(selected_features) > 1 and getattr(stat_cfg, 'run_consensus', True):
            logger.info("Finding consensus features across methods...")
            
            # Find features selected by at least N methods
            threshold = getattr(stat_cfg, 'consensus_threshold', 0.5)
            min_methods = max(1, int(len(selected_features) * threshold))
            
            feature_counts = {}
            for method, features in selected_features.items():
                for feat in features:
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
            
            consensus_features = [
                f for f, count in feature_counts.items()
                if count >= min_methods
            ]
            
            results['consensus'] = {
                'features': consensus_features,
                'n_methods_required': min_methods,
                'total_methods': len(selected_features)
            }
            selected_features['consensus'] = consensus_features
            logger.info(f"✓ Consensus found {len(consensus_features)} features")
        
        # 4. RDA Variance Partitioning (if configured)
        if getattr(stat_cfg, 'run_variance_partitioning', True):
            logger.info("Running variance partitioning (RDA)...")
            try:
                rda_factors = getattr(stat_cfg, 'variance_factors', [])
                
                # Check that all factors exist
                available_factors = [f for f in rda_factors if f in workflow.adata.obs.columns]
                
                if available_factors:
                    analyzer = VariancePartitioningAnalyzer()
                    rda_results = analyzer.run_rda(
                        X=X_ko_clr,
                        metadata=workflow.adata.obs[available_factors],
                        formula=getattr(stat_cfg, 'rda_formula', None),
                        significance_threshold=getattr(stat_cfg, 'rda_significance_threshold', 0.05)
                    )
                    
                    results['rda'] = rda_results
                    logger.info(f"✓ RDA variance partitioning complete")
                    
                    # Log variance explained
                    if 'variance_explained' in rda_results:
                        for factor, var_exp in rda_results['variance_explained'].items():
                            logger.info(f"  {factor}: {var_exp:.1%}")
                else:
                    logger.warning(f"⚠️ None of {rda_factors} found in obs. Skipping RDA.")
                    
            except Exception as e:
                logger.warning(f"⚠️ RDA failed: {e}. Continuing...")
                results['rda_error'] = str(e)
        
        # 5. Store results
        all_selected = {}
        for method, features in selected_features.items():
            all_selected[f'{method}_features'] = features
        
        workflow.adata.uns['statistical_analysis'] = {
            **results,
            **all_selected,
            'n_kos_tested': X_ko_clr.shape[1],
            'n_samples': X_ko_clr.shape[0],
            'outcome_variable': getattr(stat_cfg, 'outcome_variable', None),
            'batch_variable': getattr(stat_cfg, 'batch_column', None)
        }
        
        logger.info(f"✓ Results stored ({len(results)} analyses, {len(selected_features)} feature sets)")
        logger.info("✅ Module 4 Complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistical Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_validation_module(workflow: Any) -> bool:
    """
    Module 5: Validation (Cross-validation with Measured Data)
    
    Orchestrates:
    1. Metal measurement validation (if available)
    2. Metatranscriptome expression validation (if available)
    3. Result aggregation and confidence scoring
    
    Output: Results stored in adata.uns['validation']
    """
    logger = get_logger("workflow_16s")
    
    val_cfg = getattr(workflow.config, 'validation', None)
    if not (val_cfg and getattr(val_cfg, 'enabled', True)):
        logger.info("⊘ Validation disabled in config")
        return False
    
    logger.info("="*70)
    logger.info("MODULE 5: Validation (Cross-validation with Measured Data)")
    logger.info("="*70)
    
    try:
        # Require Module 3 (KO profiles)
        if 'KO_CLR' not in workflow.adata.obsm:
            logger.warning("⚠️ Module 5 requires Module 3 output. Skipping.")
            return False
        
        validation_results = {}
        all_confidences = {}
        
        # 1. Metal measurement validation
        metals_file = getattr(val_cfg, 'measured_metals_file', None)
        if metals_file and Path(metals_file).exists():
            logger.info(f"Loading measured metals from {metals_file}...")
            try:
                metals_df = pd.read_csv(metals_file, index_col=0)
                
                # Find sample overlap
                sample_overlap = metals_df.index.intersection(workflow.adata.obs_names)
                if len(sample_overlap) < 5:
                    logger.warning(f"⚠️ Only {len(sample_overlap)} samples overlap with metals. Skipping.")
                else:
                    # Subset data to common samples
                    X_ko_subset = workflow.adata.obsm['KO_CLR'][
                        [i for i, s in enumerate(workflow.adata.obs_names) if s in sample_overlap]
                    ]
                    metals_subset = metals_df.loc[sample_overlap]
                    
                    # Run validation
                    metal_validator = MeasuredMetalValidator(
                        method=getattr(val_cfg, 'metal_correlation_method', 'spearman')
                    )
                    metal_results = metal_validator.validate(
                        X_ko=X_ko_subset,
                        metals_measured=metals_subset,
                        pvalue_threshold=getattr(val_cfg, 'metal_pvalue_threshold', 0.05),
                        min_r_squared=getattr(val_cfg, 'confidence_thresholds', {}).get('low', 0.10)
                    )
                    
                    validation_results['metal_validation'] = metal_results
                    all_confidences.update(metal_results.get('confidence_scores', {}))
                    
                    logger.info(f"✓ Metal validation complete ({len(metal_results.get('correlations', {}))} KO-metal pairs)")
                    
            except Exception as e:
                logger.warning(f"⚠️ Metal validation failed: {e}")
                validation_results['metal_validation_error'] = str(e)
        
        # 2. Metatranscriptome validation
        expr_file = getattr(val_cfg, 'metatranscriptome_file', None)
        if expr_file and Path(expr_file).exists():
            logger.info(f"Loading metatranscriptome from {expr_file}...")
            try:
                expr_df = pd.read_csv(expr_file, index_col=0)
                
                # Find sample overlap
                sample_overlap = expr_df.index.intersection(workflow.adata.obs_names)
                if len(sample_overlap) < 5:
                    logger.warning(f"⚠️ Only {len(sample_overlap)} samples overlap with metatranscriptome. Skipping.")
                else:
                    # Subset data
                    X_ko_subset = workflow.adata.obsm['KO_CLR'][
                        [i for i, s in enumerate(workflow.adata.obs_names) if s in sample_overlap]
                    ]
                    expr_subset = expr_df.loc[sample_overlap]
                    
                    # Normalize if configured
                    norm_method = getattr(val_cfg, 'expression_normalization', 'log2')
                    if norm_method == 'log2' and (expr_subset < 0).any().any():
                        # Already log-transformed
                        pass
                    
                    # Run validation
                    expr_validator = MetatranscriptomeValidator(
                        method=getattr(val_cfg, 'expression_correlation_method', 'spearman')
                    )
                    expr_results = expr_validator.compare_abundance_expression(
                        X_ko=X_ko_subset,
                        expression=expr_subset,
                        pvalue_threshold=getattr(val_cfg, 'metal_pvalue_threshold', 0.05),
                        min_r_squared=getattr(val_cfg, 'confidence_thresholds', {}).get('low', 0.10)
                    )
                    
                    validation_results['metatranscriptome_validation'] = expr_results
                    all_confidences.update(expr_results.get('confidence_scores', {}))
                    
                    logger.info(f"✓ Metatranscriptome validation complete ({len(expr_results.get('correlations', {}))} KO-gene pairs)")
                    
            except Exception as e:
                logger.warning(f"⚠️ Metatranscriptome validation failed: {e}")
                validation_results['metatranscriptome_validation_error'] = str(e)
        
        if not validation_results:
            logger.warning("⚠️ No validation data files provided or accessible. Skipping Module 5.")
            return False
        
        # 3. Categorize confidence
        confidence_thresholds = getattr(val_cfg, 'confidence_thresholds', {
            'high': 0.50,
            'moderate': 0.30,
            'low': 0.10
        })
        
        confidence_categories = {'high': [], 'moderate': [], 'low': []}
        for ko_id, conf_score in all_confidences.items():
            if conf_score >= confidence_thresholds['high']:
                confidence_categories['high'].append(ko_id)
            elif conf_score >= confidence_thresholds['moderate']:
                confidence_categories['moderate'].append(ko_id)
            elif conf_score >= confidence_thresholds['low']:
                confidence_categories['low'].append(ko_id)
        
        # 4. Store results
        validation_results['confidence_categories'] = confidence_categories
        validation_results['confidence_scores'] = all_confidences
        validation_results['n_samples_validated'] = len(sample_overlap) if 'sample_overlap' in locals() else 0
        validation_results['n_ko_validated'] = len(all_confidences)
        
        workflow.adata.uns['validation'] = validation_results
        
        logger.info(f"✓ Validated {len(all_confidences)} KOs")
        logger.info(f"  High confidence: {len(confidence_categories['high'])}")
        logger.info(f"  Moderate confidence: {len(confidence_categories['moderate'])}")
        logger.info(f"  Low confidence: {len(confidence_categories['low'])}")
        logger.info("✅ Module 5 Complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    print("Integration orchestration module for Modules 2-5")
    print("Import functions: run_asv_mag_mapping_module, run_functional_profiling_module, ...")
