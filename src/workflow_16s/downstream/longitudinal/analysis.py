import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import anndata as ad
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger('workflow_16s')

# Check for R and rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    logger.warning("rpy2 not available. R-based longitudinal methods (ZIBR, MaAsLin2) will not work.")

def _check_r_package(package_name: str) -> bool:
    if not R_AVAILABLE: return False
    try:
        importr(package_name)
        return True
    except Exception:
        return False

def run_zibr(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    group_col: Optional[str] = None,
    feature: Optional[str] = None,
    alpha: float = 0.05,
    min_prevalence: float = 0.1
) -> pd.DataFrame:
    """Run ZIBR (Zero-Inflated Beta Regression) via R."""
    if not _check_r_package('ZIBR'):
        raise RuntimeError("ZIBR R package not available.")
    
    logger.info("Running ZIBR longitudinal analysis...")
    
    # Filter features
    prevalence = (adata.X > 0).mean(axis=0)
    if hasattr(prevalence, 'A1'): prevalence = prevalence.A1
    keep_features = prevalence >= min_prevalence
    
    if feature:
        if feature not in adata.var_names: raise ValueError(f"Feature '{feature}' not found")
        features_to_test = [feature]
    else:
        features_to_test = adata.var_names[keep_features]

    # Prepare data
    abundance_data = adata.to_df()
    rel_abund = abundance_data.div(abundance_data.sum(axis=1), axis=0)
    metadata = adata.obs[[time_col, subject_col]].copy()
    if group_col: metadata[group_col] = adata.obs[group_col]

    results = []
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.r('library(ZIBR)')
        
        for feat in features_to_test:
            feat_data = pd.DataFrame({
                'abundance': rel_abund[feat].values,
                'time': metadata[time_col].values,
                'subject': metadata[subject_col].values
            })
            if group_col: feat_data['group'] = metadata[group_col].values
            
            try:
                r_data = pandas2ri.py2rpy(feat_data)
                ro.r.assign('feat_data', r_data)
                cov_str = "c('time', 'subject', 'group')" if group_col else "c('time', 'subject')"
                
                ro.r(f'''
                zibr_fit <- zibr(
                    logistic.cov = feat_data[, {cov_str}],
                    beta.cov = feat_data[, {cov_str}],
                    Y = feat_data$abundance,
                    subject.ind = feat_data$subject,
                    time.ind = feat_data$time
                )
                logistic_coef <- zibr_fit$logistic.est.table
                beta_coef <- zibr_fit$beta.est.table
                ''')
                
                l_coef = pandas2ri.rpy2py(ro.r('logistic_coef'))
                b_coef = pandas2ri.rpy2py(ro.r('beta_coef'))
                
                l_p = l_coef.loc[l_coef.index.str.contains('time'), 'Pr(>|z|)'].iloc[0]
                b_p = b_coef.loc[b_coef.index.str.contains('time'), 'Pr(>|t|)'].iloc[0]
                
                results.append({
                    'feature': feat,
                    'logistic_time_p': l_p,
                    'beta_time_p': b_p,
                    'min_p': min(l_p, b_p)
                })
            except Exception as e:
                continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        _, p_adj, _, _ = multipletests(results_df['min_p'], method='fdr_bh')
        results_df['p_adj'] = p_adj
        results_df = results_df.sort_values('p_adj')
        
    return results_df

def run_maaslin2_longitudinal(
    adata: ad.AnnData,
    time_col: str,
    subject_col: str,
    fixed_effects: List[str],
    random_effects: List[str],
    output_dir: Path,
    alpha: float = 0.05,
    normalization: str = 'TSS',
    transform: str = 'LOG'
) -> pd.DataFrame:
    """Run MaAsLin 2 mixed-effects models via R."""
    if not _check_r_package('Maaslin2'):
        raise RuntimeError("Maaslin2 R package not available.")
    
    logger.info("Running MaAsLin 2 (Longitudinal)...")
    
    if subject_col not in random_effects:
        random_effects.append(subject_col)
    
    abundance = adata.to_df()
    meta_cols = list(set([subject_col, time_col] + fixed_effects + random_effects))
    metadata = adata.obs[meta_cols].copy()
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.r.assign('abundance', pandas2ri.py2rpy(abundance))
        ro.r.assign('metadata', pandas2ri.py2rpy(metadata))
        ro.r.assign('output_dir', str(output_dir))
        
        fixed_str = ", ".join([f'"{x}"' for x in fixed_effects])
        random_str = ", ".join([f'"{x}"' for x in random_effects])
        
        ro.r(f'''
        library(Maaslin2)
        Maaslin2(
            input_data = abundance, input_metadata = metadata, output = output_dir,
            fixed_effects = c({fixed_str}), random_effects = c({random_str}),
            normalization = "{normalization}", transform = "{transform}",
            analysis_method = "LM", max_significance = {alpha},
            plot_heatmap = FALSE, plot_scatter = FALSE
        )
        ''')

    res_file = output_dir / "significant_results.tsv"
    return pd.read_csv(res_file, sep='\t') if res_file.exists() else pd.DataFrame()