# ==================================================================================== #
#                       downstream/steps/metadata_profiler.py
# ==================================================================================== #

"""
Metadata Profiler: Comprehensive metadata quality assessment and ML readiness checks.

Generates:
- Dataset-level statistics (study counts, sample sizes)
- Column completeness profiles
- Categorical distribution summaries with hierarchical breakdowns
- ML warnings (class imbalance, confounding, missing data issues)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from anndata import AnnData
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import re
from urllib.parse import quote

logger = logging.getLogger('workflow_16s')


def calculate_entropy(series: pd.Series) -> float:
    """Calculate Shannon entropy for categorical distribution."""
    counts = series.value_counts()
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def detect_confounding(adata: AnnData, col1: str, col2: str, threshold: float = 0.95) -> float:
    """
    Detect potential confounding between two categorical variables.
    Returns: Cramér's V statistic (0-1, higher = stronger association)
    """
    try:
        obs = adata.obs[[col1, col2]].dropna()
        if len(obs) < 10:
            return 0.0
        
        contingency = pd.crosstab(obs[col1], obs[col2])
        chi2 = 0
        row_totals = contingency.sum(axis=1)
        col_totals = contingency.sum(axis=0)
        n = contingency.sum().sum()
        
        for i in range(len(contingency)):
            for j in range(len(contingency.columns)):
                expected = row_totals.iloc[i] * col_totals.iloc[j] / n
                if expected > 0:
                    chi2 += (contingency.iloc[i, j] - expected) ** 2 / expected
        
        # Cramér's V
        min_dim = min(len(contingency), len(contingency.columns)) - 1
        if min_dim == 0:
            return 0.0
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return cramers_v
    except Exception as e:
        logger.debug(f"Could not calculate confounding for {col1} vs {col2}: {e}")
        return 0.0


def profile_metadata(
    adata: AnnData,
    output_dir: Path,
    ml_targets: Optional[List[str]] = None,
    priority_columns: Optional[List[str]] = None
) -> Dict:
    """
    Generate comprehensive metadata profiling report.
    
    Args:
        adata: AnnData object with metadata in .obs
        output_dir: Directory to save reports
        ml_targets: Target columns for ML analysis (to check for issues)
        priority_columns: Important columns to highlight
        
    Returns:
        Dictionary with profiling results and warnings
    """
    logger.info("=" * 80)
    logger.info("METADATA PROFILING REPORT")
    logger.info("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    obs = adata.obs
    n_samples = len(obs)
    
    # 1. Dataset-level statistics
    logger.info("\n1. DATASET OVERVIEW")
    logger.info("-" * 80)
    
    dataset_stats = {}
    
    # Count unique values in common ID columns
    id_columns = [
        'accession', 'project_name', 'study_accession', 
        'experiment_accession', 'sample_accession'
    ]
    for col in id_columns:
        if col in obs.columns:
            n_unique = obs[col].nunique()
            dataset_stats[col] = n_unique
            logger.info(f"  Unique {col:25s}: {n_unique:>8,}")
    
    logger.info(f"  {'Total samples':25s}: {n_samples:>8,}")
    
    # 2. Column completeness
    logger.info("\n2. METADATA COMPLETENESS")
    logger.info("-" * 80)
    logger.info(f"  {'Column':<40s} {'Complete':<12s} {'Missing':<12s} {'% Full':<10s}")
    logger.info("  " + "-" * 74)
    
    completeness = []
    for col in sorted(obs.columns):
        n_missing = obs[col].isna().sum()
        n_complete = n_samples - n_missing
        pct_full = 100 * n_complete / n_samples
        
        completeness.append({
            'column': col,
            'n_complete': n_complete,
            'n_missing': n_missing,
            'pct_full': pct_full,
            'dtype': str(obs[col].dtype)
        })
        
        # Highlight very sparse or very complete columns
        marker = ""
        if pct_full < 10:
            marker = " ⚠️  VERY SPARSE"
        elif pct_full == 100:
            marker = " ✓"
        
        logger.info(f"  {col:<40s} {n_complete:>10,}  {n_missing:>10,}  {pct_full:>8.1f}%{marker}")
    
    completeness_df = pd.DataFrame(completeness).sort_values('pct_full', ascending=False)
    completeness_df.to_csv(output_dir / 'metadata_completeness.csv', index=False)
    
    # 3. Categorical distributions with hierarchical breakdowns
    logger.info("\n3. CATEGORICAL DISTRIBUTIONS")
    logger.info("-" * 80)
    
    categorical_cols = obs.select_dtypes(include=['object', 'category']).columns
    distributions = []
    
    for col in categorical_cols:
        value_counts = obs[col].value_counts(dropna=False)
        n_categories = len(value_counts)
        entropy = calculate_entropy(obs[col].dropna())
        
        # Check for severe imbalance
        if len(value_counts) > 1:
            majority_pct = 100 * value_counts.iloc[0] / value_counts.sum()
            minority_pct = 100 * value_counts.iloc[-1] / value_counts.sum()
            imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
        else:
            majority_pct = 100.0
            minority_pct = 100.0
            imbalance_ratio = 1.0
        
        distributions.append({
            'column': col,
            'n_categories': n_categories,
            'entropy': entropy,
            'majority_pct': majority_pct,
            'minority_pct': minority_pct,
            'imbalance_ratio': imbalance_ratio
        })
        
        # Log top categories for important columns
        if col in (priority_columns or []) or col in ['environment_biome', 'country', 'location']:
            logger.info(f"\n  {col}:")
            logger.info(f"    Categories: {n_categories}, Entropy: {entropy:.2f}")
            logger.info(f"    Top 10 values:")
            for val, count in value_counts.head(10).items():
                pct = 100 * count / n_samples
                logger.info(f"      {str(val)[:50]:<50s}: {count:>8,} ({pct:>5.1f}%)")
            
            if n_categories > 10:
                logger.info(f"      ... and {n_categories - 10} more categories")
    
    distributions_df = pd.DataFrame(distributions).sort_values('n_categories', ascending=False)
    distributions_df.to_csv(output_dir / 'categorical_distributions.csv', index=False)
    
    # 4. Numeric distributions
    logger.info("\n4. NUMERIC DISTRIBUTIONS")
    logger.info("-" * 80)
    logger.info(f"  {'Column':<40s} {'Mean':<12s} {'Std':<12s} {'Min':<12s} {'Max':<12s}")
    logger.info("  " + "-" * 88)
    
    numeric_cols = obs.select_dtypes(include=[np.number]).columns
    numeric_stats = []
    
    for col in sorted(numeric_cols):
        values = obs[col].dropna()
        if len(values) > 0:
            stats = {
                'column': col,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            }
            numeric_stats.append(stats)
            logger.info(
                f"  {col:<40s} {stats['mean']:>10.2f}  {stats['std']:>10.2f}  "
                f"{stats['min']:>10.2f}  {stats['max']:>10.2f}"
            )
    
    if numeric_stats:
        numeric_df = pd.DataFrame(numeric_stats)
        numeric_df.to_csv(output_dir / 'numeric_statistics.csv', index=False)
    
    # 5. ML Readiness Assessment
    logger.info("\n5. MACHINE LEARNING READINESS")
    logger.info("-" * 80)
    
    warnings = []
    ml_targets = ml_targets or []
    
    for target in ml_targets:
        if target not in obs.columns:
            warnings.append({
                'severity': 'ERROR',
                'category': 'Missing Target',
                'target': target,
                'message': f"ML target '{target}' not found in metadata"
            })
            logger.error(f"  ❌ Target '{target}' not found in metadata")
            continue
        
        # Check target completeness
        n_missing = obs[target].isna().sum()
        if n_missing > 0:
            warnings.append({
                'severity': 'WARNING',
                'category': 'Missing Target Data',
                'target': target,
                'message': f"Target '{target}' has {n_missing:,} ({100*n_missing/n_samples:.1f}%) missing values"
            })
            logger.warning(f"  ⚠️  Target '{target}' missing {n_missing:,} values ({100*n_missing/n_samples:.1f}%)")
        
        # Check class imbalance for categorical targets
        if obs[target].dtype in ['object', 'category'] or obs[target].nunique() < 50:
            value_counts = obs[target].value_counts()
            n_classes = len(value_counts)
            
            if n_classes < 2:
                warnings.append({
                    'severity': 'ERROR',
                    'category': 'Insufficient Classes',
                    'target': target,
                    'message': f"Target '{target}' has only {n_classes} class - need ≥2 for classification"
                })
                logger.error(f"  ❌ Target '{target}' has only {n_classes} class")
            elif n_classes >= 2:
                # Get majority and minority class info
                majority_class = value_counts.index[0]
                majority_count = value_counts.iloc[0]
                minority_class = value_counts.index[-1]
                minority_count = value_counts.iloc[-1]
                imbalance = majority_count / minority_count
                
                # Format class names (truncate if too long)
                maj_label = str(majority_class)[:30] + ('...' if len(str(majority_class)) > 30 else '')
                min_label = str(minority_class)[:30] + ('...' if len(str(minority_class)) > 30 else '')
                
                if imbalance > 100:
                    warnings.append({
                        'severity': 'ERROR',
                        'category': 'Severe Imbalance',
                        'target': target,
                        'message': f"Target '{target}' severely imbalanced: {imbalance:.0f}:1 ratio ('{maj_label}': {majority_count} vs '{min_label}': {minority_count})"
                    })
                    logger.error(f"  ❌ Target '{target}' severely imbalanced ({imbalance:.0f}:1): '{maj_label}' ({majority_count}) vs '{min_label}' ({minority_count})")
                elif imbalance > 10:
                    warnings.append({
                        'severity': 'WARNING',
                        'category': 'Class Imbalance',
                        'target': target,
                        'message': f"Target '{target}' imbalanced: {imbalance:.0f}:1 ratio ('{maj_label}': {majority_count} vs '{min_label}': {minority_count}) - consider SMOTE/class_weight"
                    })
                    logger.warning(f"  ⚠️  Target '{target}' imbalanced ({imbalance:.0f}:1): '{maj_label}' ({majority_count}) vs '{min_label}' ({minority_count})")
                else:
                    logger.info(f"  ✓ Target '{target}' balanced ({imbalance:.1f}:1): '{maj_label}' ({majority_count}) vs '{min_label}' ({minority_count})")
                
                # Check minimum class size
                min_class_size = value_counts.iloc[-1]
                if min_class_size < 30:
                    warnings.append({
                        'severity': 'WARNING',
                        'category': 'Small Class',
                        'target': target,
                        'message': f"Target '{target}' minority class '{min_label}' has only {min_class_size} samples"
                    })
                    logger.warning(f"  ⚠️  Target '{target}' minority class '{min_label}' has only {min_class_size} samples")
    
    # 6. Confounding detection
    logger.info("\n6. POTENTIAL CONFOUNDING FACTORS")
    logger.info("-" * 80)
    
    if ml_targets:
        confounding_pairs = []
        # Exclude sample IDs and accessions from confounding analysis
        exclude_patterns = ['sample', 'accession', 'run', 'experiment', 'biosample', 'id', 'sra']
        batch_columns = ['project_name', 'study_accession', 'sequencing_method', 'instrument_model', 
                         'library_selection', 'library_source', 'library_strategy']
        # Filter out ID-like columns
        batch_columns = [col for col in batch_columns if col in obs.columns]
        # Also exclude any column that looks like an ID
        batch_columns = [col for col in batch_columns 
                         if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        for target in ml_targets:
            if target not in obs.columns:
                continue
            # Skip if target is an ID-like column
            if any(pattern in target.lower() for pattern in exclude_patterns):
                continue
            
            for batch_col in batch_columns:
                if batch_col == target:
                    continue
                
                cramers_v = detect_confounding(adata, target, batch_col)
                
                confounding_pairs.append({
                    'target': target,
                    'confound': batch_col,
                    'cramers_v': cramers_v
                })
                
                if cramers_v > 0.7:
                    warnings.append({
                        'severity': 'ERROR',
                        'category': 'Severe Confounding',
                        'target': target,
                        'message': f"'{target}' severely confounded with '{batch_col}' (Cramér's V={cramers_v:.2f})"
                    })
                    logger.error(f"  ❌ '{target}' ↔ '{batch_col}': Cramér's V = {cramers_v:.2f} (SEVERE)")
                elif cramers_v > 0.4:
                    warnings.append({
                        'severity': 'WARNING',
                        'category': 'Moderate Confounding',
                        'target': target,
                        'message': f"'{target}' moderately confounded with '{batch_col}' (Cramér's V={cramers_v:.2f})"
                    })
                    logger.warning(f"  ⚠️  '{target}' ↔ '{batch_col}': Cramér's V = {cramers_v:.2f} (MODERATE)")
                elif cramers_v > 0.2:
                    logger.info(f"  ℹ️  '{target}' ↔ '{batch_col}': Cramér's V = {cramers_v:.2f} (weak)")
        
        if confounding_pairs:
            confounding_df = pd.DataFrame(confounding_pairs).sort_values('cramers_v', ascending=False)
            confounding_df.to_csv(output_dir / 'confounding_analysis.csv', index=False)
    
    # 7. Summary
    logger.info("\n7. SUMMARY")
    logger.info("-" * 80)
    logger.info(f"  Total samples: {n_samples:,}")
    logger.info(f"  Total metadata columns: {len(obs.columns)}")
    logger.info(f"  Categorical columns: {len(categorical_cols)}")
    logger.info(f"  Numeric columns: {len(numeric_cols)}")
    logger.info(f"  Warnings generated: {len(warnings)}")
    
    error_count = sum(1 for w in warnings if w['severity'] == 'ERROR')
    warning_count = sum(1 for w in warnings if w['severity'] == 'WARNING')
    
    if error_count > 0:
        logger.error(f"  ❌ {error_count} ERRORS found - ML analysis may fail")
    if warning_count > 0:
        logger.warning(f"  ⚠️  {warning_count} WARNINGS found - review before ML analysis")
    if error_count == 0 and warning_count == 0:
        logger.info(f"  ✓ No critical issues detected")
    
    # Save warnings
    if warnings:
        warnings_df = pd.DataFrame(warnings)
        warnings_df.to_csv(output_dir / 'ml_warnings.csv', index=False)
        
        # Group by severity
        logger.info("\n  Warnings by category:")
        for category in warnings_df['category'].unique():
            count = len(warnings_df[warnings_df['category'] == category])
            logger.info(f"    - {category}: {count}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Metadata profiling complete. Reports saved to: {output_dir}")
    logger.info("=" * 80 + "\n")
    
    # Generate visualizations
    logger.info("Generating metadata visualizations...")
    try:
        generate_metadata_visualizations(adata, output_dir)
        logger.info(f"✓ Visualizations saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
    
    return {
        'dataset_stats': dataset_stats,
        'completeness': completeness_df,
        'distributions': distributions_df,
        'numeric_stats': numeric_df if numeric_stats else None,
        'warnings': warnings_df if warnings else None,
        'n_errors': error_count,
        'n_warnings': warning_count
    }


def generate_metadata_visualizations(adata: AnnData, output_dir: Path):
    """Generate comprehensive metadata visualizations."""
    
    obs = adata.obs
    
    # 0. PRIORITY: Dataset breakdown visualizations
    generate_dataset_breakdown(obs, output_dir)
    
    # 0.1 PRIORITY: Dataset table with ENA links and citations
    generate_dataset_table(obs, output_dir)
    
    # 1. Geographic sample map with nuclear facilities
    if 'latitude' in obs.columns and 'longitude' in obs.columns:
        generate_sample_map(adata, output_dir)
    
    # 2. Numeric metadata heatmap
    numeric_cols = obs.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        generate_numeric_heatmap(obs, numeric_cols, output_dir)
    
    # 3. Sample distribution plots
    generate_sample_distribution_plots(obs, output_dir)
    
    # 4. Category breakdown plots
    generate_category_plots(obs, output_dir)


def fetch_publication_info(study_accession: str) -> Optional[Dict[str, str]]:
    """Fetch publication information from ENA API."""
    try:
        # Query ENA for study information
        url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{study_accession}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        # Extract publication information from XML
        text = response.text
        
        # Look for PubMed ID
        pubmed_match = re.search(r'<PUBMED_ID>(\d+)</PUBMED_ID>', text)
        doi_match = re.search(r'<DOI>([^<]+)</DOI>', text)
        
        if not pubmed_match and not doi_match:
            return None
        
        result = {}
        
        if pubmed_match:
            pubmed_id = pubmed_match.group(1)
            result['pubmed_id'] = pubmed_id
            result['pubmed_url'] = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
            
            # Try to fetch citation from PubMed
            try:
                pubmed_api = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pubmed_id}&retmode=json"
                pm_response = requests.get(pubmed_api, timeout=10)
                if pm_response.status_code == 200:
                    pm_data = pm_response.json()
                    if 'result' in pm_data and pubmed_id in pm_data['result']:
                        article = pm_data['result'][pubmed_id]
                        authors = article.get('authors', [])
                        author_str = authors[0]['name'] if authors else 'Unknown'
                        if len(authors) > 1:
                            author_str += ' et al.'
                        
                        title = article.get('title', 'Unknown title')
                        year = article.get('pubdate', '').split()[0] if article.get('pubdate') else ''
                        journal = article.get('source', '')
                        
                        result['citation'] = f"{author_str} ({year}). {title}. {journal}."
            except Exception as e:
                logger.debug(f"Could not fetch citation for PubMed {pubmed_id}: {e}")
        
        if doi_match:
            doi = doi_match.group(1)
            result['doi'] = doi
            result['doi_url'] = f"https://doi.org/{doi}"
        
        return result if result else None
        
    except Exception as e:
        logger.debug(f"Could not fetch publication info for {study_accession}: {e}")
        return None


def generate_dataset_breakdown(obs: pd.DataFrame, output_dir: Path):
    """Generate priority visualizations for dataset breakdowns."""
    
    # Identify dataset grouping columns
    dataset_cols = []
    for col in ['study_accession', 'project_name', 'accession']:
        if col in obs.columns:
            dataset_cols.append(col)
    
    if not dataset_cols:
        logger.warning("No dataset grouping columns found")
        return
    
    # Group by first available dataset column
    dataset_col = dataset_cols[0]
    
    # Create comprehensive breakdown figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Samples per Dataset',
            'Samples per Instrument Model',
            'Samples per Sequencing Method',
            'Samples per Primer/Region',
            'Samples per Library Strategy',
            'Dataset Size Distribution'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'histogram'}]
        ]
    )
    
    # 1. Samples per dataset
    dataset_counts = obs[dataset_col].value_counts().head(30)
    fig.add_trace(
        go.Bar(
            x=dataset_counts.index,
            y=dataset_counts.values,
            name='Datasets',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    
    # 2. Samples per instrument model
    if 'instrument_model' in obs.columns:
        instrument_counts = obs['instrument_model'].value_counts().head(15)
        fig.add_trace(
            go.Bar(
                x=instrument_counts.index,
                y=instrument_counts.values,
                name='Instruments',
                marker_color='coral'
            ),
            row=1, col=2
        )
        fig.update_xaxes(tickangle=-45, row=1, col=2)
    
    # 3. Samples per sequencing method
    if 'sequencing_method' in obs.columns:
        method_counts = obs['sequencing_method'].value_counts().head(15)
        fig.add_trace(
            go.Bar(
                x=method_counts.index,
                y=method_counts.values,
                name='Methods',
                marker_color='mediumseagreen'
            ),
            row=2, col=1
        )
        fig.update_xaxes(tickangle=-45, row=2, col=1)
    
    # 4. Samples per primer/region
    primer_col = None
    for col in ['target_subfragment', 'pcr_primers', 'target_gene']:
        if col in obs.columns:
            primer_col = col
            break
    
    if primer_col:
        primer_counts = obs[primer_col].value_counts().head(15)
        fig.add_trace(
            go.Bar(
                x=primer_counts.index,
                y=primer_counts.values,
                name='Primers/Regions',
                marker_color='mediumpurple'
            ),
            row=2, col=2
        )
        fig.update_xaxes(tickangle=-45, row=2, col=2)
    
    # 5. Samples per library strategy
    if 'library_strategy' in obs.columns:
        lib_counts = obs['library_strategy'].value_counts().head(15)
        fig.add_trace(
            go.Bar(
                x=lib_counts.index,
                y=lib_counts.values,
                name='Library Strategy',
                marker_color='gold'
            ),
            row=3, col=1
        )
        fig.update_xaxes(tickangle=-45, row=3, col=1)
    
    # 6. Dataset size distribution
    dataset_sizes = obs.groupby(dataset_col).size()
    fig.add_trace(
        go.Histogram(
            x=dataset_sizes.values,
            nbinsx=30,
            name='Dataset Sizes',
            marker_color='indianred'
        ),
        row=3, col=2
    )
    fig.update_xaxes(title_text="Samples per Dataset", row=3, col=2)
    
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text=f"Dataset Breakdown Overview (n={len(obs):,} samples, {len(dataset_sizes)} datasets)"
    )
    
    fig.write_html(output_dir / 'dataset_breakdown.html')
    logger.info(f"  ✓ Dataset breakdown visualizations saved ({len(dataset_sizes)} datasets)")
    
    # Also create a detailed instrument × primer combination table
    create_instrument_primer_matrix(obs, output_dir)


def create_instrument_primer_matrix(obs: pd.DataFrame, output_dir: Path):
    """Create a cross-tabulation of instrument models and primer pairs."""
    
    instrument_col = 'instrument_model' if 'instrument_model' in obs.columns else None
    primer_col = None
    for col in ['target_subfragment', 'pcr_primers', 'target_gene']:
        if col in obs.columns:
            primer_col = col
            break
    
    if not instrument_col or not primer_col:
        return
    
    # Create cross-tabulation
    crosstab = pd.crosstab(
        obs[instrument_col], 
        obs[primer_col],
        margins=True,
        margins_name='Total'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        colorscale='Blues',
        text=crosstab.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Sample Count")
    ))
    
    fig.update_layout(
        title=f'Sample Count: Instrument Model × Primer/Region',
        xaxis=dict(title=primer_col.replace('_', ' ').title(), tickangle=-45),
        yaxis=dict(title='Instrument Model'),
        height=max(400, len(crosstab) * 30),
        width=max(800, len(crosstab.columns) * 60)
    )
    
    fig.write_html(output_dir / 'instrument_primer_matrix.html')
    logger.info(f"  ✓ Instrument × Primer matrix saved")


def generate_dataset_table(obs: pd.DataFrame, output_dir: Path):
    """Generate comprehensive dataset table with ENA links and citations."""
    
    # Identify dataset column
    dataset_col = None
    for col in ['study_accession', 'project_name', 'accession']:
        if col in obs.columns:
            dataset_col = col
            break
    
    if not dataset_col:
        logger.warning("No dataset column found for table generation")
        return
    
    logger.info("Generating dataset table with ENA links and citations...")
    
    # Group by dataset and collect statistics
    datasets = []
    
    for dataset_id in obs[dataset_col].unique():
        if pd.isna(dataset_id):
            continue
        
        dataset_samples = obs[obs[dataset_col] == dataset_id]
        
        # Get study accession for ENA link
        study_acc = None
        if 'study_accession' in dataset_samples.columns:
            study_acc = dataset_samples['study_accession'].iloc[0]
        elif dataset_col == 'study_accession':
            study_acc = dataset_id
        
        # Fetch publication info
        pub_info = None
        if study_acc and isinstance(study_acc, str) and study_acc.startswith(('PRJ', 'ERP', 'DRP', 'SRP')):
            pub_info = fetch_publication_info(study_acc)
        
        dataset_info = {
            'Dataset': dataset_id,
            'Samples': len(dataset_samples),
            'Study_Accession': study_acc if study_acc else dataset_id,
            'ENA_URL': f"https://www.ebi.ac.uk/ena/browser/view/{study_acc}" if study_acc else '',
        }
        
        # Add publication info if available
        if pub_info:
            if 'pubmed_url' in pub_info:
                dataset_info['PubMed_URL'] = pub_info['pubmed_url']
            if 'doi_url' in pub_info:
                dataset_info['DOI_URL'] = pub_info['doi_url']
            if 'citation' in pub_info:
                dataset_info['Citation'] = pub_info['citation']
        
        # Add metadata statistics
        if 'instrument_model' in dataset_samples.columns:
            instruments = dataset_samples['instrument_model'].dropna().unique()
            dataset_info['Instruments'] = ', '.join([str(x) for x in instruments[:3]])
            if len(instruments) > 3:
                dataset_info['Instruments'] += f' (+{len(instruments)-3} more)'
        
        if 'sequencing_method' in dataset_samples.columns:
            methods = dataset_samples['sequencing_method'].dropna().unique()
            dataset_info['Sequencing_Method'] = ', '.join([str(x) for x in methods[:2]])
        
        primer_col = None
        for col in ['target_subfragment', 'pcr_primers', 'target_gene']:
            if col in dataset_samples.columns:
                primer_col = col
                break
        
        if primer_col:
            primers = dataset_samples[primer_col].dropna().unique()
            dataset_info['Primer/Region'] = ', '.join([str(x) for x in primers[:2]])
            if len(primers) > 2:
                dataset_info['Primer/Region'] += f' (+{len(primers)-2} more)'
        
        if 'collection_year' in dataset_samples.columns:
            years = dataset_samples['collection_year'].dropna()
            if len(years) > 0:
                dataset_info['Year_Range'] = f"{int(years.min())}-{int(years.max())}"
        
        if 'env_biome' in dataset_samples.columns:
            biomes = dataset_samples['env_biome'].dropna().unique()
            dataset_info['Biomes'] = ', '.join([str(x) for x in biomes[:2]])
            if len(biomes) > 2:
                dataset_info['Biomes'] += f' (+{len(biomes)-2} more)'
        
        datasets.append(dataset_info)
    
    # Create DataFrame and sort by sample count
    dataset_df = pd.DataFrame(datasets).sort_values('Samples', ascending=False)
    
    # Save CSV
    dataset_df.to_csv(output_dir / 'dataset_summary_table.csv', index=False)
    logger.info(f"  ✓ Dataset summary table saved ({len(dataset_df)} datasets)")
    
    # Generate HTML table with clickable links
    generate_dataset_html_table(dataset_df, output_dir)


def generate_dataset_html_table(dataset_df: pd.DataFrame, output_dir: Path):
    """Generate interactive HTML table with clickable links."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Summary Table</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #3498db; color: white; position: sticky; top: 0; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #e8f4f8; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .citation { font-size: 12px; color: #555; font-style: italic; }
            .stats { font-weight: bold; color: #27ae60; }
            .link-btn { 
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                font-size: 11px;
            }
            .link-btn:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <h1>Dataset Summary Table</h1>
        <p><strong>Total Datasets:</strong> """ + f"{len(dataset_df):,}" + """</p>
        <p><strong>Total Samples:</strong> """ + f"{dataset_df['Samples'].sum():,}" + """</p>
        
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Samples</th>
                    <th>Links</th>
                    <th>Instruments</th>
                    <th>Method</th>
                    <th>Primer/Region</th>
                    <th>Years</th>
                    <th>Biomes</th>
                    <th>Publication</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in dataset_df.iterrows():
        html += "<tr>\n"
        html += f"<td><strong>{row['Dataset']}</strong></td>\n"
        html += f"<td class='stats'>{row['Samples']:,}</td>\n"
        
        # Links column
        links = []
        if row.get('ENA_URL'):
            links.append(f"<a href='{row['ENA_URL']}' target='_blank' class='link-btn'>ENA</a>")
        if row.get('PubMed_URL'):
            links.append(f"<a href='{row['PubMed_URL']}' target='_blank' class='link-btn'>PubMed</a>")
        if row.get('DOI_URL'):
            links.append(f"<a href='{row['DOI_URL']}' target='_blank' class='link-btn'>DOI</a>")
        html += f"<td>{''.join(links) if links else '-'}</td>\n"
        
        html += f"<td>{row.get('Instruments', '-')}</td>\n"
        html += f"<td>{row.get('Sequencing_Method', '-')}</td>\n"
        html += f"<td>{row.get('Primer/Region', '-')}</td>\n"
        html += f"<td>{row.get('Year_Range', '-')}</td>\n"
        html += f"<td>{row.get('Biomes', '-')}</td>\n"
        
        # Citation column
        citation = row.get('Citation', '')
        if citation:
            html += f"<td class='citation'>{citation}</td>\n"
        else:
            html += "<td>-</td>\n"
        
        html += "</tr>\n"
    
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(output_dir / 'dataset_summary_table.html', 'w') as f:
        f.write(html)
    
    logger.info(f"  ✓ Interactive dataset table HTML saved")


def generate_sample_map(adata: AnnData, output_dir: Path):
    """Create interactive map of sample locations with nuclear facilities."""
    obs = adata.obs.copy()
    
    # Filter valid coordinates
    valid_coords = obs.dropna(subset=['latitude', 'longitude'])
    if len(valid_coords) == 0:
        logger.warning("No valid coordinates for map")
        return
    
    # Limit to reasonable geographic ranges
    valid_coords = valid_coords[
        (valid_coords['latitude'].between(-90, 90)) & 
        (valid_coords['longitude'].between(-180, 180))
    ]
    
    # Create base map
    fig = go.Figure()
    
    # Add nuclear facilities if available
    if 'facility_latitude' in obs.columns and 'facility_longitude' in obs.columns:
        facilities = obs.dropna(subset=['facility_latitude', 'facility_longitude'])[
            ['facility_latitude', 'facility_longitude', 'facility_name']
        ].drop_duplicates()
        
        if len(facilities) > 0:
            fig.add_trace(go.Scattergeo(
                lon=facilities['facility_longitude'],
                lat=facilities['facility_latitude'],
                text=facilities['facility_name'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', line=dict(width=2, color='darkred')),
                name='Nuclear Facilities',
                hovertemplate='<b>%{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
    
    # Color by various categorical columns
    color_columns = []
    for col in ['project_name', 'env_biome', 'env_feature', 'sequencing_method', 'facility_match']:
        if col in valid_coords.columns and valid_coords[col].nunique() < 20:
            color_columns.append(col)
    
    # Create maps for each color category
    for i, color_col in enumerate(color_columns[:5]):  # Limit to 5
        visible = i == 0  # Only first trace visible by default
        
        for category in valid_coords[color_col].unique():
            if pd.isna(category):
                continue
            
            subset = valid_coords[valid_coords[color_col] == category]
            fig.add_trace(go.Scattergeo(
                lon=subset['longitude'],
                lat=subset['latitude'],
                text=[f"{color_col}: {category}" for _ in range(len(subset))],
                mode='markers',
                marker=dict(size=6, opacity=0.6),
                name=f'{category}',
                visible=visible,
                hovertemplate='<b>%{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
    
    # Add numeric metadata overlays
    numeric_cols = []
    for col in ['n_counts', 'facility_age_at_sampling', 'facility_distance_km', 'collection_year']:
        if col in valid_coords.columns:
            numeric_cols.append(col)
    
    for num_col in numeric_cols[:3]:  # Limit to 3
        # --- FIX: Safe Numeric Conversion ---
        # Ensure we have numeric data before plotting to prevent "Unknown format code 'f'" error
        subset = valid_coords.copy()
        subset[num_col] = pd.to_numeric(subset[num_col], errors='coerce')
        subset = subset.dropna(subset=[num_col])
        
        if len(subset) > 0:
            fig.add_trace(go.Scattergeo(
                lon=subset['longitude'],
                lat=subset['latitude'],
                text=[f"{num_col}: {val:.2f}" for val in subset[num_col]],
                mode='markers',
                marker=dict(
                    size=6,
                    color=subset[num_col],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=num_col)
                ),
                name=num_col,
                visible=False,
                hovertemplate='<b>%{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
    
    # Create dropdown menu
    buttons = []
    n_facility_traces = 1 if 'facility_latitude' in obs.columns else 0
    
    # Categorical color options
    for i, color_col in enumerate(color_columns[:5]):
        n_categories = valid_coords[color_col].nunique()
        visibility = [False] * len(fig.data)
        visibility[:n_facility_traces] = [True] * n_facility_traces  # Keep facilities visible
        
        # Calculate trace indices for this category
        start_idx = n_facility_traces + sum([valid_coords[c].nunique() for c in color_columns[:i]])
        end_idx = start_idx + n_categories
        visibility[start_idx:end_idx] = [True] * n_categories
        
        buttons.append(dict(
            label=f"Color by {color_col}",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # Numeric color options
    cat_traces = n_facility_traces + sum([valid_coords[c].nunique() for c in color_columns[:5]])
    for i, num_col in enumerate(numeric_cols[:3]):
        visibility = [False] * len(fig.data)
        visibility[:n_facility_traces] = [True] * n_facility_traces
        visibility[cat_traces + i] = True
        
        buttons.append(dict(
            label=f"Color by {num_col}",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    fig.update_layout(
        title=f'Sample Locations (n={len(valid_coords):,})',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            showocean=True,
            oceancolor='lightblue'
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top"
        )],
        height=600
    )
    
    fig.write_html(output_dir / 'sample_map_interactive.html')
    logger.info(f"  ✓ Interactive sample map saved ({len(valid_coords):,} samples)")


def generate_numeric_heatmap(obs: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
    """Generate correlation heatmap of numeric metadata."""
    
    # Limit to reasonable number of columns
    if len(numeric_cols) > 30:
        # Select most variable columns
        variances = obs[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(30).index.tolist()
    
    # Calculate correlation matrix
    corr_data = obs[numeric_cols].dropna(how='all', axis=1)
    if len(corr_data.columns) < 2:
        return
    
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f'Numeric Metadata Correlations (n={len(corr_matrix)} variables)',
        xaxis=dict(tickangle=-45),
        height=max(500, len(corr_matrix) * 20),
        width=max(600, len(corr_matrix) * 20)
    )
    
    fig.write_html(output_dir / 'numeric_metadata_heatmap.html')
    logger.info(f"  ✓ Numeric metadata heatmap saved ({len(corr_matrix)} variables)")


def generate_sample_distribution_plots(obs: pd.DataFrame, output_dir: Path):
    """Generate plots showing sample distributions."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Samples per Study',
            'Samples per Year',
            'Sequencing Depth Distribution',
            'Feature Count Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'histogram'}]]
    )
    
    # Samples per study
    if 'project_name' in obs.columns:
        study_counts = obs['project_name'].value_counts().head(20)
        fig.add_trace(
            go.Bar(x=study_counts.index, y=study_counts.values, name='Studies'),
            row=1, col=1
        )
    
    # Samples per year
    if 'collection_year' in obs.columns:
        year_counts = obs['collection_year'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=year_counts.index, y=year_counts.values, name='Years'),
            row=1, col=2
        )
    
    # Sequencing depth
    if 'n_counts' in obs.columns:
        fig.add_trace(
            go.Histogram(x=np.log10(obs['n_counts'].dropna() + 1), nbinsx=50, name='Depth'),
            row=2, col=1
        )
        fig.update_xaxes(title_text="log10(Read Count)", row=2, col=1)
    
    # Feature count
    if 'n_genes_by_counts' in obs.columns:
        fig.add_trace(
            go.Histogram(x=obs['n_genes_by_counts'].dropna(), nbinsx=50, name='Features'),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Features per Sample", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Sample Distribution Overview (n={len(obs):,} samples)"
    )
    
    fig.write_html(output_dir / 'sample_distributions.html')
    logger.info(f"  ✓ Sample distribution plots saved")


def generate_category_plots(obs: pd.DataFrame, output_dir: Path):
    """Generate plots for key categorical variables."""
    
    # Find important categorical columns
    cat_cols = obs.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Priority columns to visualize
    priority = ['env_biome', 'env_feature', 'sequencing_method', 'instrument_model', 
                'library_selection', 'facility_match', 'nuclear_contamination_status']
    priority = [col for col in priority if col in cat_cols]
    
    if not priority:
        return
    
    # Create bar plots for each
    n_cols = min(len(priority), 6)
    rows = (n_cols + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[col.replace('_', ' ').title() for col in priority[:n_cols]]
    )
    
    for i, col in enumerate(priority[:n_cols]):
        row = i // 2 + 1
        col_idx = i % 2 + 1
        
        value_counts = obs[col].value_counts().head(15)
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col,
                showlegend=False
            ),
            row=row, col=col_idx
        )
        
        fig.update_xaxes(tickangle=-45, row=row, col=col_idx)
    
    fig.update_layout(
        height=300 * rows,
        title_text=f"Categorical Metadata Overview (n={len(obs):,} samples)"
    )
    
    fig.write_html(output_dir / 'categorical_distributions.html')
    logger.info(f"  ✓ Categorical distribution plots saved")


def generate_html_report(profile_results: Dict, output_path: Path):
    """Generate HTML version of metadata profiling report."""
    
    output_dir = output_path.parent
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Metadata Profiling Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .error {{ background-color: #e74c3c; color: white; padding: 10px; margin: 10px 0; }}
            .warning {{ background-color: #f39c12; color: white; padding: 10px; margin: 10px 0; }}
            .info {{ background-color: #3498db; color: white; padding: 10px; margin: 10px 0; }}
            .success {{ background-color: #27ae60; color: white; padding: 10px; margin: 10px 0; }}
            .viz-link {{ 
                display: inline-block; 
                background-color: #3498db; 
                color: white; 
                padding: 10px 20px; 
                margin: 10px 5px; 
                text-decoration: none; 
                border-radius: 5px; 
            }}
            .viz-link:hover {{ background-color: #2980b9; }}
        </style>
    </head>
    <body>
        <h1>Metadata Profiling Report</h1>
        
        <h2>Interactive Visualizations</h2>
        <div>
    """
    
    # Add links to visualizations if they exist (prioritize dataset visualizations)
    viz_files = [
        ('dataset_summary_table.html', '📋 Dataset Summary Table (with ENA links & citations)'),
        ('dataset_breakdown.html', '📊 Dataset Breakdown by Instrument/Primers/Method'),
        ('instrument_primer_matrix.html', '🔬 Instrument × Primer Matrix'),
        ('sample_map_interactive.html', '🗺️ Interactive Sample Map'),
        ('numeric_metadata_heatmap.html', '📊 Numeric Metadata Heatmap'),
        ('sample_distributions.html', '📈 Sample Distributions'),
        ('categorical_distributions.html', '📊 Categorical Distributions')
    ]
    
    for filename, label in viz_files:
        if (output_dir / filename).exists():
            html += f'<a href="{filename}" class="viz-link">{label}</a>\n'
    
    html += """
        </div>
        
        <h2>Dataset Overview</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """
    
    for key, value in profile_results['dataset_stats'].items():
        html += f"<tr><td>{key}</td><td>{value:,}</td></tr>\n"
    
    html += "</table>\n"
    
    # Warnings section
    if profile_results['warnings'] is not None and len(profile_results['warnings']) > 0:
        html += "<h2>ML Readiness Warnings</h2>\n"
        
        for _, row in profile_results['warnings'].iterrows():
            css_class = 'error' if row['severity'] == 'ERROR' else 'warning'
            html += f'<div class="{css_class}"><strong>{row["severity"]}</strong>: {row["message"]}</div>\n'
    else:
        html += '<div class="success">✓ No ML warnings - data appears ready for analysis</div>\n'
    
    # Completeness table (top 20 most complete)
    html += "<h2>Metadata Completeness (Top 20)</h2>\n"
    html += "<table><tr><th>Column</th><th>% Complete</th><th>Missing</th></tr>\n"
    
    for _, row in profile_results['completeness'].head(20).iterrows():
        html += f"<tr><td>{row['column']}</td><td>{row['pct_full']:.1f}%</td><td>{row['n_missing']:,}</td></tr>\n"
    
    html += "</table>\n</body>\n</html>"
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report saved to: {output_path}")