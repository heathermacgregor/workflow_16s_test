# Quick Start Guide: New Statistical Features

## 🚀 Getting Started with Enhanced Analyses

This guide shows how to integrate the new state-of-the-art statistical methods into your workflow_16s analysis.

---

## 1. Beta Diversity with PERMDISP Validation

**Add to your beta diversity analysis:**

```python
from workflow_16s.downstream.diversity.beta import run_beta_diversity_and_stats, run_permdisp, check_permanova_validity

# Run beta diversity (existing code)
beta_results = run_beta_diversity_and_stats(
    adata, 
    plot_dir_beta, 
    group_col='nuclear_contamination_status'
)

# NEW: Validate PERMANOVA with PERMDISP
for metric_name, distance_matrix in beta_results['distance_matrices'].items():
    # Run PERMDISP
    permdisp_result = run_permdisp(
        distance_matrix, 
        grouping=adata.obs['nuclear_contamination_status'],
        permutations=9999  # Increased from 999
    )
    
    # Check if PERMANOVA is valid
    validation = check_permanova_validity(
        beta_results['permanova'][metric_name],
        permdisp_result
    )
    
    logger.info(f"{metric_name}: {validation['message']}")
```

**Interpretation:**
- ✓ Valid: PERMANOVA p < 0.05 AND PERMDISP p ≥ 0.05 → True compositional differences
- ⚠️ Invalid: PERMANOVA p < 0.05 AND PERMDISP p < 0.05 → May be dispersion, not location

---

## 2. Alpha Diversity with Effect Sizes

**Enhance alpha diversity testing:**

```python
from workflow_16s.downstream.diversity import run_alpha_diversity
from workflow_16s.downstream.statistics import calculate_all_effect_sizes

# Calculate alpha diversity (existing)
run_alpha_diversity(adata, plot_dir_alpha)

# NEW: Add effect size for significant results
contaminated = adata.obs[adata.obs['nuclear_contamination_status'] == True]['shannon']
control = adata.obs[adata.obs['nuclear_contamination_status'] == False]['shannon']

effect_sizes = calculate_all_effect_sizes(
    contaminated.values,
    control.values,
    group_names=('Contaminated', 'Control')
)

logger.info(f"Shannon diversity effect size:")
logger.info(f"  Cohen's d: {effect_sizes['cohens_d']:.3f} ({effect_sizes['cohens_d_interpretation']})")
logger.info(f"  Cliff's Delta: {effect_sizes['cliffs_delta']:.3f} ({effect_sizes['cliffs_delta_interpretation']})")
logger.info(f"  Biological significance: {effect_sizes['biological_significance']}")
```

---

## 3. Rarefaction Curve Diagnostics

**Add before running alpha diversity:**

```python
from workflow_16s.downstream.diversity.alpha import generate_rarefaction_curves

# NEW: Generate rarefaction curves
rarefaction_data = generate_rarefaction_curves(
    adata,
    output_dir=plot_dir_alpha,
    metric='observed_features',
    n_depths=20,
    n_iterations=10,
    max_samples_to_plot=50,
    group_col='nuclear_contamination_status'
)

# Check for plateau
plateau_samples = rarefaction_data.groupby('sample_id').apply(
    lambda x: (x['observed_features_mean'].iloc[-1] - x['observed_features_mean'].iloc[0]) 
              / x['observed_features_mean'].iloc[0] < 0.05
).sum()

logger.info(f"Samples with plateau: {plateau_samples}/{adata.n_obs}")
```

**What to look for:**
- Curves that plateau → Adequate sequencing depth
- Curves still increasing → Consider deeper sequencing
- Group differences in curves → Potential sequencing depth bias

---

## 4. Machine Learning with Nested CV

**Replace simple CV with nested CV:**

```python
# OLD (biased):
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(model, params, cv=5)
# grid.fit(X, y)
# print(f"Score: {grid.best_score_}")  # OPTIMISTICALLY BIASED

# NEW (unbiased):
from workflow_16s.downstream.machine_learning import nested_cross_validation, compare_with_simple_cv

# Get unbiased performance estimate
nested_results = nested_cross_validation(
    X=X_clr,
    y=adata.obs['facility_match'],
    task_type='classification',
    outer_cv=5,
    inner_cv=3,
    random_state=42,
    n_jobs=-1
)

logger.info(f"Unbiased ROC-AUC: {nested_results['mean_score']:.3f} ± {nested_results['std_score']:.3f}")

# Compare with simple CV to quantify bias
comparison = compare_with_simple_cv(X_clr, adata.obs['facility_match'])
logger.info(f"Optimistic bias: {comparison['bias']:.3f} ({comparison['bias_percent']:.1f}%)")
```

---

## 5. Differential Abundance with FDR

**Add FDR correction to statistical tests:**

```python
from workflow_16s.downstream.statistics import (
    apply_multiple_testing_correction,
    export_fdr_results
)

# Run statistical tests (existing code produces p_values array)
# p_values = ... your statistical tests ...

# NEW: Apply Benjamini-Hochberg FDR instead of Bonferroni
reject, p_adjusted, _ = apply_multiple_testing_correction(
    p_values,
    method='fdr_bh',  # More powerful than Bonferroni
    alpha=0.05
)

# Export results with FDR
results_df = export_fdr_results(
    feature_names=adata.var_names,
    p_values=p_values,
    method='fdr_bh',
    additional_data={
        'log2_fold_change': fold_changes,
        'mean_abundance': mean_abundances
    },
    output_path=output_dir / 'fdr_corrected_results.csv'
)

logger.info(f"Significant features (FDR < 0.05): {reject.sum()}")
```

---

## 6. ANCOM-BC Compositional Analysis

**Use ANCOM-BC for differential abundance:**

```python
from workflow_16s.downstream.statistics import ancom_bc_wrapper, simple_compositional_da

# Try ANCOM-BC (requires R + ANCOMBC package)
try:
    ancom_results = ancom_bc_wrapper(
        adata,
        group_col='nuclear_contamination_status',
        formula='~nuclear_contamination_status + batch_original',  # Can include covariates
        output_dir=output_dir,
        p_adj_method='fdr_bh',
        alpha=0.05
    )
    
    # Top differentially abundant features
    top_features = ancom_results[ancom_results['significant']].head(20)
    logger.info(f"Top 20 differentially abundant taxa:\n{top_features[['feature', 'log_fold_change', 'q_value']]}")
    
except Exception as e:
    logger.warning(f"ANCOM-BC failed (R not configured?): {e}")
    logger.info("Falling back to CLR + Mann-Whitney...")
    
    # Fallback: Simple compositional DA (no R required)
    da_results = simple_compositional_da(
        adata,
        group_col='nuclear_contamination_status',
        method='mannwhitneyu',
        fdr_method='fdr_bh',
        output_dir=output_dir
    )
```

---

## 7. Complete Integration Example

**Full pipeline with all enhancements:**

```python
from workflow_16s.downstream.diversity import run_alpha_diversity
from workflow_16s.downstream.diversity.beta import run_beta_diversity_and_stats, run_permdisp
from workflow_16s.downstream.diversity.alpha import generate_rarefaction_curves
from workflow_16s.downstream.statistics import (
    calculate_all_effect_sizes,
    apply_multiple_testing_correction,
    ancom_bc_wrapper
)
from workflow_16s.downstream.machine_learning import nested_cross_validation

# 1. Quality Control: Rarefaction Curves
logger.info("=== STEP 1: Rarefaction Analysis ===")
rarefaction_data = generate_rarefaction_curves(
    adata, plot_dir_alpha, 
    group_col='nuclear_contamination_status'
)

# 2. Alpha Diversity with Effect Sizes
logger.info("=== STEP 2: Alpha Diversity ===")
run_alpha_diversity(adata, plot_dir_alpha, tree_path=tree_file)

# Calculate effect sizes for Shannon
contaminated = adata.obs[adata.obs['nuclear_contamination_status'] == True]['shannon']
control = adata.obs[adata.obs['nuclear_contamination_status'] == False]['shannon']
effect_sizes = calculate_all_effect_sizes(contaminated.values, control.values)
logger.info(f"Shannon effect size (Cliff's Delta): {effect_sizes['cliffs_delta']:.3f}")

# 3. Beta Diversity with PERMDISP
logger.info("=== STEP 3: Beta Diversity + PERMDISP ===")
beta_results = run_beta_diversity_and_stats(adata, plot_dir_beta, group_col='nuclear_contamination_status')

for metric in ['braycurtis', 'jaccard']:
    permdisp_result = run_permdisp(
        beta_results['distance_matrices'][metric],
        adata.obs['nuclear_contamination_status'],
        permutations=9999
    )

# 4. Differential Abundance with ANCOM-BC
logger.info("=== STEP 4: Differential Abundance ===")
da_results = ancom_bc_wrapper(
    adata,
    group_col='nuclear_contamination_status',
    output_dir=output_dir
)

# 5. Machine Learning with Nested CV
logger.info("=== STEP 5: Machine Learning ===")
X_clr = clr_transform(adata)
nested_results = nested_cross_validation(
    X_clr,
    adata.obs['facility_match'],
    task_type='classification',
    outer_cv=5,
    n_jobs=-1
)
logger.info(f"Unbiased classification performance: {nested_results['mean_score']:.3f}")

logger.info("=== Analysis Complete ===")
```

---

## 📊 Expected Output

### Console Output Example:

```
=== STEP 1: Rarefaction Analysis ===
Depth range: 1,234 to 45,678 reads
Samples reaching plateau: 127/150 (84.7%)
✓ 84.7% of samples show plateau - adequate sampling depth.

=== STEP 2: Alpha Diversity ===
Alpha metrics added to adata.obs: observed_features, shannon, faith_pd
Shannon effect size (Cliff's Delta): 0.487 (medium)

=== STEP 3: Beta Diversity + PERMDISP ===
PERMANOVA (Bray-Curtis): F=3.45, p=0.0012
PERMDISP (Bray-Curtis): F=0.89, p=0.4123
✓ No dispersion differences - PERMANOVA valid

=== STEP 4: Differential Abundance ===
ANCOM-BC: 47/1,234 features significant (3.8%)
Top feature: Pseudomonas (log2FC=2.34, q=0.0001)

=== STEP 5: Machine Learning ===
Nested CV: 0.847 ± 0.032 (unbiased estimate)
Simple CV: 0.891 (biased - DO NOT REPORT)
Optimistic bias: 0.044 (4.9%)
```

---

## 🎯 Best Practices

### 1. Always Run PERMDISP with PERMANOVA
```python
# DO THIS:
permanova_result = run_permanova(...)
permdisp_result = run_permdisp(...)
validation = check_permanova_validity(permanova_result, permdisp_result)

# NOT THIS:
permanova_result = run_permanova(...)  # Stop here - incomplete!
```

### 2. Report Effect Sizes with P-Values
```python
# DO THIS:
print(f"Shannon: MW U=123, p=0.003, Cliff's δ=0.42 (medium effect)")

# NOT THIS:
print(f"Shannon: p=0.003")  # P-value alone insufficient
```

### 3. Use Nested CV for ML
```python
# DO THIS (unbiased):
nested_results = nested_cross_validation(X, y, ...)
print(f"Performance: {nested_results['mean_score']}")

# NOT THIS (biased):
grid = GridSearchCV(model, params, cv=5).fit(X, y)
print(f"Performance: {grid.best_score_}")  # Overoptimistic!
```

### 4. Apply FDR Instead of Bonferroni
```python
# DO THIS (more power):
reject, p_adj, _ = apply_multiple_testing_correction(p_vals, method='fdr_bh')

# AVOID THIS (too conservative):
reject, p_adj, _ = apply_multiple_testing_correction(p_vals, method='bonferroni')
```

---

## 🔧 Troubleshooting

**Q: ANCOM-BC says "R package not found"**  
A: Install R and ANCOMBC:
```bash
R -e "install.packages('BiocManager'); BiocManager::install('ANCOMBC')"
```

**Q: Nested CV is very slow**  
A: Reduce folds or use fewer features:
```python
nested_cross_validation(X, y, outer_cv=3, inner_cv=2)  # Faster
```

**Q: Rarefaction curves crash with "Out of memory"**  
A: Reduce iterations or samples:
```python
generate_rarefaction_curves(..., n_iterations=5, max_samples_to_plot=20)
```

---

## 📚 References

See [SCIENTIFIC_ENHANCEMENTS.md](SCIENTIFIC_ENHANCEMENTS.md) for full citations and methodological details.

---

**Last Updated:** January 7, 2026
