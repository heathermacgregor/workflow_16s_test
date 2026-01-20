# Scientific Enhancements to workflow_16s

## Overview

This document describes the state-of-the-art statistical and methodological enhancements added to the workflow_16s pipeline in January 2026. These additions address key limitations in microbiome analysis and bring the pipeline in line with current best practices.

---

## New Features

### 1. PERMDISP - Homogeneity of Dispersion Testing

**Location:** `downstream/diversity/beta/dispersion.py`

**Purpose:** Validates PERMANOVA results by testing if group dispersions are homogeneous.

**Why it matters:**
- PERMANOVA can be confounded by dispersion differences, not just location differences
- Significant PERMDISP (p < 0.05) invalidates PERMANOVA interpretation
- Critical for publication-quality beta diversity analyses

**Usage:**
```python
from workflow_16s.downstream.diversity.beta import run_permdisp, check_permanova_validity

# Run PERMDISP
permdisp_result = run_permdisp(distance_matrix, grouping, permutations=9999)

# Validate PERMANOVA results
validation = check_permanova_validity(permanova_result, permdisp_result)
print(validation['message'])
```

**Output:**
- Test statistic (F-value)
- P-value from permutations
- Interpretation guidance
- Warning flag if dispersions differ

---

### 2. Effect Size Calculations

**Location:** `downstream/statistics/effect_sizes.py`

**Purpose:** Quantify biological significance alongside statistical significance.

**Why it matters:**
- P-values alone don't indicate biological importance
- Effect sizes distinguish trivial from meaningful differences
- Required by many journals (e.g., Nature, Cell)

**Available metrics:**
- **Cohen's d**: Standardized mean difference (parametric)
- **Cliff's Delta**: Non-parametric alternative (robust to outliers)
- **Glass's Delta**: Uses control group SD only
- **Hedges' g**: Bias-corrected for small samples

**Usage:**
```python
from workflow_16s.downstream.statistics import calculate_all_effect_sizes

# Compare two groups
effect_sizes = calculate_all_effect_sizes(
    group1_values, 
    group2_values,
    group_names=('Contaminated', 'Control')
)

print(f"Cohen's d: {effect_sizes['cohens_d']:.3f} ({effect_sizes['cohens_d_interpretation']})")
print(f"Cliff's Delta: {effect_sizes['cliffs_delta']:.3f}")
```

**Interpretation thresholds:**
- Cohen's d: < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), ≥ 0.8 (large)
- Cliff's Delta: < 0.147 (negligible), 0.147-0.33 (small), 0.33-0.474 (medium), ≥ 0.474 (large)

---

### 3. Rarefaction Curves

**Location:** `downstream/diversity/alpha/rarefaction.py`

**Purpose:** Assess sampling adequacy and diversity saturation.

**Why it matters:**
- Validates that sequencing depth is sufficient
- Identifies samples needing deeper sequencing
- Standard diagnostic for alpha diversity analyses

**Usage:**
```python
from workflow_16s.downstream.diversity.alpha import generate_rarefaction_curves

# Generate curves for all samples
rarefaction_data = generate_rarefaction_curves(
    adata,
    output_dir=plot_dir,
    metric='observed_features',
    n_depths=20,
    group_col='nuclear_contamination_status'
)
```

**Output:**
- Individual sample curves (HTML interactive plot)
- Group-averaged curves
- Plateau detection (< 5% increase in final 20% of depth)
- Sampling adequacy summary

---

### 4. Enhanced Multiple Testing Correction

**Location:** `downstream/statistics/multiple_testing.py`

**Purpose:** Control false discovery rate with modern methods.

**Why it matters:**
- Bonferroni is too conservative for high-dimensional microbiome data
- FDR methods (Benjamini-Hochberg) provide better power
- Different methods appropriate for different scenarios

**Available methods:**
- `fdr_bh`: Benjamini-Hochberg (recommended for independent tests)
- `fdr_by`: Benjamini-Yekutieli (for dependent tests, e.g., taxonomic levels)
- `bonferroni`: Conservative family-wise error rate
- `holm`: Sequential Bonferroni
- `sidak`: Less conservative than Bonferroni

**Usage:**
```python
from workflow_16s.downstream.statistics import apply_multiple_testing_correction

# Apply FDR correction
reject, p_adjusted, _ = apply_multiple_testing_correction(
    p_values,
    method='fdr_bh',
    alpha=0.05
)

# Stratified FDR (within taxonomic levels)
reject, p_adjusted = stratified_fdr_correction(
    p_values,
    strata=taxonomic_levels,
    method='fdr_bh'
)
```

---

### 5. Nested Cross-Validation

**Location:** `downstream/machine_learning/nested_cv.py`

**Purpose:** Unbiased performance estimation for machine learning models.

**Why it matters:**
- Simple CV overestimates performance (optimistic bias)
- Nested CV prevents data leakage during hyperparameter tuning
- Required for publication-quality ML results

**Architecture:**
- **Outer loop** (5 folds): Evaluates generalization performance
- **Inner loop** (3 folds): Optimizes hyperparameters
- Total models trained: 5 × 3 × n_param_combinations

**Usage:**
```python
from workflow_16s.downstream.machine_learning import nested_cross_validation

# Run nested CV
results = nested_cross_validation(
    X=feature_matrix,
    y=target_variable,
    task_type='classification',
    outer_cv=5,
    inner_cv=3,
    n_jobs=-1
)

print(f"Unbiased performance: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
print(f"Best params per fold: {results['best_params_per_fold']}")
```

**Compare with simple CV:**
```python
from workflow_16s.downstream.machine_learning import compare_with_simple_cv

comparison = compare_with_simple_cv(X, y, task_type='classification')
print(f"Simple CV: {comparison['simple_cv_score']:.3f} (BIASED)")
print(f"Nested CV: {comparison['nested_cv_score']:.3f} (UNBIASED)")
print(f"Optimistic bias: {comparison['bias']:.3f}")
```

---

### 6. ANCOM-BC Differential Abundance

**Location:** `downstream/statistics/differential_abundance.py`

**Purpose:** Compositionally aware differential abundance testing.

**Why it matters:**
- Traditional t-tests/Mann-Whitney ignore compositionality
- ANCOM-BC corrects for sampling fraction bias
- Handles batch effects and confounders
- State-of-the-art method (Nature Communications 2020)

**Requirements:**
- R with ANCOMBC package installed
- rpy2 Python package

**Installation:**
```bash
# Install rpy2
pip install rpy2

# Install ANCOMBC in R
R -e "install.packages('BiocManager'); BiocManager::install('ANCOMBC')"
```

**Usage:**
```python
from workflow_16s.downstream.statistics import ancom_bc_wrapper

# Run ANCOM-BC
results = ancom_bc_wrapper(
    adata,
    group_col='nuclear_contamination_status',
    formula='~nuclear_contamination_status + batch',
    output_dir=output_dir,
    p_adj_method='fdr_bh',
    alpha=0.05
)

# View significant features
significant = results[results['significant']]
print(f"Found {len(significant)} differentially abundant features")
```

**Fallback method (no R required):**
```python
from workflow_16s.downstream.statistics import simple_compositional_da

# CLR + Mann-Whitney
results = simple_compositional_da(
    adata,
    group_col='nuclear_contamination_status',
    method='mannwhitneyu',
    fdr_method='fdr_bh'
)
```

---

### 7. Increased PERMANOVA Permutations

**Location:** `downstream/diversity_old.py` (line 422)

**Change:** `permutations=999` → `permutations=9999`

**Why it matters:**
- More permutations = more robust p-value estimation
- Recommended minimum: 9999 for publication
- Reduces Monte Carlo error in p-value

**Impact:**
- More stable p-values across repeated runs
- Better resolution for small p-values (e.g., p < 0.001)
- Slightly longer computation time (~10x)

---

## Taxonomy Database Updates

### Current: SILVA 138-99 (2019)

### Recommended Updates:

#### Option 1: SILVA 138.1 (2020)
- Minor update to SILVA 138
- Improved curation
- Same V4 region compatibility

#### Option 2: GTDB r220 (2024)
- Prokaryote-specific (bacteria + archaea)
- Phylogenetically consistent taxonomy
- Higher resolution for environmental samples
- Requires re-downloading classifier

**Update procedure:**
```bash
# Download GTDB classifier (example for V4 region)
cd workflow_16s/resources/classifier
wget https://data.gtdb.ecogenomic.org/releases/latest/gtdb_qiime2_classifiers/V4_515-806_gtdb_r220.qza

# Update config.yaml
qiime2:
  per_dataset:
    taxonomy:
      classifier_dir: "../resources/classifier/gtdb_r220"
      classifier: "V4_515-806_gtdb_r220"
```

---

## Integration with Existing Workflow

### Modular Design

All new features are **optional** and can be enabled individually:

```python
# In downstream analysis
from workflow_16s.downstream.diversity.beta import run_permdisp
from workflow_16s.downstream.statistics import calculate_all_effect_sizes
from workflow_16s.downstream.diversity.alpha import generate_rarefaction_curves

# Add to your analysis pipeline as needed
```

### Backward Compatibility

- Existing code continues to work unchanged
- New features are opt-in additions
- No breaking changes to core functionality

---

## Performance Considerations

### Computational Cost

| Feature | Time Complexity | Memory | Recommendation |
|---------|----------------|---------|----------------|
| PERMDISP | Same as PERMANOVA | Low | Always run with PERMANOVA |
| Effect sizes | O(n) | Low | No overhead |
| Rarefaction | O(n × d × i) | Medium | Limit to 50 samples for plots |
| Nested CV | O(k² × m) | Medium | Use n_jobs=-1 for parallelization |
| ANCOM-BC | O(n × m) | High | May take 5-10 min for 1000+ features |
| FDR correction | O(n log n) | Low | No overhead |

Where:
- n = number of samples
- d = number of depth points
- i = number of iterations
- k = number of CV folds
- m = number of features

### Optimization Tips

1. **Rarefaction curves**: Use `max_samples_to_plot=50` for large datasets
2. **Nested CV**: Use `n_jobs=-1` to utilize all CPU cores
3. **PERMDISP**: Run in parallel with PERMANOVA (same permutation matrix)
4. **ANCOM-BC**: Pre-filter low-abundance features (< 10 reads total)

---

## Validation and Testing

All new features have been tested against:
- Published benchmarks (where available)
- R package implementations (PERMDISP, ANCOM-BC)
- Simulated data with known ground truth

### Example validation:

```python
# Test PERMDISP against R vegan::betadisper
from skbio.stats.distance import permdisp
# Results match R implementation within Monte Carlo error

# Test effect sizes against effsize R package
from workflow_16s.downstream.statistics import cohens_d
# Cohen's d matches within numerical precision
```

---

## Citation Guidelines

If you use these features, please cite:

**PERMDISP:**
> Anderson, M.J. (2006). Distance-based tests for homogeneity of multivariate dispersions. *Biometrics*, 62(1), 245-253.

**ANCOM-BC:**
> Lin, H., & Peddada, S. D. (2020). Analysis of compositions of microbiomes with bias correction. *Nature Communications*, 11(1), 3514.

**Nested CV:**
> Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7(1), 91.

**Benjamini-Hochberg FDR:**
> Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

---

## Future Enhancements (Roadmap)

### High Priority (Next 3 months)
- [ ] Integration with MaAsLin 2 for multivariable associations
- [ ] Phylogenetic placement with SEPP
- [ ] Contamination screening with decontam
- [ ] Batch effect visualization (RLE plots, PCA on batch)

### Medium Priority (3-6 months)
- [ ] Graph Neural Networks for network-based predictions
- [ ] Metabolic pathway enrichment (GSEA-style)
- [ ] Longitudinal mixed-effects models
- [ ] Zero-inflated negative binomial models

### Low Priority (6-12 months)
- [ ] Deep learning autoencoders for dimensionality reduction
- [ ] Integration with metabolomics data
- [ ] Real-time streaming analysis
- [ ] Cloud deployment (AWS/GCP)

---

## Troubleshooting

### ANCOM-BC fails with "R package not found"

**Solution:**
```bash
# Verify R installation
which R

# Install ANCOMBC in R
R
> install.packages("BiocManager")
> BiocManager::install("ANCOMBC")
> quit()

# Test rpy2
python -c "from rpy2.robjects.packages import importr; importr('ANCOMBC')"
```

### Nested CV runs out of memory

**Solution:**
- Reduce outer/inner CV folds
- Reduce feature set size
- Use `n_jobs=1` to limit parallelism
- Filter low-variance features first

### Rarefaction curves are too slow

**Solution:**
- Reduce `n_iterations` from 10 to 5
- Reduce `n_depths` from 20 to 10
- Use `max_samples_to_plot=20` instead of 50

---

## Contact & Support

For questions or issues with new features:
1. Check this documentation
2. Review code docstrings (all functions fully documented)
3. Open GitHub issue with minimal reproducible example

---

**Last Updated:** January 7, 2026
**Version:** 2.0 (State-of-the-art enhancements)
