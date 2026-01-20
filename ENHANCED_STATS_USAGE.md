# Enhanced Statistical Analysis - Usage Examples

This notebook demonstrates how to use the newly implemented scientifically-appropriate methods:

1. **Effect Sizes** - Beyond p-values (Cliff's delta, Cohen's d, fold-change)
2. **Batch Correction** - Microbiome-appropriate methods (NOT ComBat/limma)
3. **Rarefaction Analysis** - Sequencing depth validation
4. **Volcano Plots** - Differential abundance visualization

## Why These Methods Matter

### Problem 1: P-values Without Effect Sizes
- **Issue**: p < 0.05 doesn't mean biologically important
- **Solution**: Calculate effect sizes (magnitude of difference)
- **Primary method**: Cliff's delta (non-parametric, robust to outliers)

### Problem 2: Inappropriate Batch Correction
- **Issue**: ComBat/limma assume continuous, normal data (gene expression)
- **Reality**: Microbiome data is compositional, sparse, count-based
- **Solution**: Percentile normalization (non-parametric) or ConQuR (microbiome-specific)

### Problem 3: Unknown Sequencing Depth Adequacy
- **Issue**: Insufficient reads → missed rare taxa → biased diversity
- **Solution**: Rarefaction curves with plateau detection

---

## Setup

```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import logging

# Add workflow to path
sys.path.insert(0, '/usr2/people/macgregor/amplicon/workflow_16s/src')

from workflow_16s.downstream.enhanced_stats import (
    add_effect_sizes_to_stats,
    check_and_correct_batch_effects,
    validate_sequencing_depth,
    create_differential_abundance_plots,
    quick_effect_size_report
)

from workflow_16s.stats.test import mwu_bonferroni
from workflow_16s.utils.data import table_to_df

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('workflow_16s')
```

---

## Example 1: Add Effect Sizes to Statistical Results

```python
# Suppose you've run Mann-Whitney U test:
# stats_df = mwu_bonferroni(table, metadata, group_column='treatment', ...)

# Mock example (replace with your actual data)
stats_df = pd.DataFrame({
    'feature': ['Genus_Bacillus', 'Genus_Streptococcus', 'Genus_Lactobacillus'],
    'p_value': [0.001, 0.045, 0.234],
    'p_adj': [0.003, 0.067, 0.351]
})

# Add effect sizes
enhanced_stats = add_effect_sizes_to_stats(
    stats_df=stats_df,
    adata=adata,  # Your AnnData object
    group_col='treatment',
    methods=['cliffs_delta', 'cohens_d', 'log2fc']
)

print(enhanced_stats)
```

**Expected output:**
```
          feature  p_value  p_adj  cliffs_delta cliffs_delta_interpretation  cohens_d  ...
0  Genus_Bacillus    0.001  0.003         0.521                      Large     1.234  ...
1  Genus_Streptococcus 0.045  0.067     0.285                      Small     0.432  ...
2  Genus_Lactobacillus 0.234  0.351     0.102                  Negligible     0.187  ...
```

**Interpretation:**
- Genus_Bacillus: Statistically significant (p_adj=0.003) AND large effect (δ=0.521)
  → **Biologically meaningful hit!**
- Genus_Streptococcus: NOT significant (p_adj=0.067) even though small effect
  → Underpowered, need more samples
- Genus_Lactobacillus: Neither significant nor large effect → Not interesting

**Filtering for biological importance:**
```python
# Only features that are BOTH statistically significant AND medium/large effect
hits = enhanced_stats[
    (enhanced_stats['p_adj'] < 0.05) &
    (abs(enhanced_stats['cliffs_delta']) > 0.33)  # Medium threshold
]

print(f"Found {len(hits)} biologically meaningful features")
```

---

## Example 2: Batch Effect Detection and Correction

```python
# Check if batch effects are present
batch_detected, corrected_adata = check_and_correct_batch_effects(
    adata=adata,
    batch_col='sequencing_run',
    method='percentile',  # Use 'conqur' for ConQuR, 'covariate' to add as dummy variables
    output_dir=Path('results/batch_correction')
)

if batch_detected:
    print("Batch effects were detected and corrected!")
    print("Diagnostic plots saved to results/batch_correction/")
    
    # Use corrected data for downstream analysis
    adata = corrected_adata
else:
    print("No significant batch effects detected")
```

**What happens:**
1. **Detection**: PCA + ANOVA test
   - If p < 0.01 AND batch explains >10% variance → batch effects present
2. **Visualization**: PCA plots colored by batch (before/after)
3. **Correction**: Percentile normalization (quantile matching)

**Why NOT ComBat/limma?**
```
ComBat/limma assumptions:
✗ Continuous data       → Microbiome is count-based
✗ Normal distribution   → Microbiome is zero-inflated
✗ Non-compositional     → Microbiome is relative abundance

Percentile normalization:
✓ Non-parametric        → No distributional assumptions
✓ Handles zeros         → Safe for sparse data
✓ Compositional-safe    → Quantile matching preserves structure
```

**Alternative methods:**
```python
# Method 1: ConQuR (R package for microbiome)
# Requires: install.packages("devtools"); devtools::install_github("wdl2459/ConQuR")
batch_detected, corrected = check_and_correct_batch_effects(
    adata, batch_col='sequencing_run', method='conqur'
)

# Method 2: Include batch as covariate (most conservative)
batch_detected, adata_with_dummies = check_and_correct_batch_effects(
    adata, batch_col='sequencing_run', method='covariate'
)
# Now adata.obs has columns: batch_run1, batch_run2, ... (dummy coded)
# Include these in your statistical models
```

---

## Example 3: Sequencing Depth Validation

```python
# IMPORTANT: Use RAW COUNTS before normalization/rarefaction!
raw_adata = ad.read_h5ad('path/to/raw_counts.h5ad')

# Validate sequencing depth
depth_results = validate_sequencing_depth(
    adata=raw_adata,
    output_dir=Path('results/rarefaction'),
    min_adequate_pct=0.80,  # 80% of samples should reach plateau
    plot=True
)

print(f"Samples reaching plateau: {depth_results['n_adequate']}/{depth_results['n_samples']}")
print(f"Percentage adequate: {depth_results['pct_adequate']:.1f}%")
print(f"Mean plateau ratio: {depth_results['mean_plateau_ratio']:.3f}")
print(f"Suggested rarefaction depth: {depth_results['suggested_rarefaction_depth']:,} reads")
print(f"QC Pass: {depth_results['passes_qc']}")
```

**Interpretation:**
```
Samples reaching plateau: 342/370
Percentage adequate: 92.4%
Mean plateau ratio: 0.963
Suggested rarefaction depth: 5,234 reads
QC Pass: True
```

**What this means:**
- 92.4% of samples captured most of their diversity (plateau)
- Mean plateau ratio = 0.963 (final richness / asymptote)
  - 1.0 = perfect plateau
  - <0.95 = steep slope, need more reads
- Suggested rarefaction depth = 5,234 reads
  - This is the 10th percentile (standard practice)
  - Excludes low-quality samples
  - Safe depth for rarefying without losing too many samples

**If QC fails:**
```python
if not depth_results['passes_qc']:
    print("WARNING: Sequencing depth is inadequate!")
    print("Options:")
    print("  1. Rarefy to suggested depth (may lose samples)")
    print("  2. Re-sequence samples with low read counts")
    print("  3. Proceed with caution (diversity estimates may be biased)")
```

---

## Example 4: Volcano Plots and Visualizations

```python
# After adding effect sizes to your stats results:
plot_paths = create_differential_abundance_plots(
    stats_df=enhanced_stats,
    output_dir=Path('figures/volcano'),
    feature_col='feature',
    p_col='p_adj',
    fc_col='log2_fold_change',
    effect_col='cliffs_delta',
    fc_threshold=1.0,      # 2-fold change
    p_threshold=0.05,      # FDR < 0.05
    effect_threshold=0.33, # Medium effect (Cliff's delta)
    top_n=10               # Label top 10 features
)

print("Generated plots:")
for plot_type, path in plot_paths.items():
    print(f"  {plot_type}: {path}")
```

**Output:**
```
Generated plots:
  volcano: figures/volcano/volcano_plot.png
  ma: figures/volcano/ma_plot.png
  effect_volcano: figures/volcano/effect_size_volcano.png
```

**Plot types:**

1. **Classic Volcano Plot**
   - X-axis: log2 fold-change
   - Y-axis: -log10(p-value)
   - Colors: Upregulated (red), Downregulated (blue), Not significant (gray)
   - Top N features labeled

2. **MA Plot**
   - X-axis: log10 mean abundance
   - Y-axis: log2 fold-change
   - **Purpose**: Detect low-abundance artifacts
   - If significant features cluster at low abundance → potential false positives

3. **Effect Size Volcano**
   - X-axis: Cliff's delta (effect size)
   - Y-axis: -log10(p-value)
   - **Emphasizes biological importance over fold-change**
   - Cliff's delta is more robust than fold-change for microbiome data

---

## Example 5: Complete Workflow Integration

```python
# 1. LOAD DATA
adata = ad.read_h5ad('path/to/your_data.h5ad')

# 2. VALIDATE SEQUENCING DEPTH (using raw counts)
logger.info("Step 1: Validating sequencing depth...")
depth_results = validate_sequencing_depth(
    adata, 
    output_dir=Path('qc/rarefaction')
)

# 3. CHECK FOR BATCH EFFECTS
logger.info("Step 2: Checking for batch effects...")
batch_detected, adata = check_and_correct_batch_effects(
    adata,
    batch_col='sequencing_run',
    method='percentile',
    output_dir=Path('qc/batch')
)

# 4. RUN STATISTICAL TEST
logger.info("Step 3: Running statistical tests...")
from biom import load_table
table = load_table('path/to/feature_table.biom')
metadata = pd.read_csv('path/to/metadata.csv')

stats_df = mwu_bonferroni(
    table=table,
    metadata=metadata,
    group_column='treatment',
    group_column_values=['control', 'exposed']
)

# 5. ADD EFFECT SIZES
logger.info("Step 4: Calculating effect sizes...")
enhanced_stats = add_effect_sizes_to_stats(
    stats_df=stats_df,
    adata=adata,
    group_col='treatment',
    methods=['cliffs_delta', 'cohens_d', 'log2fc']
)

# 6. GENERATE PLOTS
logger.info("Step 5: Generating visualization...")
plot_paths = create_differential_abundance_plots(
    enhanced_stats,
    output_dir=Path('figures/volcano')
)

# 7. FILTER FOR BIOLOGICALLY MEANINGFUL HITS
significant_hits = enhanced_stats[
    (enhanced_stats['p_adj'] < 0.05) &
    (abs(enhanced_stats['cliffs_delta']) > 0.33)
].sort_values('cliffs_delta', key=abs, ascending=False)

print(f"\nFound {len(significant_hits)} biologically meaningful features:")
print(significant_hits[['feature', 'p_adj', 'cliffs_delta', 'cliffs_delta_interpretation']].head(10))

# 8. GENERATE SUMMARY REPORT
print(quick_effect_size_report(enhanced_stats, effect_col='cliffs_delta'))
```

---

## Comparison: Before vs After

### BEFORE (Old Pipeline)
```python
# Old way: Only p-values
stats_df = mwu_bonferroni(table, metadata, group_column='treatment')
significant = stats_df[stats_df['p_adj'] < 0.05]
print(f"{len(significant)} significant features")
```

**Problems:**
- ✗ No effect sizes → Don't know if differences are biologically meaningful
- ✗ No batch correction → Confounded by technical artifacts
- ✗ No sequencing depth validation → Biased diversity estimates
- ✗ No volcano plots → Hard to interpret results

### AFTER (New Pipeline)
```python
# New way: Effect sizes + batch correction + QC + visualization
depth_ok = validate_sequencing_depth(adata, Path('qc/rarefaction'))
batch_detected, adata = check_and_correct_batch_effects(adata, 'sequencing_run')
stats_df = mwu_bonferroni(table, metadata, group_column='treatment')
enhanced = add_effect_sizes_to_stats(stats_df, adata, 'treatment')
plots = create_differential_abundance_plots(enhanced, Path('figures'))

# Filter for BOTH statistical AND biological significance
hits = enhanced[
    (enhanced['p_adj'] < 0.05) &
    (abs(enhanced['cliffs_delta']) > 0.33)
]
```

**Benefits:**
- ✓ Effect sizes → Know which differences matter biologically
- ✓ Batch correction → Remove technical artifacts (using appropriate methods!)
- ✓ Sequencing depth QC → Confidence in diversity estimates
- ✓ Volcano plots → Publication-ready visualizations
- ✓ Interpretations → Automatic categorization (small/medium/large)

---

## Interpreting Effect Sizes

### Cliff's Delta (Primary for Microbiome)
```
|δ| < 0.147    → Negligible (don't bother)
0.147 ≤ |δ| < 0.33  → Small (interesting if consistent)
0.33 ≤ |δ| < 0.474  → Medium (likely biologically relevant)
|δ| ≥ 0.474    → Large (definitely investigate!)
```

**Why Cliff's delta?**
- Non-parametric (no assumptions about distribution)
- Robust to outliers (common in microbiome data)
- Based on Mann-Whitney U (already using non-parametric tests)
- Intuitive: Proportion of times group1 > group2

### Cohen's d (Use with caution)
```
|d| < 0.2  → Small
0.2 ≤ |d| < 0.5  → Medium
0.5 ≤ |d| < 0.8  → Large
|d| ≥ 0.8  → Very large
```

**Warning:** Cohen's d assumes normal distribution (violated by microbiome data)

### Log2 Fold-Change
```
|log2FC| < 1  → Less than 2-fold change
|log2FC| ≥ 1  → At least 2-fold change
|log2FC| ≥ 2  → At least 4-fold change
```

**Note:** Fold-change can be misleading for low-abundance taxa

---

## Configuration for Workflow

To enable these features in the main workflow, add to `config.yaml`:

```yaml
# Enhanced statistical analysis
enhanced_stats:
  enabled: true
  
  # Sequencing depth validation
  rarefaction:
    enabled: true
    min_adequate_pct: 0.80
    
  # Batch correction
  batch_correction:
    enabled: true
    batch_column: 'sequencing_run'
    method: 'percentile'  # Options: percentile, conqur, covariate
    detection_p_threshold: 0.01
    detection_r2_threshold: 0.10
  
  # Effect sizes
  effect_sizes:
    enabled: true
    methods:
      - cliffs_delta  # PRIMARY for microbiome
      - cohens_d
      - log2fc
      - hedges_g
    
  # Visualization
  plots:
    enabled: true
    volcano_plot: true
    ma_plot: true
    effect_size_volcano: true
    top_n_labels: 10
    fc_threshold: 1.0      # 2-fold
    p_threshold: 0.05      # FDR
    effect_threshold: 0.33  # Medium Cliff's delta
```

---

## Troubleshooting

### Issue: ConQuR batch correction fails
```
Error: R package 'ConQuR' not found
```

**Solution:**
```r
# In R:
install.packages("devtools")
devtools::install_github("wdl2459/ConQuR")
```

Then ensure `rpy2` is installed:
```bash
conda install -c conda-forge rpy2
```

### Issue: Rarefaction curves all fail
```
Error: No samples have sufficient reads
```

**Solution:** Your data might already be normalized. Use raw counts!
```python
# Make sure you're using counts, not relative abundance
print(adata.X.sum(axis=1))  # Should be large integers, not ~1.0
```

### Issue: Effect sizes are all NaN
```
Warning: Feature 'Genus_X' not found in adata.var_names
```

**Solution:** Feature names must match between stats_df and adata
```python
# Check feature names
print(stats_df['feature'].iloc[0])
print(adata.var_names[0])

# If they don't match, harmonize them first
```

---

## References

### Effect Sizes
- Cliff's Delta: Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. Psychological Bulletin.
- Cohen's d: Cohen, J. (1988). Statistical power analysis for the behavioral sciences.

### Batch Correction
- ConQuR: Ling et al. (2022). Batch effects removal for microbiome data via conditional quantile regression. Nature Communications.
- Why NOT ComBat: Johnson et al. (2007) designed for continuous gene expression, not compositional count data.

### Rarefaction
- McMurdie & Holmes (2014). Waste not, want not: Why rarefying microbiome data is inadmissible. PLoS Computational Biology.
  - Note: They argue AGAINST rarefying for analysis, but FOR rarefying to check depth adequacy!

---

## Next Steps

After running this enhanced workflow, you should:

1. **Report effect sizes in publications**
   - Not just p-values!
   - "Genus X was significantly enriched (p < 0.001, Cliff's δ = 0.52, large effect)"

2. **Include batch correction methods in methods section**
   - "Batch effects were detected (R² = 0.23, p < 0.001) and corrected using percentile normalization, a non-parametric method appropriate for compositional microbiome data."

3. **Report sequencing depth adequacy**
   - "Rarefaction curves indicated that 92% of samples reached a plateau (≥95% of asymptote), suggesting adequate sequencing depth."

4. **Use volcano plots in figures**
   - Much more informative than bar charts of individual taxa!

---

## Summary

The enhanced workflow adds:
- ✅ **Effect sizes** → Know which differences matter
- ✅ **Batch correction** → Remove artifacts (appropriately!)
- ✅ **Sequencing depth QC** → Validate data quality
- ✅ **Volcano plots** → Publication-ready visualization

All using **scientifically-appropriate methods for microbiome data**, not gene expression methods!
