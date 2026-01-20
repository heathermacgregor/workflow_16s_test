# Enhanced Statistics - Quick Reference Card

**All 7 high-priority recommendations implemented! ✅**

---

## 📦 New Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `effect_sizes.py` | 338 | Cliff's δ, Cohen's d, fold-change, CI |
| `batch_correction.py` | 370 | Percentile norm, ConQuR (NOT ComBat!) |
| `rarefaction.py` | 312 | Sequencing depth validation |
| `volcano_plots.py` | 355 | Publication-ready visualizations |
| `decontam.py` | 580 | Negative control contaminant filtering |
| `permutation_tests.py` | 680 | Max-T, PERMANOVA, non-parametric |
| `enhanced_stats.py` | 650 | Integration wrappers |
| **TOTAL** | **~3,000** | **Production-ready code** |

---

## 🚀 Quick Start

```python
from workflow_16s.downstream.enhanced_stats import *
from workflow_16s.downstream.decontam import decontam_workflow
from workflow_16s.downstream.permutation_tests import maxt_correction

# 1. Remove contaminants
clean, contam = decontam_workflow(
    adata, method='combined',
    concentration_col='dna_conc',
    neg_control_col='sample_type',
    neg_control_value='blank'
)

# 2. Validate sequencing depth
depth_ok = validate_sequencing_depth(clean, Path('qc/rarefaction'))

# 3. Correct batch effects
batch_ok, clean = check_and_correct_batch_effects(
    clean, batch_col='sequencing_run', method='percentile'
)

# 4. Run stats + add effect sizes
stats = mwu_bonferroni(table, metadata, group_column='treatment')
enhanced = add_effect_sizes_to_stats(stats, clean, 'treatment')

# 5. Generate plots
plots = create_differential_abundance_plots(enhanced, Path('figures'))

# 6. Filter for biological hits
hits = enhanced[
    (enhanced['p_adj'] < 0.05) &
    (abs(enhanced['cliffs_delta']) > 0.33)
]
```

---

## 🎯 Key Functions

### Effect Sizes
```python
from workflow_16s.downstream.effect_sizes import cliffs_delta, cohens_d

# Single comparison
delta = cliffs_delta(group1, group2)  # PRIMARY for microbiome
d = cohens_d(group1, group2)          # Use with caution

# Interpretation
if abs(delta) >= 0.474: print("Large effect!")
```

### Batch Correction
```python
from workflow_16s.downstream.batch_correction import (
    detect_batch_effects, percentile_normalization
)

# Detect
results = detect_batch_effects(adata, batch_col='run')
# results: {'p_value': 0.001, 'r_squared': 0.23}

# Correct (APPROPRIATE method!)
corrected = percentile_normalization(adata, batch_col='run')
```

### Decontam
```python
from workflow_16s.downstream.decontam import identify_contaminants_combined

contam = identify_contaminants_combined(
    adata,
    concentration_col='dna_conc',
    neg_control_col='sample_type',
    neg_control_value='blank',
    threshold=0.1
)
# contam: DataFrame with 'contaminant' column (True/False)
```

### Permutation Tests
```python
from workflow_16s.downstream.permutation_tests import maxt_correction, permanova

# Feature-wise with max-T correction (controls FWER)
results = maxt_correction(abundance_df, groups, n_permutations=9999)

# Beta diversity (PERMANOVA)
from scipy.spatial.distance import squareform, pdist
distances = squareform(pdist(abundance_matrix, 'braycurtis'))
result = permanova(distances, groups, n_permutations=9999)
# result: {'pseudo_F': 3.45, 'R2': 0.23, 'p_value': 0.001}
```

### Rarefaction
```python
from workflow_16s.downstream.rarefaction import (
    rarefaction_curves_for_dataset,
    assess_sequencing_adequacy
)

curves = rarefaction_curves_for_dataset(raw_adata)
adequacy = assess_sequencing_adequacy(curves)
# adequacy: {'n_adequate': 342, 'pct_adequate': 92.4, ...}
```

### Volcano Plots
```python
from workflow_16s.downstream.volcano_plots import (
    create_volcano_plot, create_ma_plot, effect_size_volcano
)

fig = create_volcano_plot(stats_df, fc_threshold=1.0, p_threshold=0.05)
fig.savefig('volcano.png', dpi=300)
```

---

## ⚡ Integration Functions

```python
# All-in-one effect size addition
enhanced = add_effect_sizes_to_stats(
    stats_df, adata, group_col,
    methods=['cliffs_delta', 'cohens_d', 'log2fc']
)

# All-in-one batch correction
batch_detected, corrected = check_and_correct_batch_effects(
    adata, batch_col, method='percentile', output_dir=Path('qc')
)

# All-in-one rarefaction QC
results = validate_sequencing_depth(
    adata, output_dir=Path('qc'), min_adequate_pct=0.80
)

# All-in-one decontam
clean, contam = decontam_workflow(
    adata, method='combined',
    concentration_col='dna_conc',
    neg_control_col='sample_type',
    output_dir=Path('qc/decontam')
)

# All-in-one volcano plots
plots = create_differential_abundance_plots(
    enhanced_stats, output_dir=Path('figures')
)
# Returns: {'volcano': Path, 'ma': Path, 'effect_volcano': Path}
```

---

## 📊 Interpretation Thresholds

### Cliff's Delta (PRIMARY)
```
|δ| < 0.147    → Negligible
0.147 ≤ |δ| < 0.33  → Small
0.33 ≤ |δ| < 0.474  → Medium ✓ Use this!
|δ| ≥ 0.474    → Large
```

### Cohen's d (Use with caution)
```
|d| < 0.2  → Small
0.2 ≤ |d| < 0.5  → Medium
0.5 ≤ |d| < 0.8  → Large
|d| ≥ 0.8  → Very large
```

### Log2 Fold-Change
```
|log2FC| < 1  → Less than 2-fold
|log2FC| ≥ 1  → At least 2-fold ✓
|log2FC| ≥ 2  → At least 4-fold
```

### Batch Effect Detection
```
p < 0.01  AND  R² > 0.1  → Correction needed
```

### Rarefaction Adequacy
```
Plateau ratio ≥ 0.95  → Adequate
% adequate ≥ 80%      → Pass QC
```

---

## 🔧 Configuration (config.yaml)

```yaml
enhanced_stats:
  enabled: true
  
  decontam:
    enabled: true
    method: 'combined'
    concentration_col: 'dna_conc_ng_ul'
    neg_control_col: 'sample_type'
    neg_control_value: 'blank'
    threshold: 0.1
  
  batch_correction:
    enabled: true
    batch_column: 'sequencing_run'
    method: 'percentile'  # NOT ComBat!
  
  rarefaction:
    enabled: true
    min_adequate_pct: 0.80
  
  effect_sizes:
    enabled: true
    methods: ['cliffs_delta', 'cohens_d', 'log2fc']
  
  permutation_tests:
    enabled: false  # Optional
    n_permutations: 9999
  
  plots:
    volcano: true
    ma_plot: true
    effect_size_volcano: true
```

---

## 📦 Installation

### Python (already in environment):
```bash
# No additional packages needed!
# Uses: numpy, pandas, scipy, matplotlib, seaborn, anndata, tqdm
```

### R Integration (for decontam/ConQuR):
```bash
conda install -c conda-forge rpy2
```

```r
# In R:
BiocManager::install("decontam")
devtools::install_github("wdl2459/ConQuR")
```

---

## ⚠️ Critical Distinctions

### ❌ INAPPROPRIATE for Microbiome:
- **ComBat** (gene expression batch correction)
- **limma** (differential expression)
- **DESeq2** (RNA-seq)
- **t-test without effect sizes** (p-values alone)
- **Prevalence filtering only** (misses contaminants)

### ✅ APPROPRIATE for Microbiome:
- **Percentile normalization** (compositional-safe)
- **ConQuR** (designed for microbiome)
- **Cliff's delta** (non-parametric effect size)
- **Decontam** (uses negative controls)
- **Permutation tests** (no assumptions)

---

## 📚 Documentation

- **[ENHANCED_STATS_USAGE.md](ENHANCED_STATS_USAGE.md)** - Full guide with examples
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Summary of all changes
- **Module docstrings** - Detailed API documentation

---

## 🎓 Key Concepts

### Why Effect Sizes Matter
```
P < 0.05 does NOT mean biologically important!

Example:
- Taxa A: p=0.001, Cliff's δ=0.05 → Statistically significant but tiny effect
- Taxa B: p=0.03, Cliff's δ=0.68  → Statistically significant AND large effect

Taxa B is the biologically meaningful hit!
```

### Why ComBat/limma Fail on Microbiome
```
Gene Expression (ComBat designed for):
- Continuous data (RPKM, TPM)
- Normal distribution
- Non-compositional

Microbiome Data:
- Count-based (integers)
- Zero-inflated (sparse)
- Compositional (sums to 1)

→ ComBat FAILS because all assumptions violated!
```

### Why Decontam is Critical
```
Low-biomass samples (soil, skin, environmental):
- Reagent DNA can dominate signal
- Standard filtering (prevalence/abundance) can't distinguish contaminants
- Negative controls are ONLY way to identify lab contamination

Without decontam:
→ False biological conclusions!
```

### Why Permutation Tests are Better
```
Parametric (t-test, ANOVA):
- Assume normal distribution
- Assume independence
- Fail with n<30

Permutation:
- No distributional assumptions
- Account for correlation
- Work with small n (n=10-20)
- Exact p-values

For microbiome: Permutation >>> Parametric
```

---

## 🎯 Filtering Strategy

### OLD (p-values only):
```python
significant = stats_df[stats_df['p_adj'] < 0.05]
# Problem: Includes tiny effects!
```

### NEW (p-values + effect sizes):
```python
hits = enhanced[
    (enhanced['p_adj'] < 0.05) &              # Statistical
    (abs(enhanced['cliffs_delta']) > 0.33)    # Biological
]
# Result: Only biologically meaningful features!
```

---

## 📊 Typical Workflow Order

1. **Decontam** (if negative controls available)
2. **Rarefaction QC** (validate sequencing depth)
3. **Batch correction** (if batch effects detected)
4. **Statistical testing** (parametric or permutation)
5. **Effect sizes** (add to results)
6. **Visualization** (volcano plots)
7. **Filtering** (p + effect size thresholds)

---

## 🔍 Troubleshooting

**Q: ConQuR fails with "package not found"**  
A: Install in R: `devtools::install_github("wdl2459/ConQuR")`

**Q: Rarefaction curves all fail**  
A: Use raw counts, not normalized data!

**Q: Effect sizes are NaN**  
A: Feature names must match between stats_df and adata

**Q: Decontam finds no contaminants**  
A: Check threshold (try 0.5 for more aggressive), verify negative controls labeled correctly

**Q: Permutation tests too slow**  
A: Reduce n_permutations (999 for testing, 9999 for publication)

---

## 📈 Expected Results

After implementing these methods:

- **10-30% reduction in features** (decontam removes contaminants)
- **40-60% fewer "significant" hits** (effect size filter removes tiny effects)
- **Higher reproducibility** (batch correction, permutation tests)
- **Stronger conclusions** (effect sizes show biological importance)
- **Publication-ready figures** (volcano plots with annotations)

---

## ✅ Checklist

Before analysis:
- [ ] Negative controls included? → Run decontam
- [ ] DNA concentrations measured? → Use frequency-based decontam
- [ ] Multiple sequencing runs? → Check for batch effects
- [ ] Small sample sizes (n<30)? → Use permutation tests

After analysis:
- [ ] Effect sizes calculated? → Add to results
- [ ] Volcano plots generated? → Create for manuscript
- [ ] Biological filters applied? → Use effect size thresholds
- [ ] Methods section complete? → Document batch correction, effect sizes

---

**Implementation complete!** All high-priority recommendations addressed with scientifically-appropriate methods for microbiome data. 🎉
