# Advanced Analysis Guide for workflow_16s

This guide covers the new advanced analysis capabilities added to the workflow_16s pipeline for publication-quality microbiome analysis.

## Table of Contents

1. [Compositional Data Analysis](#compositional-data-analysis)
2. [Batch Effect Detection & Correction](#batch-effect-detection--correction)
3. [Decontamination](#decontamination)
4. [Phylogenetic Diversity](#phylogenetic-diversity)
5. [Differential Abundance Testing](#differential-abundance-testing)
6. [Compositional Networks](#compositional-networks)
7. [Longitudinal Analysis](#longitudinal-analysis)
8. [Power Analysis](#power-analysis)

---

## Compositional Data Analysis

### Why It Matters

16S amplicon sequencing produces compositional data (relative abundances that sum to 1 or 100%). Standard statistical methods can produce spurious correlations when applied to compositional data.

### Zero Handling

The pipeline implements proper zero-handling before log-ratio transformations:

```python
from workflow_16s.downstream.compositional import handle_zeros_multiplicative

# Replace zeros with multiplicative replacement (Martin-Fernández 2003)
counts_no_zeros = handle_zeros_multiplicative(counts, delta=0.65)
```

**Configuration:**
```yaml
compositional:
  enabled: True
  zero_handling: "multiplicative"
  delta_multiplicative: 0.65
```

### Log-Ratio Transformations

**Centered Log-Ratio (CLR):**
```python
from workflow_16s.downstream.compositional import clr_transform

clr_data = clr_transform(adata)
```

**Isometric Log-Ratio (ILR):**
```python
from workflow_16s.downstream.compositional import ilr_transform

ilr_data = ilr_transform(adata)
```

**Phylogenetic ILR (PhILR):**
```python
from workflow_16s.downstream.compositional import philr_transform

# Requires phylogenetic tree
philr_data = philr_transform(adata, tree)
```

### Diagnostics

Check for zero patterns in your data:

```python
from workflow_16s.downstream.compositional import diagnose_zeros

diagnosis = diagnose_zeros(adata)
print(f"Samples with >50% zeros: {diagnosis['high_zero_samples']}")
print(f"Features with >90% zeros: {diagnosis['high_zero_features']}")
```

**References:**
- Martin-Fernández et al. (2003). *Mathematical Geology*
- Gloor et al. (2017). *Frontiers in Microbiology*
- Quinn et al. (2019). *mSystems*

---

## Batch Effect Detection & Correction

### Why It Matters

Batch effects are technical variations introduced during sample processing, sequencing, or data handling. They can confound biological signals and lead to false discoveries.

### Detection

**Automatic batch effect diagnostics:**

```python
from workflow_16s.downstream.batch_effects import detect_batch_effects

results = detect_batch_effects(
    adata,
    batch_col='batch_original',
    group_col='treatment'
)

print(f"PERMANOVA R²: {results['permanova_r2']:.3f}")
print(f"p-value: {results['permanova_p']}")
```

**Configuration:**
```yaml
batch_effects:
  enabled: True
  batch_column: "batch_original"
  detection:
    enabled: True
    methods: ["permanova", "pca", "silhouette"]
    plot_pca: True
```

### Correction

**ConQuR (recommended for microbiome data):**

```python
from workflow_16s.downstream.batch_effects import apply_conqur_correction

corrected_adata = apply_conqur_correction(
    adata,
    batch_col='batch_original',
    covariate_cols=['treatment']
)
```

**ComBat (parametric correction):**

```python
from workflow_16s.downstream.batch_effects import apply_combat_correction

corrected_adata = apply_combat_correction(
    adata,
    batch_col='batch_original',
    covariate_cols=['treatment']
)
```

**Full workflow:**

```python
from workflow_16s.downstream.batch_effects import batch_effect_workflow

results = batch_effect_workflow(
    adata,
    batch_col='batch_original',
    group_col='treatment',
    correction_method='conqur'
)
```

**Configuration:**
```yaml
batch_effects:
  correction:
    enabled: True
    method: "conqur"  # or "combat"
```

**When to use ConQuR vs ComBat:**
- **ConQuR**: Recommended for microbiome data (handles compositionality, zero-inflation)
- **ComBat**: General-purpose, faster, good for high-dimensional data

**References:**
- Ling et al. (2022). *Nature Communications* - ConQuR
- Johnson et al. (2007). *Biostatistics* - ComBat

---

## Decontamination

### Why It Matters

Contaminants from reagents, lab environment, or cross-contamination can dominate low-biomass samples and skew results.

### Requirements

- **Negative control samples** (extraction blanks, PCR blanks)
- Metadata column indicating control samples
- (Optional) DNA concentration measurements

### Frequency Method

Identifies contaminants based on inverse correlation with DNA concentration:

```python
from workflow_16s.downstream.decontamination import identify_contaminants

contaminants = identify_contaminants(
    adata,
    method='frequency',
    neg_control_col='is_negative_control',
    concentration_col='dna_concentration',
    threshold=0.1
)
```

### Prevalence Method

Identifies contaminants more prevalent in negative controls:

```python
contaminants = identify_contaminants(
    adata,
    method='prevalence',
    neg_control_col='is_negative_control',
    threshold=0.5
)
```

### Combined Method

Uses both frequency and prevalence evidence:

```python
contaminants = identify_contaminants(
    adata,
    method='combined',
    neg_control_col='is_negative_control',
    concentration_col='dna_concentration',
    threshold=0.1
)
```

### Remove Contaminants

```python
from workflow_16s.downstream.decontamination import remove_contaminants

clean_adata = remove_contaminants(
    adata,
    contaminant_features=['Taxon1', 'Taxon2', ...]
)
```

**Configuration:**
```yaml
decontamination:
  enabled: True
  neg_control_column: "is_negative_control"
  method: "combined"
  threshold: 0.1
  remove_contaminants: True
```

**References:**
- Davis et al. (2018). *Microbiome* - decontam R package

---

## Phylogenetic Diversity

### Why It Matters

Phylogenetic diversity metrics incorporate evolutionary relationships, providing more statistical power than taxonomy-based metrics when detecting community differences.

### Faith's Phylogenetic Diversity

Measures total phylogenetic branch length:

```python
from workflow_16s.downstream.phylogenetic_diversity import calculate_faith_pd

faith_pd = calculate_faith_pd(adata, tree)
```

### UniFrac Distances

**Unweighted UniFrac** (presence/absence):
```python
from workflow_16s.downstream.phylogenetic_diversity import calculate_unifrac

unifrac_dist = calculate_unifrac(
    adata,
    tree,
    weighted=False
)
```

**Weighted UniFrac** (abundance-weighted):
```python
unifrac_dist = calculate_unifrac(
    adata,
    tree,
    weighted=True
)
```

### Tree Building

If you don't have a phylogenetic tree:

**FastTree (fast de novo):**
```python
from workflow_16s.downstream.phylogenetic_diversity import build_tree_fasttree

tree = build_tree_fasttree(sequences, threads=16)
```

**SEPP (fragment insertion):**
```python
from workflow_16s.downstream.phylogenetic_diversity import insert_sequences_sepp

tree = insert_sequences_sepp(
    sequences,
    reference_db='greengenes',
    threads=16
)
```

**Full workflow:**
```python
from workflow_16s.downstream.phylogenetic_diversity import phylogenetic_diversity_workflow

results = phylogenetic_diversity_workflow(
    adata,
    tree_path=None,  # Will build tree
    tree_method='fasttree'
)
```

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: True
  tree_path: null  # or path to existing tree
  tree_method: "fasttree"  # or "sepp"
  metrics:
    faith_pd: True
    unifrac_unweighted: True
    unifrac_weighted: True
```

**References:**
- Faith (1992). *Biological Conservation* - Faith's PD
- Lozupone & Knight (2005). *Applied and Environmental Microbiology* - UniFrac

---

## Differential Abundance Testing

### Why It Matters

Different DA methods have different assumptions and sensitivities. Using multiple methods and consensus analysis increases robustness.

### Available Methods

1. **DESeq2** - Negative binomial model (from RNA-seq)
2. **corncob** - Beta-binomial model (microbiome-specific)
3. **LinDA** - Linear mixed model with compositional adjustment
4. **ALDEx2** - Compositionally-aware with Monte Carlo sampling
5. **Wilcoxon** - Non-parametric (distribution-free)

### Single Method

```python
from workflow_16s.downstream.differential_abundance import run_deseq2

da_results = run_deseq2(
    adata,
    group_col='treatment',
    control_group='control',
    test_group='treated'
)
```

### Consensus Analysis

Run multiple methods and find agreement:

```python
from workflow_16s.downstream.differential_abundance import (
    run_deseq2, run_aldex2, run_wilcoxon, consensus_da_features
)

# Run methods
deseq2_results = run_deseq2(adata, ...)
aldex2_results = run_aldex2(adata, ...)
wilcoxon_results = run_wilcoxon(adata, ...)

# Find consensus
consensus = consensus_da_features(
    [deseq2_results, aldex2_results, wilcoxon_results],
    method_names=['DESeq2', 'ALDEx2', 'Wilcoxon'],
    min_agreement=2  # At least 2 methods must agree
)

print(f"Consensus features: {len(consensus['consensus_features'])}")
```

### Method Comparison

Compare results across methods:

```python
from workflow_16s.downstream.differential_abundance import compare_da_methods

comparison = compare_da_methods(
    [deseq2_results, aldex2_results, wilcoxon_results],
    method_names=['DESeq2', 'ALDEx2', 'Wilcoxon']
)

# Generates upset plots, correlation matrices, etc.
```

**Configuration:**
```yaml
differential_abundance:
  enabled: True
  group_column: "treatment"
  methods: ["deseq2", "aldex2", "wilcoxon"]
  fdr_threshold: 0.05
  log_fc_threshold: 1.0
  consensus:
    enabled: True
    min_agreement: 2
```

**Method Selection Guide:**

| Method | Best For | Assumptions |
|--------|----------|-------------|
| DESeq2 | Count data, large samples | Negative binomial |
| corncob | Microbiome data, zeros | Beta-binomial |
| LinDA | Mixed designs, covariates | Linear model |
| ALDEx2 | Compositional data | Monte Carlo sampling |
| Wilcoxon | Small samples, no assumptions | Non-parametric |

**References:**
- Love et al. (2014). *Genome Biology* - DESeq2
- Martin et al. (2020). *Annals of Applied Statistics* - corncob
- Zhou et al. (2022). *Genome Biology* - LinDA
- Fernandes et al. (2013). *Microbiome* - ALDEx2

---

## Compositional Networks

### Why It Matters

Standard correlation methods fail for compositional data due to spurious correlations. These methods account for compositionality.

### SPIEC-EASI

Sparse inverse covariance estimation:

```python
from workflow_16s.downstream.compositional_networks import run_spiec_easi

network = run_spiec_easi(
    adata,
    method='glasso',  # or 'mb'
    nlambda=20
)
```

### SparCC

Compositionally-robust correlation:

```python
from workflow_16s.downstream.compositional_networks import run_sparcc

network = run_sparcc(
    adata,
    iterations=20,
    threshold=0.1
)
```

### Proportionality

Log-ratio based correlation:

```python
from workflow_16s.downstream.compositional_networks import run_proportionality

network = run_proportionality(
    adata,
    metric='rho'  # or 'phi'
)
```

### Network Comparison

```python
from workflow_16s.downstream.compositional_networks import compare_network_methods

comparison = compare_network_methods(
    adata,
    methods=['spiec_easi', 'sparcc', 'proportionality']
)
```

### Visualization

```python
from workflow_16s.downstream.compositional_networks import plot_network

fig = plot_network(
    network,
    layout='spring',
    node_size_by='abundance'
)
```

**Configuration:**
```yaml
compositional_networks:
  enabled: True
  methods: ["sparcc", "spiec_easi"]
  min_abundance: 0.001
  min_prevalence: 0.1
  plot_networks: True
```

**Method Selection:**
- **SPIEC-EASI**: Most rigorous, computationally intensive
- **SparCC**: Fast, handles compositionality well
- **Proportionality**: Interpretable, good for exploratory analysis

**References:**
- Kurtz et al. (2015). *PLoS Computational Biology* - SPIEC-EASI
- Friedman & Alm (2012). *PLoS Computational Biology* - SparCC
- Lovell et al. (2015). *PLOS Computational Biology* - Proportionality

---

## Longitudinal Analysis

### Why It Matters

Standard methods assume independent samples. Longitudinal data has repeated measurements requiring specialized methods.

### Requirements

- Subject IDs (to link repeated measurements)
- Time point information
- At least 3 time points recommended

### ZIBR

Zero-Inflated Beta Regression for compositional time series:

```python
from workflow_16s.downstream.longitudinal import run_zibr

zibr_results = run_zibr(
    adata,
    subject_col='subject_id',
    time_col='time_point',
    group_col='treatment',
    formula_1='time_point + treatment',  # Presence/absence
    formula_2='time_point * treatment'   # Abundance
)
```

### MaAsLin 2

Mixed-effects models with random effects:

```python
from workflow_16s.downstream.longitudinal import run_maaslin2_longitudinal

maaslin_results = run_maaslin2_longitudinal(
    adata,
    subject_col='subject_id',
    fixed_effects=['time_point', 'treatment'],
    random_effects=['subject_id']
)
```

### Trajectory Clustering

Group subjects by similar temporal patterns:

```python
from workflow_16s.downstream.longitudinal import trajectory_clustering

clusters = trajectory_clustering(
    adata,
    subject_col='subject_id',
    time_col='time_point',
    n_clusters=3
)
```

### Temporal Stability

Measure microbiome stability over time:

```python
from workflow_16s.downstream.longitudinal import calculate_temporal_stability

stability = calculate_temporal_stability(
    adata,
    subject_col='subject_id',
    time_col='time_point',
    metric='braycurtis'
)
```

**Configuration:**
```yaml
longitudinal:
  enabled: True
  subject_column: "subject_id"
  time_column: "time_point"
  methods: ["maaslin2", "zibr"]
  temporal_stability:
    enabled: True
    metric: "braycurtis"
  plot_trajectories: True
```

**References:**
- Chen & Li (2016). *Genome Biology* - ZIBR
- Mallick et al. (2021). *eLife* - MaAsLin 2

---

## Power Analysis

### Why It Matters

Underpowered studies waste resources and can't detect real effects. Power analysis helps design adequately powered experiments.

### PERMANOVA Power

Estimate sample size for beta diversity analysis:

```python
from workflow_16s.downstream.power_analysis import estimate_permanova_power

power_results = estimate_permanova_power(
    pilot_adata,  # Your pilot data
    group_col='treatment',
    target_power=0.8,
    alpha=0.05
)

print(f"Minimum sample size: {power_results['min_sample_size']}")
```

### Differential Abundance Power

Estimate sample size for DA testing:

```python
from workflow_16s.downstream.power_analysis import estimate_da_power

da_power = estimate_da_power(
    mean_effect_size=0.5,  # Expected log2 fold change
    within_group_variance=1.0,
    target_power=0.8
)

print(f"Minimum per group: {da_power['min_sample_size_per_group']}")
```

### Comprehensive Pilot Analysis

Analyze pilot data for all metrics:

```python
from workflow_16s.downstream.power_analysis import pilot_data_power_analysis

full_analysis = pilot_data_power_analysis(
    pilot_adata,
    group_col='treatment',
    target_power=0.8
)
```

### Power Curves

Visualize power across sample sizes:

```python
from workflow_16s.downstream.power_analysis import plot_power_curves

fig = plot_power_curves(power_results)
fig.write_html('power_curves.html')
```

**Configuration:**
```yaml
power_analysis:
  enabled: True
  target_power: 0.8
  alpha: 0.05
  effect_sizes: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
  sample_sizes: [10, 20, 30, 50, 75, 100, 150, 200]
  plot_power_curves: True
```

**References:**
- Kelly et al. (2015). *Bioinformatics* - PERMANOVA power
- La Rosa et al. (2012). *PLoS ONE* - Microbiome power analysis

---

## Best Practices

### 1. Compositional Data

- **Always use log-ratio transformations** for Euclidean-based methods (PCA, correlation, etc.)
- **Handle zeros properly** before transformations
- Use CLR for most analyses, ILR/PhILR when interpretability matters

### 2. Batch Effects

- **Check for batch effects** before analysis (especially multi-center studies)
- Use ConQuR for microbiome data (handles compositionality)
- **Validate correction** - ensure biological signal retained

### 3. Decontamination

- **Always include negative controls** in experimental design
- Run decontamination **before** other analyses
- **Document removed contaminants** for methods section

### 4. Differential Abundance

- **Use multiple methods** - no single method is perfect
- Require **consensus** across methods for high-confidence calls
- **Report effect sizes**, not just p-values

### 5. Networks

- **Pre-filter** low abundance/prevalence features
- Use SPIEC-EASI for publication-quality networks
- **Validate** networks with independent data when possible

### 6. Longitudinal Studies

- **Account for repeated measures** in statistical models
- Check temporal structure before analysis
- Consider subject-level heterogeneity

### 7. Power Analysis

- **Run on pilot data** before full study
- Add 10-20% to sample size estimates for QC losses
- **Report power** in publications

---

## Example Workflow

Complete analysis incorporating all features:

```python
import anndata as ad
from workflow_16s.downstream import *

# 1. Load data
adata = ad.read_h5ad('feature_table.h5ad')

# 2. Decontamination (if negative controls available)
contaminants = identify_contaminants(
    adata,
    method='combined',
    neg_control_col='is_negative_control'
)
adata = remove_contaminants(adata, contaminants)

# 3. Compositional transformation
from workflow_16s.downstream.compositional import clr_transform
adata_clr = clr_transform(adata)

# 4. Batch effect correction
from workflow_16s.downstream.batch_effects import batch_effect_workflow
corrected_results = batch_effect_workflow(
    adata_clr,
    batch_col='batch',
    group_col='treatment',
    correction_method='conqur'
)
adata_corrected = corrected_results['corrected_adata']

# 5. Phylogenetic diversity
from workflow_16s.downstream.phylogenetic_diversity import phylogenetic_diversity_workflow
phylo_results = phylogenetic_diversity_workflow(
    adata_corrected,
    tree_method='fasttree'
)

# 6. Differential abundance (consensus)
from workflow_16s.downstream.differential_abundance import *
deseq2_res = run_deseq2(adata_corrected, ...)
aldex2_res = run_aldex2(adata_corrected, ...)
wilcoxon_res = run_wilcoxon(adata_corrected, ...)

consensus = consensus_da_features(
    [deseq2_res, aldex2_res, wilcoxon_res],
    method_names=['DESeq2', 'ALDEx2', 'Wilcoxon'],
    min_agreement=2
)

# 7. Compositional network
from workflow_16s.downstream.compositional_networks import run_sparcc
network = run_sparcc(adata_corrected)

# 8. Generate report
print("Analysis complete!")
print(f"Consensus DA features: {len(consensus['consensus_features'])}")
```

---

## Troubleshooting

### Import Errors

If you get R-related import errors:
```bash
conda install -c conda-forge r-base r-deseq2 r-aldex2
```

### Memory Issues

For large datasets:
- Filter low-abundance features before network analysis
- Use `sparse=True` where available
- Reduce Monte Carlo samples in ALDEx2

### Convergence Issues

If models fail to converge:
- Check for low-abundance features (filter them out)
- Reduce model complexity
- Check for collinear covariates

---

## Getting Help

- **Documentation**: See individual module docstrings
- **Issues**: GitHub issues for bug reports
- **Questions**: Contact maintainer

---

## Citation

If you use these methods, please cite the original papers listed in each section's references.
