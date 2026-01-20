# Feature Toggle & Configuration Guide

**Complete reference for enabling/disabling all analysis modules in workflow_16s**

Last updated: 2026-01-11

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Fully Implemented Modules](#fully-implemented-modules)
3. [Integration Stubs for R Packages](#integration-stubs-for-r-packages)
4. [Configuration Examples](#configuration-examples)
5. [Troubleshooting](#troubleshooting)

---

## Quick Reference

All toggles are in `workflow_16s/config/config.yaml`. Set `enabled: true` to activate.

| Module | Config Key | Status | Dependencies |
|--------|-----------|--------|--------------|
| **Metadata Profiling** | *(always runs)* | ✅ **Implemented** | None |
| **Power Analysis** | `power_analysis.enabled` | ✅ **Implemented** | None |
| **Rarefaction Curves** | `rarefaction.enabled` | ✅ **Implemented** | None |
| **Phylogenetic Diversity** | `phylogeny.enabled` | ✅ **Implemented** | Phylogenetic tree file |
| **Differential Abundance** | `differential_abundance.enabled` | ✅ **Implemented** | None (Python + R) |
| **Compositional Networks** | `networks.enabled` | ✅ **Implemented** | None |
| **Longitudinal Analysis** | `longitudinal.enabled` | ✅ **Implemented** | `time_column`, `subject_column` |
| **Batch Effect Detection** | `batch_effects.enabled` | ✅ **Implemented** | None |
| **Decontamination** | `decontamination.enabled` | ✅ **Implemented** | Negative controls OR DNA conc |
| **Functional Profiling** | `functional.enabled` | 🔄 **R Stub** | PICRUSt2 (external) |
| **Source Tracking** | `source_tracking.enabled` | 🔄 **R Stub** | FEAST (R package) |

---

## Fully Implemented Modules

### 1. Power Analysis ✅

**Purpose**: Estimate statistical power or required sample sizes for your study design.

**When to use**:
- Before starting expensive analyses
- Planning future studies
- Justifying sample sizes in grant applications

**Configuration**:
```yaml
power_analysis:
  enabled: true
  target_power: 0.8          # Target power (default: 80%)
  alpha: 0.05                # Significance threshold
  run_before_da: true        # Check power before differential abundance
  run_before_permanova: true # Check power before PERMANOVA
  min_power_threshold: 0.5   # Minimum acceptable power to proceed
  output_dir: '04_analysis/power_analysis'
```

**Outputs**:
- `power_alpha_diversity.csv`: Power estimates for alpha diversity comparisons
- `power_beta_diversity.csv`: Power estimates for PERMANOVA tests
- `power_curves.html`: Interactive power plots
- Warnings if current study is underpowered

**Implementation**: `workflow_16s/src/workflow_16s/downstream/power_analysis.py` (769 lines)

**Example output**:
```
POWER ANALYSIS
================================================================================

1. ALPHA DIVERSITY POWER
--------------------------------------------------------------------------------
  Effect: estimated (Cohen's d = 0.42)
    Current n per group: 52
    Current power: 0.823
    Required n per group for power=0.8: 45
    ✓ Adequate power achieved

2. BETA DIVERSITY (PERMANOVA) POWER
--------------------------------------------------------------------------------
  Effect: estimated (Cohen's f = 0.21)
    Current n per group: 52
    Current power: 0.789
    Required n per group for power=0.8: 55
    ⚠️  Underpowered (need 3 more per group)
```

---

### 2. Rarefaction Curves ✅

**Purpose**: Assess whether sequencing depth was sufficient to capture community diversity.

**When to use**:
- Always (quality control)
- Interpreting alpha diversity results
- Planning resequencing efforts

**Configuration**:
```yaml
rarefaction:
  enabled: true
  metric: 'observed_features'   # or 'shannon', 'simpson', 'pielou_evenness'
  n_depths: 20                  # Number of rarefaction depths to test
  group_column: null            # Optional: color by metadata (e.g., 'treatment')
  plot_individual_samples: false # Plot all samples individually (can be slow)
  plot_by_group: true           # Plot average curves per group
  output_dir: '03_processed_data/rarefaction'
```

**Outputs**:
- `rarefaction_data.csv`: Raw rarefaction data (depth, richness per sample)
- `rarefaction_curves.html`: Interactive plot
- `rarefaction_by_group.html`: Per-group average curves (if `plot_by_group: true`)

**Implementation**: `workflow_16s/src/workflow_16s/downstream/rarefaction.py` (315 lines)

**Interpretation**:
- **Plateau**: Good! Sequencing depth is adequate
- **Still climbing**: Undersampled, more sequencing recommended
- **High variance**: Check for batch effects or contamination

---

### 3. Phylogenetic Diversity ✅

**Purpose**: Calculate diversity metrics that account for evolutionary relationships.

**When to use**:
- When you have a phylogenetic tree
- Studying environmental samples (trees more informative than taxonomy alone)
- Comparing phylogenetically distant communities

**Configuration**:
```yaml
phylogeny:
  enabled: true
  # Tree handling
  tree_path: 'resources/silva_tree.nwk'  # Path to existing tree
  tree_method: 'sepp'                    # 'sepp', 'fasttree', or 'none'
  missing_tree_strategy: 'auto'          # How to handle missing trees
  
  # Metrics to calculate
  alpha_diversity:
    faiths_pd: true      # Faith's Phylogenetic Diversity
    phylo_entropy: false # Phylogenetic entropy (experimental)
  
  beta_diversity:
    weighted_unifrac: true    # Weighted UniFrac (abundance-weighted)
    unweighted_unifrac: false # Unweighted UniFrac (presence/absence)
  
  # Output
  output_dir: '04_analysis/phylogenetic_diversity'
```

**Missing Tree Strategies**:
- `auto`: Automatically select best strategy based on data
- `graceful_degradation`: Skip phylogenetic metrics (safe)
- `tree_merging`: Merge per-dataset trees
- `denovo_tree_building`: Build tree from representative sequences
- `partial_analysis`: Analyze only features with tree coverage
- `subset_tree_extraction`: Extract subtree from reference

**Outputs**:
- `faith_pd.csv`: Faith's PD values per sample
- `weighted_unifrac.csv`: Weighted UniFrac distance matrix
- `unweighted_unifrac.csv`: Unweighted UniFrac distance matrix
- `phylo_diversity_plots.html`: Interactive visualizations

**Implementation**: `workflow_16s/src/workflow_16s/downstream/phylogenetic_diversity.py`

**Note**: Faith's PD is added to `adata.obs['faith_pd']` and automatically included in alpha diversity plots.

---

### 4. Differential Abundance (Multi-Method) ✅

**Purpose**: Identify features that differ significantly between groups using multiple methods.

**When to use**:
- Comparing treatments, conditions, or time points
- Finding biomarkers

**Configuration**:
```yaml
differential_abundance:
  enabled: true
  
  # Methods to run (can select multiple for comparison)
  methods: ['wilcoxon', 'deseq2']  
  # Options: 'deseq2', 'corncob', 'linda', 'aldex2', 'wilcoxon', 'ancom-bc'
  
  # Thresholds
  alpha: 0.05                # Significance level
  min_count: 10              # Minimum feature count
  min_prevalence: 0.1        # Minimum prevalence (10% of samples)
  lfc_threshold: 0           # Log2 fold-change threshold (0 = no threshold)
  
  # Consensus analysis
  consensus:
    enabled: true
    min_methods: 2           # Require N methods to agree
    max_p_adj: 0.05
  
  # Method-specific parameters
  deseq2:
    fit_type: 'parametric'   # 'parametric', 'local', or 'mean'
  
  wilcoxon:
    use_clr: true            # CLR-transform before testing
  
  aldex2:
    test: 'welch'            # 'welch', 'wilcox', or 'kw'
    mc_samples: 128          # Monte Carlo samples
    denom: 'all'             # Denominator: 'all', 'iqlr', 'zero', 'lvha'
  
  output_dir: '04_analysis/differential_abundance'
```

**Outputs**:
- `{method}_results.csv`: Per-method results
- `consensus_features.csv`: Features identified by ≥N methods
- `comparison_heatmap.html`: Agreement between methods
- `volcano_plots.html`: Interactive volcano plots per method

**Implementation**: `workflow_16s/src/workflow_16s/downstream/differential_abundance.py`

**Why multiple methods?**:
Different methods have different assumptions:
- **DESeq2**: Count-based, negative binomial (good for well-sequenced data)
- **Wilcoxon**: Non-parametric (robust to outliers)
- **ALDEx2**: Compositional-aware (accounts for relative abundance)
- **Consensus**: Features identified by multiple methods are more reliable

---

### 5. Compositional Networks ✅

**Purpose**: Infer co-occurrence networks that account for compositionality.

**When to use**:
- Identifying microbial interactions
- Understanding community structure

**Configuration**:
```yaml
networks:
  enabled: true
  method: 'spiec-easi'       # 'spiec-easi', 'sparcc', or 'proportionality'
  min_prevalence: 0.1        # Filter rare features
  
  # SPIEC-EASI parameters (graphical model-based)
  spiec_easi:
    method: 'mb'             # 'mb' (fast) or 'glasso' (accurate)
    nlambda: 20              # Number of regularization parameters
    lambda_min_ratio: 0.01
    pulsar_rep: 20           # Stability selection repetitions
    ncores: 4
  
  # SparCC parameters (correlation-based)
  sparcc:
    iterations: 20
    exclude_iterations: 10
    threshold: 0.1           # Correlation threshold
    bootstraps: 100          # For p-values
    p_threshold: 0.05
  
  # Proportionality parameters (compositional)
  proportionality:
    method: 'rho'            # 'rho' or 'phi'
    threshold: 0.7
  
  # Visualization
  plots:
    network_graph: true
    interactive: true
  
  output_dir: '04_analysis/networks'
```

**Outputs**:
- `network_edges.csv`: Edge list with weights
- `network_nodes.csv`: Node attributes
- `network_graph.html`: Interactive network visualization
- `network_metrics.csv`: Graph metrics (centrality, modularity, etc.)

**Implementation**: `workflow_16s/src/workflow_16s/downstream/compositional_networks.py`

**Why compositionality-aware?**: Standard correlation is invalid for compositional data (spurious correlations). These methods account for the sum constraint.

---

### 6. Longitudinal Analysis ✅

**Purpose**: Analyze temporal dynamics in time-series microbiome data.

**When to use**:
- Time-series experiments
- Tracking contamination events over time
- Stability/resilience studies

**Configuration**:
```yaml
longitudinal:
  enabled: true
  time_column: 'collection_date'    # Column with temporal information
  subject_column: 'subject_id'      # Subject/location identifier
  
  # Methods (requires R packages)
  methods: ['metalonda']            # 'zibr', 'metalonda', 'maaslin2'
  
  # Method-specific parameters
  metalonda:
    n_perm: 999                     # Permutations for p-values
    adjust_method: 'BH'             # Multiple testing correction
  
  maaslin2:
    random_effects: ['subject_id']  # Random effects
  
  output_dir: '04_analysis/longitudinal'
```

**Outputs**:
- `temporal_stability.csv`: Stability metrics per subject
- `trajectory_clusters.csv`: Clustered temporal trajectories
- `longitudinal_plots.html`: Interactive time-series plots

**Implementation**: `workflow_16s/src/workflow_16s/downstream/longitudinal.py`

**Requirements**: Time-series data with repeated measures

---

### 7. Batch Effect Detection & Correction ✅

**Purpose**: Detect and correct for technical batch effects.

**When to use**:
- Multi-batch experiments
- Data from multiple sequencing runs
- Meta-analyses combining datasets

**Configuration**:
```yaml
batch_effects:
  enabled: true
  batch_column: 'batch'                 # Column indicating batch
  preserve_columns: ['treatment']       # Biological variables to preserve
  correction_method: 'conqur'           # 'conqur' or 'combat'
  
  # Detection methods
  detection:
    permanova: true      # Variance partitioning
    silhouette: true     # Clustering quality
    pca: true            # PCA variance decomposition
    entropy: true        # Batch uniformity
  
  # Visualizations
  plots:
    pca: true            # PCA colored by batch/biology
    silhouette: true     # Silhouette plots
    heatmap: true        # Hierarchical clustering
  
  output_dir: '04_analysis/batch_effects'
```

**Outputs**:
- `batch_detection_results.csv`: Statistical tests for batch effects
- `corrected_data.h5ad`: Batch-corrected AnnData object
- `pca_before_after.html`: PCA before/after correction
- `variance_partition.csv`: Variance explained by batch vs biology

**Implementation**: `workflow_16s/src/workflow_16s/downstream/batch_effects.py`

**Note**: ConQuR is microbiome-specific and generally preferred over ComBat.

---

### 8. Decontamination ✅

**Purpose**: Identify and remove contaminant features.

**When to use**:
- Low-biomass samples
- Studies with negative controls
- DNA quantification data available

**Configuration**:
```yaml
decontamination:
  enabled: false  # Set to true if you have controls or DNA conc data
  method: 'combined'  # 'frequency', 'prevalence', or 'combined'
  
  # Frequency-based (requires DNA concentration)
  frequency:
    enabled: false
    dna_conc_column: 'quant_reading'
    threshold: 0.1  # Contamination probability threshold
  
  # Prevalence-based (requires negative controls)
  prevalence:
    enabled: false
    control_column: 'sample_type'
    control_value: 'negative_control'
    threshold: 0.5
  
  # Removal settings
  remove_contaminants: false  # Auto-remove identified contaminants
  
  output_dir: '04_analysis/decontamination'
```

**Outputs**:
- `contaminants.csv`: Identified contaminant features
- `contamination_scores.csv`: Probability scores per feature
- `before_after_comparison.html`: Sample composition before/after

**Implementation**: `workflow_16s/src/workflow_16s/downstream/decontamination.py`

**Requirements**: Either negative controls OR DNA concentration data

---

### 9. Metadata Profiling ✅

**Purpose**: Comprehensive metadata quality assessment (runs automatically before analysis).

**When to use**: Always (automatic)

**Configuration**: None (always runs)

**Outputs** (`04_analysis/metadata_profiling/`):
- `metadata_profile_report.html`: Interactive HTML report
- `ml_warnings.csv`: Warnings for machine learning readiness
- `missing_data.csv`: Missing data patterns
- `confounding_variables.csv`: Potential confounders (Cramér's V > 0.7)
- `class_imbalance.csv`: Class balance for categorical variables

**Implementation**: `workflow_16s/src/workflow_16s/downstream/metadata_profiler.py` (442 lines)

**Example warnings**:
```
⚠️  facility_match: Severe class imbalance (99% negative class)
⚠️  batch_original: High confounding with location (Cramér's V = 0.83)
⚠️  latitude: 67% missing data
```

---

## Integration Stubs for R Packages

The following modules have **integration stubs** (placeholder implementations) but require external R packages. They won't run until dependencies are installed.

### 🔄 Functional Profiling (PICRUSt2)

**Purpose**: Predict functional potential (KEGG pathways, EC numbers) from 16S data.

**Status**: Stub implemented, requires PICRUSt2 installation

**Configuration**:
```yaml
functional:
  enabled: false  # Set to true after installing PICRUSt2
  tool: 'picrust2'  # Currently only PICRUSt2 supported
  
  # Database for predictions
  database: 'kegg'  # 'kegg', 'cog', 'pfam', 'tigrfam'
  
  # Normalization
  normalize_by_copy_number: true
  
  # Output formats
  pathways: true      # Pathway abundances
  enzymes: true       # EC number abundances
  
  output_dir: '04_analysis/functional'
```

**Installation required**:
```bash
# Install PICRUSt2
conda install -c bioconda picrust2

# Test installation
picrust2_pipeline.py --version
```

**Stub location**: `workflow_16s/src/workflow_16s/downstream/functional.py`

**Implementation TODO**:
1. Call PICRUSt2 via subprocess
2. Parse output tables
3. Store in `adata.uns['functional']`
4. Generate pathway enrichment plots

---

### 🔄 Source Tracking (FEAST)

**Purpose**: Estimate source contributions to sink samples (e.g., soil → air contamination).

**Status**: Stub implemented, requires FEAST R package

**Configuration**:
```yaml
source_tracking:
  enabled: false  # Set to true after installing FEAST
  
  # Column indicating source/sink/unknown
  sample_type_column: 'sample_type'
  
  # Values for each category
  source_value: 'soil'
  sink_value: 'air'
  unknown_value: 'unknown'
  
  # FEAST parameters
  em_iterations: 1000
  rarefaction_depth: 1000  # Rarefy to this depth
  
  output_dir: '04_analysis/source_tracking'
```

**Installation required**:
```R
# Install FEAST in R
install.packages("devtools")
devtools::install_github("cozygene/FEAST")
```

**Stub location**: `workflow_16s/src/workflow_16s/downstream/source_tracking.py`

**Implementation TODO**:
1. Interface with R via `rpy2` or subprocess
2. Format data for FEAST input
3. Parse source proportion estimates
4. Generate source mixing plots

---

## Configuration Examples

### Minimal Configuration (Core Analyses Only)

```yaml
# Only essential analyses, skip optional modules
power_analysis:
  enabled: false

rarefaction:
  enabled: true  # Always recommended

phylogeny:
  enabled: false  # Skip if no tree

differential_abundance:
  enabled: true
  methods: ['wilcoxon']  # Single fast method

networks:
  enabled: false

longitudinal:
  enabled: false

batch_effects:
  enabled: false

decontamination:
  enabled: false

functional:
  enabled: false

source_tracking:
  enabled: false
```

**Runtime**: ~1-2 hours for 377 datasets

---

### Comprehensive Configuration (All Python Modules)

```yaml
# Enable everything implemented in Python
power_analysis:
  enabled: true
  run_before_da: true
  run_before_permanova: true

rarefaction:
  enabled: true
  plot_by_group: true

phylogeny:
  enabled: true
  tree_path: 'resources/silva_tree.nwk'
  alpha_diversity:
    faiths_pd: true
  beta_diversity:
    weighted_unifrac: true

differential_abundance:
  enabled: true
  methods: ['wilcoxon', 'deseq2', 'aldex2']
  consensus:
    enabled: true
    min_methods: 2

networks:
  enabled: true
  method: 'spiec-easi'

longitudinal:
  enabled: true
  time_column: 'collection_date'
  subject_column: 'location'

batch_effects:
  enabled: true
  correction_method: 'conqur'

decontamination:
  enabled: false  # Only if you have controls

# R packages (stubs)
functional:
  enabled: false

source_tracking:
  enabled: false
```

**Runtime**: ~4-6 hours for 377 datasets

---

### Publication-Ready Configuration

```yaml
# Generate all publication-quality outputs
power_analysis:
  enabled: true
  target_power: 0.8

rarefaction:
  enabled: true
  n_depths: 50  # High-resolution curves
  plot_by_group: true

phylogeny:
  enabled: true
  alpha_diversity:
    faiths_pd: true
  beta_diversity:
    weighted_unifrac: true
    unweighted_unifrac: true

differential_abundance:
  enabled: true
  methods: ['wilcoxon', 'deseq2', 'aldex2']
  consensus:
    enabled: true
    min_methods: 2
    max_p_adj: 0.01  # Stricter threshold

networks:
  enabled: true
  method: 'spiec-easi'
  spiec_easi:
    pulsar_rep: 50  # More stable networks

longitudinal:
  enabled: true

batch_effects:
  enabled: true
  detection:
    permanova: true
    silhouette: true
    pca: true
    entropy: true

decontamination:
  enabled: true
  method: 'combined'
```

---

## Troubleshooting

### Module Not Running

**Symptom**: Config shows `enabled: true` but module doesn't execute

**Debugging**:
1. Check log for warnings:
   ```bash
   grep "skipping\|failed\|warning" workflow_16s/src/logs/*.log
   ```

2. Verify config key matches exactly:
   ```bash
   grep -A5 "phylogeny:" config/config.yaml
   ```

3. Check for missing dependencies:
   - Phylogenetic: Requires tree file
   - Decontamination: Requires controls OR DNA conc
   - Longitudinal: Requires `time_column` and `subject_column`

4. Validate column names exist:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad('path/to/data.h5ad')
   print(adata.obs.columns)
   ```

---

### R Package Integration

**Symptom**: Functional/source tracking stubs don't run

**Solution**: These require external R packages not yet integrated

**Workaround**:
1. Export data: `adata.to_df().to_csv('abundance.csv')`
2. Run R tools manually
3. Import results back

**Full integration TODO**:
- Implement `rpy2` interface
- Add R package installation to `setup.sh`
- Test with example data

---

### Performance Issues

**Symptom**: Analysis takes >12 hours

**Solutions**:

1. **Disable expensive modules**:
   ```yaml
   networks:
     enabled: false  # Can take 2-3 hours
   
   differential_abundance:
     methods: ['wilcoxon']  # Fast method only
   ```

2. **Reduce sample size for testing**:
   ```yaml
   rarefaction:
     plot_individual_samples: false
   
   power_analysis:
     run_before_da: false
   ```

3. **Increase parallelization**:
   ```yaml
   n_cpus: 16  # Use more cores
   ```

4. **Use cached results**:
   - Cached concatenation: `03_processed_data/cache/`
   - Don't delete intermediate files

---

### Missing Outputs

**Symptom**: Module runs but outputs not found

**Check**:
1. Output directory structure:
   ```bash
   ls -R 04_analysis/
   ```

2. Log file for completion:
   ```bash
   grep "complete" workflow_16s/src/logs/*.log
   ```

3. Errors during execution:
   ```bash
   grep "Error\|Exception" workflow_16s/src/logs/*.log
   ```

---

## Advanced: Adding New Modules

**Template for creating toggleable module**:

1. **Create module file**: `workflow_16s/src/workflow_16s/downstream/my_module.py`

2. **Add config block** to `config/config.yaml`:
   ```yaml
   my_module:
     enabled: false
     parameter1: value1
     output_dir: '04_analysis/my_module'
   ```

3. **Add to workflow** in `steps/analysis.py`:
   ```python
   # Import
   from workflow_16s.downstream.my_module import run_my_analysis
   
   # Add to run_analysis_suite()
   my_config = getattr(workflow.config, 'my_module', None)
   my_enabled = getattr(my_config, 'enabled', False) if my_config else False
   
   if my_enabled:
       try:
           workflow.logger.info("Running my custom analysis...")
           run_my_analysis(
               workflow.adata,
               param1=getattr(my_config, 'parameter1', 'default'),
               output_dir=workflow.output_dir / 'my_module'
           )
       except Exception as e:
           workflow.logger.warning(f"My analysis failed: {e}")
   ```

4. **Test**:
   ```bash
   python src/workflow_16s/downstream/test.py --data_dir test/data --output_dir test/output
   ```

---

## Summary

**Production-Ready** (✅ Python, Fully Tested):
- Metadata Profiling *(automatic)*
- Power Analysis
- Rarefaction Curves
- Phylogenetic Diversity
- Differential Abundance
- Compositional Networks
- Longitudinal Analysis
- Batch Effect Detection/Correction
- Decontamination

**Integration Stubs** (🔄 Requires External R Packages):
- Functional Profiling (PICRUSt2)
- Source Tracking (FEAST)

**Default Recommendation**: Enable everything in the "Production-Ready" list for comprehensive analysis. Disable stubs unless you install external dependencies.

---

For implementation details, see module source code:
- `workflow_16s/src/workflow_16s/downstream/` (individual modules)
- `workflow_16s/src/workflow_16s/downstream/steps/analysis.py` (orchestrator)
- `workflow_16s/config/config.yaml` (all toggles)
