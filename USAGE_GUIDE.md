# workflow_16s Usage Guide
## Integrated Scientific Enhancements - January 7, 2026

This guide explains how to use the newly integrated scientific methods in your 16S rRNA analysis pipeline.

---

## 🎯 Quick Start

The new features are **already integrated** into the main workflow. Simply enable them in your `config.yaml`:

```yaml
# Enable batch effect correction
batch_effects:
  enabled: True
  batch_column: "batch_original"
  correction:
    enabled: True
    method: "conqur"  # or "combat"

# Enable phylogenetic diversity metrics
phylogenetic_diversity:
  enabled: True
  metrics:
    faith_pd: True
    unifrac_weighted: True
    unifrac_unweighted: True

# Enable multi-method differential abundance
differential_abundance:
  enabled: True
  group_column: "nuclear_contamination_status"
  methods: ["deseq2", "aldex2", "wilcoxon"]
  consensus:
    enabled: True
    min_agreement: 2  # Features significant in ≥2 methods

# Enable compositional networks (optional)
compositional_networks:
  enabled: False  # Set to True when ready
  methods: ["sparcc"]

# Enable decontamination (if you have negative controls)
decontamination:
  enabled: False  # Set to True if you have controls
  method: "combined"
  neg_control_column: "sample_type"
  control_value: "negative_control"
```

Then run as usual:
```bash
bash run.sh --config config/config.yaml
```

---

## 📊 What's New

### 1. **Batch Effect Correction** (INTEGRATED)

**What it does:** Detects and removes technical batch effects while preserving biological signal.

**When to use:** Always! Batch effects are ubiquitous in multi-dataset studies.

**How it works:**
1. **Detection:** PERMANOVA quantifies batch variance
2. **Correction:** ConQuR (recommended) or ComBat removes batch signal
3. **Validation:** Before/after comparison ensures biology is preserved

**Results location:**
- `output_dir/batch_effects/batch_diagnostics.html` - Interactive PCA plots
- `output_dir/batch_effects/batch_pca.html` - Before/after visualization
- Corrected data automatically used in downstream analysis

**Config example:**
```yaml
batch_effects:
  enabled: True
  batch_column: "batch_original"  # Column with batch IDs
  detection:
    methods: ["permanova", "pca", "silhouette"]
  correction:
    enabled: True
    method: "conqur"  # Microbiome-specific (recommended)
```

**Interpretation:**
- **Batch R² < 0.05:** Minimal batch effects, correction optional
- **Batch R² 0.05-0.10:** Moderate batch effects, correction recommended
- **Batch R² > 0.10:** Strong batch effects, correction critical

---

### 2. **Phylogenetic Diversity** (INTEGRATED)

**What it does:** Calculates tree-based diversity metrics (Faith's PD, UniFrac).

**When to use:** When you have a phylogenetic tree (automatically built or provide your own).

**How it works:**
1. **Faith's PD:** Sum of branch lengths for observed species
2. **UniFrac:** Distance between communities based on shared evolutionary history
3. **Weighted UniFrac:** Accounts for abundance differences

**Results location:**
- `adata.obs['faith_pd']` - Faith's PD values per sample
- `adata.uns['weighted_unifrac']` - UniFrac distance matrix
- `output_dir/phylogenetic_diversity/` - Plots and diagnostics

**Config example:**
```yaml
phylogenetic_diversity:
  enabled: True
  tree_path: null  # null = auto-build with FastTree
  metrics:
    faith_pd: True
    unifrac_weighted: True
    unifrac_unweighted: True
```

**Why use phylogenetic metrics:**
- More statistical power than taxonomy-only metrics
- Account for evolutionary relationships
- Detect deep vs. shallow community shifts

---

### 3. **Multi-Method Differential Abundance** (INTEGRATED)

**What it does:** Tests for differentially abundant features using 5 methods, finds consensus.

**When to use:** To identify microbes associated with your experimental variable (e.g., contamination status).

**How it works:**
1. Runs multiple DA methods (DESeq2, ALDEx2, LinDA, corncob, Wilcoxon)
2. Finds features significant in ≥N methods
3. Reports consensus features with cross-method validation

**Results location:**
- `adata.uns['da_comparison']` - Results from all methods
- `adata.uns['da_consensus']` - Consensus features (high confidence)
- `output_dir/differential_abundance/` - Method comparison plots

**Config example:**
```yaml
differential_abundance:
  enabled: True
  group_column: "nuclear_contamination_status"
  methods: ["deseq2", "aldex2", "wilcoxon"]  # 3-5 methods recommended
  fdr_threshold: 0.05
  consensus:
    enabled: True
    min_agreement: 2  # ≥2 methods = high confidence
```

**Interpretation:**
```python
# Access results in Python
consensus = adata.uns['da_consensus']
print(f"High-confidence biomarkers: {len(consensus)}")

# Features in consensus are significantly different AND validated by multiple methods
```

**Why use multiple methods:**
- Different assumptions (parametric vs. non-parametric)
- Reduces false positives
- Consensus features are more robust

---

### 4. **Decontamination** (INTEGRATED - optional)

**What it does:** Identifies and removes contaminant sequences using negative controls.

**When to use:** If you have negative control samples (extraction blanks, PCR negatives).

**How it works:**
1. **Frequency method:** Contaminants inversely correlate with DNA concentration
2. **Prevalence method:** Contaminants more prevalent in controls
3. **Combined:** Uses both methods

**Results location:**
- `adata.uns['contaminants']` - List of identified contaminants
- `output_dir/decontamination/` - Diagnostic plots
- Cleaned data automatically used downstream

**Config example:**
```yaml
decontamination:
  enabled: True
  method: "combined"  # frequency, prevalence, or combined
  neg_control_column: "sample_type"
  control_value: "negative_control"
  threshold: 0.1  # Stringency (lower = more stringent)
  remove_contaminants: True
```

**Prerequisites:**
- Negative control samples in your dataset
- DNA concentration data (for frequency method)

---

### 5. **Compositional Networks** (INTEGRATED - optional)

**What it does:** Infers co-occurrence networks accounting for compositional data constraints.

**When to use:** To study microbial interactions and network structure.

**How it works:**
1. **SPIEC-EASI:** Sparse inverse covariance estimation
2. **SparCC:** Correlation corrected for compositionality
3. **Proportionality:** rho/phi metrics

**Results location:**
- `adata.uns['network']` - NetworkX graph object
- `output_dir/compositional_networks/` - Network visualizations

**Config example:**
```yaml
compositional_networks:
  enabled: True
  methods: ["sparcc"]  # sparcc, spiec_easi, or proportionality
  min_prevalence: 0.1
  threshold: 0.3  # Correlation threshold
```

**Why not standard correlation:**
- Compositional data has spurious correlations
- These methods account for the compositional constraint
- More accurate network inference

---

## 🔬 Workflow Integration Points

The new features are integrated at specific points for scientific reasons:

```
1. Load Data
   ↓
2. Filter Low Quality
   ↓
3. [NEW] Decontamination ← Remove contaminants BEFORE batch correction
   ↓
4. [NEW] Batch Correction ← Remove batch effects BEFORE analysis
   ↓
5. [NEW] Phylogenetic Diversity ← Calculate tree-based metrics
   ↓
6. Alpha/Beta Diversity ← Standard metrics (now includes Faith's PD)
   ↓
7. [NEW] Differential Abundance ← Multi-method with consensus
   ↓
8. Machine Learning
   ↓
9. [NEW] Compositional Networks ← Interaction networks
   ↓
10. Results Synthesis
```

**Why this order:**
1. **Decontam before batch:** Contaminants can create batch effects
2. **Batch before analysis:** Batch effects dominate biological signal
3. **Phylo early:** Metrics used in diversity analysis
4. **DA after diversity:** Uses filtered, batch-corrected data

---

## 📖 Example Workflows

### Example 1: Basic Analysis (Recommended)

```yaml
# config.yaml - Minimal recommended setup
batch_effects:
  enabled: True  # Always enable
  batch_column: "batch_original"

phylogenetic_diversity:
  enabled: True  # Enable if tree available

differential_abundance:
  enabled: True
  methods: ["deseq2", "wilcoxon"]  # Fast, robust
  consensus:
    min_agreement: 2
```

### Example 2: Publication-Quality (All Features)

```yaml
# config.yaml - Maximum rigor
batch_effects:
  enabled: True
  correction:
    method: "conqur"

decontamination:
  enabled: True  # If you have controls
  method: "combined"

phylogenetic_diversity:
  enabled: True
  metrics:
    faith_pd: True
    unifrac_weighted: True
    unifrac_unweighted: True

differential_abundance:
  enabled: True
  methods: ["deseq2", "aldex2", "linda", "corncob", "wilcoxon"]  # All 5
  consensus:
    min_agreement: 3  # High confidence

compositional_networks:
  enabled: True
  methods: ["sparcc", "spiec_easi"]
```

### Example 3: Quick Exploratory

```yaml
# config.yaml - Fast, skip intensive methods
batch_effects:
  enabled: True
  correction:
    enabled: False  # Detect only, don't correct

phylogenetic_diversity:
  enabled: False  # Skip tree-based metrics

differential_abundance:
  enabled: True
  methods: ["wilcoxon"]  # Fastest method only
```

---

## 🐛 Troubleshooting

### "ConQuR package not found"

**Solution:** Install R packages:
```r
# In R console
install.packages("BiocManager")
BiocManager::install("phyloseq")
devtools::install_github("wdl2459/ConQuR")
```

### "Batch column not found"

**Solution:** Check column name in your data:
```python
import scanpy as sc
adata = sc.read_h5ad("your_file.h5ad")
print(adata.obs.columns)  # Find the actual batch column name
```

### "No phylogenetic tree found"

**Solution:** Either:
1. Let pipeline build tree automatically (`tree_path: null`)
2. Provide tree: `tree_path: "path/to/tree.nwk"`
3. Disable phylo features: `enabled: False`

### "Decontamination failed"

**Solution:** Check you have:
1. Negative control samples
2. Correct control column name
3. Correct control value

### "Not enough samples for batch correction"

**Solution:** ConQuR requires ≥5 samples per batch. Either:
1. Combine small batches
2. Use ComBat instead
3. Skip batch correction

---

## 📊 Interpreting Results

### Batch Effect Results

Located in: `output_dir/batch_effects/`

**Key files:**
- `batch_diagnostics.html` - Interactive PCA before/after correction
- `batch_silhouette.html` - How well batches separate
- `batch_permanova_results.txt` - Statistical test results

**What to look for:**
- **Before correction:** High batch R², tight batch clustering
- **After correction:** Low batch R², biology signal preserved
- **Biology R² > Batch R²:** Good correction

### Phylogenetic Diversity Results

Located in: `output_dir/phylogenetic_diversity/`

**Key files:**
- Faith's PD added to `adata.obs['faith_pd']`
- UniFrac distances in `adata.uns['weighted_unifrac']`

**What to look for:**
- Higher Faith's PD = more evolutionary diversity
- UniFrac separates communities by evolutionary distance
- Use in PCoA plots for phylogenetically-aware ordination

### Differential Abundance Results

Located in: `output_dir/differential_abundance/`

**Key files:**
- `method_comparison.csv` - Results from all methods
- `consensus_features.csv` - High-confidence biomarkers
- `venn_diagram.html` - Method overlap visualization

**What to look for:**
- Consensus features: Significant in ≥N methods
- Effect size: Log2 fold change
- Adjust p-values: FDR < 0.05

**Access in Python:**
```python
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("output_dir/final_processed_adata.h5ad")

# View consensus biomarkers
consensus = adata.uns['da_consensus']
print(f"High-confidence biomarkers: {len(consensus)}")
print(consensus[['feature', 'n_methods_significant', 'mean_log2fc']])

# View all method results
comparison = adata.uns['da_comparison']
```

---

## 🎓 Best Practices

### 1. Always Enable Batch Correction
Even if you think you don't have batches, dataset origin creates batches.

### 2. Use Consensus Differential Abundance
Single-method results have higher false positive rates.

### 3. Check Batch Correction Didn't Remove Biology
Compare biology R² before and after. Should stay similar or increase.

### 4. Phylogenetic Metrics When Possible
More powerful than taxonomy-only metrics.

### 5. Document Your Choices
Note which methods you used and why in your manuscript.

---

## 📚 Methods Section Template

For your manuscript:

> **Batch Effect Correction:** Technical batch effects were quantified using PERMANOVA and corrected using ConQuR (Jiang et al. 2020), a microbiome-specific batch correction method that preserves biological signal.
>
> **Phylogenetic Diversity:** Phylogenetic diversity was assessed using Faith's Phylogenetic Diversity (Faith 1992) and weighted/unweighted UniFrac distances (Lozupone & Knight 2005).
>
> **Differential Abundance:** Differentially abundant taxa were identified using five complementary methods (DESeq2, ALDEx2, LinDA, corncob, Wilcoxon). Consensus features were defined as those significant (FDR < 0.05) in at least 2/5 methods.
>
> **Compositional Data:** All transformations accounted for the compositional nature of microbiome data using centered log-ratio transformation with multiplicative zero replacement (Martin-Fernández et al. 2003).

---

## 🔗 References

- Jiang et al. (2020). ConQuR: batch effects removal for microbiome data. *mSystems*.
- Faith (1992). Conservation evaluation and phylogenetic diversity. *Biological Conservation*.
- Lozupone & Knight (2005). UniFrac: a new phylogenetic method. *Applied and Environmental Microbiology*.
- Martin-Fernández et al. (2003). Dealing with zeros in compositional data. *Mathematical Geology*.
- Love et al. (2014). Moderated estimation with DESeq2. *Genome Biology*.
- Fernandes et al. (2014). ALDEx2 analysis. *Microbiome*.
- Zhou et al. (2022). LinDA analysis. *Genome Biology*.

---

## ✅ Migration Checklist

Moving from old to new pipeline:

- [ ] Update config.yaml with new sections
- [ ] Enable batch_effects (recommended: always True)
- [ ] Enable phylogenetic_diversity (if tree available)
- [ ] Enable differential_abundance (recommended: always True)
- [ ] Set differential_abundance methods (≥2 recommended)
- [ ] Configure decontamination (only if you have controls)
- [ ] Test on small dataset first
- [ ] Verify batch correction preserves biology
- [ ] Check consensus DA features make biological sense
- [ ] Archive old results before running new pipeline

---

## 💡 Quick Tips

1. **Start simple:** Enable batch correction and 2-method DA first
2. **Check diagnostics:** Review batch PCA and DA Venn diagrams
3. **Use consensus:** Trust features significant in multiple methods
4. **Save intermediate:** Pipeline saves corrected data automatically
5. **Read the logs:** Pipeline reports key statistics in logs

---

**Questions?** Check IMPLEMENTATION_PROGRESS.md for technical details or raise an issue on GitHub.

**Ready to go?** Enable features in config.yaml and run: `bash run.sh`
