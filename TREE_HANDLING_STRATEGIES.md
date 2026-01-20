# Phylogenetic Tree Handling Strategies

## Overview

When working with concatenated datasets from multiple sources in the downstream workflow, phylogenetic trees may be missing, incomplete, or incompatible. This document describes the available strategies for handling these situations.

## The Problem

In a multi-dataset analysis:
- Some datasets may have phylogenetic trees, others may not
- Trees from different datasets may have disjoint feature sets
- Concatenated features may not all appear in any single tree
- Tree building may have failed for some datasets during upstream processing

## Available Strategies

### 1. Graceful Degradation (Default Fallback)

**When to use:**
- No trees available from any source
- No sequence data available for de novo tree building
- You want the fastest analysis without phylogenetic metrics

**What it does:**
- Skips all phylogenetic diversity metrics (Faith's PD, UniFrac)
- Proceeds with non-phylogenetic metrics only (Shannon, Simpson, Observed Features, Bray-Curtis, Jaccard)

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: graceful_degradation
```

**Pros:**
- Fast - no tree computation needed
- Safe - no risk of using incorrect tree topology
- Analysis can proceed immediately

**Cons:**
- Loses valuable phylogenetic information
- Cannot calculate Faith's PD or UniFrac distances

---

### 2. Tree Merging (Supertree Construction)

**When to use:**
- Multiple per-dataset trees are available in `adata.uns['phylogenetic_tree']`
- You want to preserve existing phylogenetic relationships
- Features from different datasets don't overlap much

**What it does:**
- Combines multiple per-dataset trees into a single supertree
- Uses star topology to connect disjoint tree components
- Adds missing features as single-node branches

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: tree_merging
```

**Implementation Details:**
1. Extracts trees from `adata.uns['phylogenetic_tree']` (dict format expected)
2. Creates root node
3. Adds each per-dataset tree as a child of root
4. Adds features not in any tree as single nodes with branch length 1.0

**Pros:**
- Preserves original phylogenetic relationships within each dataset
- Allows partial phylogenetic analysis
- No need for alignment/tree building

**Cons:**
- Star topology connections are artificial (no meaningful branch lengths)
- May create unusual tree structure
- Requires validation that merged tree makes biological sense

**Output:**
```
Tree structure:
root
├── dataset1_tree (original topology preserved)
│   ├── feature_A
│   ├── feature_B
│   └── ...
├── dataset2_tree (original topology preserved)
│   ├── feature_X
│   ├── feature_Y
│   └── ...
├── feature_orphan1 (no tree, added as single node)
└── feature_orphan2 (no tree, added as single node)
```

---

### 3. De Novo Tree Building

**When to use:**
- Feature sequences are available in `adata.var['sequence']`
- You want the most accurate tree for your specific feature set
- Computational time is acceptable (may take 10+ minutes for large datasets)

**What it does:**
- Exports all feature sequences to FASTA
- Runs MAFFT multiple sequence alignment
- Builds phylogenetic tree with FastTree
- Roots tree using midpoint rooting

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: denovo_tree_building

preprocessing:
  rebuild_tree:
    enabled: true  # Will be used by strategy
    alignment_method: mafft
    tree_method: fasttree
```

**Pros:**
- Tree topology is optimal for your exact feature set
- No assumptions about tree compatibility across datasets
- Handles any feature set (no missing features)

**Cons:**
- **Computationally expensive** (alignment + tree building)
- **Time-consuming** (can take 10-30 minutes for thousands of features)
- Requires feature sequences in data
- May fail if sequences are low quality or too divergent

**Performance Estimates:**
- 100 features: ~30 seconds
- 1,000 features: ~5 minutes
- 10,000 features: ~30 minutes
- 50,000+ features: May require hours

---

### 4. Partial Analysis

**When to use:**
- Tree covers ≥50% but <80% of features
- You want to maintain phylogenetic accuracy
- You're okay analyzing a subset of features for phylogenetic metrics

**What it does:**
- Identifies features present in the tree
- Prunes tree to only include these features
- Exports pruned tree for phylogenetic diversity analysis
- Full dataset used for non-phylogenetic metrics

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: partial_analysis
```

**Pros:**
- Uses only reliable phylogenetic information
- No artificial tree topology
- Maintains phylogenetic accuracy

**Cons:**
- Reduces feature count for phylogenetic analysis
- May introduce bias if missing trees are non-random
- Splits analysis into two parts (phylo subset vs full dataset)

**Example Output:**
```
Total features: 10,000
Tree covers: 6,500 features (65%)
Phylogenetic diversity calculated on 6,500 features
Other metrics calculated on all 10,000 features
```

---

### 5. Subset Tree Extraction

**When to use:**
- Single comprehensive tree exists with >80% feature coverage
- You want to use existing tree but need to prune it
- Tree was built on a superset of your current features

**What it does:**
- Loads tree from `adata.uns['phylogenetic_tree']`
- Prunes tree to only features in `adata.var_names`
- Removes unnecessary branches
- Exports cleaned tree

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: subset_tree_extraction
```

**Pros:**
- Maintains accurate phylogenetic relationships
- Fast (just tree pruning, no building)
- Removes unnecessary computational overhead

**Cons:**
- Requires good tree coverage (>80% recommended)
- Won't work if features span multiple disjoint trees
- Features must be present in source tree

---

### 6. Auto (Recommended)

**When to use:**
- You want the system to choose the best strategy automatically
- You're not sure which strategy is most appropriate

**What it does:**
Makes intelligent decisions based on available data:

1. **If tree exists in adata.uns and covers >80% of features:**
   → Use `subset_tree_extraction`

2. **If tree exists and covers 50-80% of features:**
   → Use `partial_analysis`

3. **If multiple trees exist (dict format):**
   → Use `tree_merging`

4. **If no tree but sequences are available:**
   → Use `denovo_tree_building`

5. **Otherwise:**
   → Use `graceful_degradation`

**Configuration:**
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: auto  # Default
```

**Decision Tree:**
```
Has tree?
├── No
│   ├── Has sequences? → denovo_tree_building
│   └── No sequences → graceful_degradation
└── Yes
    ├── Multiple trees (dict)? → tree_merging
    └── Single tree
        ├── Coverage ≥80%? → subset_tree_extraction
        ├── Coverage ≥50%? → partial_analysis
        └── Coverage <50%
            ├── Has sequences? → denovo_tree_building
            └── No sequences → graceful_degradation
```

---

## Configuration Examples

### Minimal Configuration (Auto Strategy)
```yaml
phylogenetic_diversity:
  enabled: true
  # missing_tree_strategy defaults to 'auto'
  metrics:
    faith_pd: true
    unifrac_weighted: true
    unifrac_unweighted: true
```

### Explicit Strategy Selection
```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: denovo_tree_building
  metrics:
    faith_pd: true
    unifrac_weighted: true
    unifrac_unweighted: false
```

### Disable Phylogenetic Analysis
```yaml
phylogenetic_diversity:
  enabled: false  # Skip entirely
```

---

## Logging and Diagnostics

When tree handling is active, the pipeline logs detailed information:

```
INFO: Phylogenetic diversity enabled but no tree file found
INFO: Attempting to handle missing tree...
INFO: Tree handling strategy: auto
INFO: Available strategies:
INFO:   - auto: Automatically select best strategy
INFO:   - graceful_degradation: Skip phylogenetic metrics
INFO:   - tree_merging: Merge per-dataset trees
INFO:   - denovo_tree_building: Build new tree from sequences
INFO:   - partial_analysis: Analyze only tree-covered features
INFO:   - subset_tree_extraction: Extract subtree for current features
INFO: Auto-selected strategy: denovo_tree_building
INFO: Strategy: Build new phylogenetic tree from concatenated features
INFO: Exporting feature sequences to FASTA...
INFO: Building phylogenetic tree (this may take several minutes)...
INFO: ✅ Successfully obtained tree: /path/to/all_features.tree
```

---

## Best Practices

### For Production Analyses

1. **First run:** Use `auto` strategy to see what the system recommends
2. **Review logs:** Check which strategy was selected and why
3. **Validate tree:** If tree was built or merged, inspect it visually
4. **Consider alternatives:** If auto selected de novo building but it's too slow, consider partial_analysis
5. **Document choice:** Note which strategy was used in your methods

### For Quick Exploratory Analyses

- Use `graceful_degradation` to skip phylogenetic metrics entirely
- Focus on taxonomy-based diversity metrics
- Re-run with phylogenetic metrics later if needed

### For Publications

- Use `denovo_tree_building` for most accurate phylogenetic relationships
- Validate tree topology makes biological sense
- Report tree building method in methods section
- Include tree file in supplementary materials

---

## Troubleshooting

### "No tree available - phylogenetic diversity will be skipped"

**Cause:** None of the strategies could produce a valid tree

**Solutions:**
1. Check if sequences are in `adata.var['sequence']`
2. Verify `adata.uns['phylogenetic_tree']` contains tree data
3. Try explicitly setting `missing_tree_strategy: denovo_tree_building`
4. Check logs for specific errors during tree building

### "Tree covers only X% of features"

**Cause:** Existing tree has poor overlap with current feature set

**Solutions:**
1. Use `denovo_tree_building` to build a new tree
2. Accept partial coverage with `partial_analysis`
3. Skip phylogenetic metrics with `graceful_degradation`

### De Novo Tree Building Takes Too Long

**Cause:** Large number of features (>10,000)

**Solutions:**
1. Use `partial_analysis` on a subset
2. Use `subset_tree_extraction` if a tree exists
3. Filter features before concatenation (prevalence, abundance thresholds)
4. Run overnight or on HPC cluster

### "Failed to parse tree"

**Cause:** Corrupted tree string in `adata.uns['phylogenetic_tree']`

**Solutions:**
1. Use `denovo_tree_building` to rebuild from sequences
2. Check upstream logs for tree building errors
3. Verify Newick format is valid

---

## Implementation Details

### Module Location

`workflow_16s/src/workflow_16s/downstream/tree_handler.py`

### Key Functions

```python
from workflow_16s.downstream.tree_handler import (
    handle_missing_tree,
    get_tree_handling_strategy,
)

# Auto-handle missing tree
tree_path = handle_missing_tree(adata, config, output_dir, strategy='auto')

# Use specific strategy
strategy = get_tree_handling_strategy('denovo_tree_building')
tree_path = strategy.handle(adata, config, output_dir)
```

### Integration Points

1. **Downstream analysis** (`steps/analysis.py`): Main entry point for tree handling
2. **Preprocessing** (`preprocessing.py`): Tree building functions (`rebuild_tree`, `export_fasta`)
3. **Upstream** (`upstream/metadata/utils.py`): Tree storage in `adata.uns['phylogenetic_tree']`

---

## Future Enhancements

Potential improvements to tree handling:

1. **Consensus tree building:** Build tree from each dataset, then create consensus
2. **Reference tree grafting:** Graft new features onto reference tree (e.g., SEPP)
3. **Tree quality metrics:** Assess and report tree confidence/support values
4. **Parallel tree building:** Speed up de novo building with parallelization
5. **Cloud-based tree building:** Offload expensive computation to cloud services

---

## References

- **MAFFT:** Katoh & Standley (2013) Molecular Biology and Evolution 30:772-780
- **FastTree:** Price et al. (2010) PLoS ONE 5(3):e9490
- **Faith's PD:** Faith (1992) Biological Conservation 61:1-10
- **UniFrac:** Lozupone & Knight (2005) Applied and Environmental Microbiology 71:8228-8235
