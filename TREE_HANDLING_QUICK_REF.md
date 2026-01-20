# Quick Reference: Handling Missing Phylogenetic Trees

## TL;DR

Add to your `config.yaml`:

```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: auto  # Let the system decide
```

**The `auto` strategy will:**
1. Use existing tree if coverage is good (>80%)
2. Merge per-dataset trees if multiple exist
3. Build de novo tree if sequences available
4. Skip phylogenetic metrics if nothing else works

---

## Strategy Cheat Sheet

| Strategy | When to Use | Speed | Accuracy |
|----------|-------------|-------|----------|
| `auto` | **Default - let system decide** | Varies | Best available |
| `graceful_degradation` | Skip phylo metrics, fast results | ⚡⚡⚡ | N/A |
| `subset_tree_extraction` | Tree exists, >80% coverage | ⚡⚡⚡ | ★★★ |
| `tree_merging` | Multiple per-dataset trees | ⚡⚡ | ★★ |
| `partial_analysis` | Tree exists, 50-80% coverage | ⚡⚡ | ★★★ |
| `denovo_tree_building` | Sequences available, need accuracy | ⚡ | ★★★★ |

---

## Common Scenarios

### Scenario 1: "I just want it to work"

```yaml
phylogenetic_diversity:
  enabled: true
  # That's it! Uses 'auto' by default
```

### Scenario 2: "I don't care about phylogenetic diversity"

```yaml
phylogenetic_diversity:
  enabled: false
```

### Scenario 3: "I have time and want the best tree"

```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: denovo_tree_building
```

### Scenario 4: "I need results fast"

```yaml
phylogenetic_diversity:
  enabled: true
  missing_tree_strategy: graceful_degradation
```

---

## What Gets Logged

Look for these messages in your logs:

✅ **Success:**
```
INFO: Auto-selected strategy: subset_tree_extraction
INFO: Tree covers 95.3% of features - will extract subset
INFO: ✅ Successfully obtained tree: /path/to/all_features.tree
```

⚠️ **Partial Success:**
```
WARNING: Tree covers only 62.1% of features
INFO: Will use partial analysis
WARNING: Phylogenetic diversity will only use 6,210 features
```

❌ **No Tree Available:**
```
WARNING: ⚠️  No tree available - phylogenetic diversity will be skipped
INFO: Available metrics: Shannon, Simpson, Observed Features, Evenness
```

---

## Decision Flow

```
Is phylogenetic_diversity.enabled = true?
│
├─ NO → Skip everything, continue with non-phylo metrics
│
└─ YES → Does tree file exist?
    │
    ├─ YES → Use it (subset_tree_extraction)
    │
    └─ NO → Check adata.uns['phylogenetic_tree']
        │
        ├─ Has tree string (>80% coverage) → Extract subset
        ├─ Has tree string (50-80% coverage) → Partial analysis
        ├─ Has multiple trees (dict) → Merge trees
        ├─ Has sequences → Build de novo tree
        └─ Nothing available → Skip phylo metrics
```

---

## Performance

| Features | Strategy | Typical Time |
|----------|----------|--------------|
| 100 | De novo build | 30 sec |
| 1,000 | De novo build | 5 min |
| 10,000 | De novo build | 30 min |
| Any | Subset extraction | <1 sec |
| Any | Tree merging | <5 sec |
| Any | Graceful degradation | 0 sec |

---

## Troubleshooting One-Liners

**Problem:** Pipeline says "No tree available"  
**Fix:** Check if `adata.var['sequence']` has sequences. If yes, use `denovo_tree_building`.

**Problem:** De novo building takes forever  
**Fix:** Use `partial_analysis` or `graceful_degradation` instead.

**Problem:** Tree only covers 40% of features  
**Fix:** Use `denovo_tree_building` for better coverage.

**Problem:** Want to test without waiting  
**Fix:** Use `graceful_degradation` for first run, enable tree later.

---

## Files Created

After successful tree handling, you'll find:

```
project_dir/
└── 04_analysis/
    ├── all_features.tree         # Final phylogenetic tree (Newick format)
    ├── all_features.fasta         # Feature sequences (if de novo building)
    └── phylogenetic_diversity/   # Results directory
        ├── faith_pd.tsv
        ├── weighted_unifrac.tsv
        └── unweighted_unifrac.tsv
```

---

## API Usage

For programmatic use:

```python
from workflow_16s.downstream.tree_handler import handle_missing_tree

# Auto-select strategy
tree_path = handle_missing_tree(
    adata, 
    config, 
    output_dir, 
    strategy='auto'
)

if tree_path:
    print(f"Tree ready: {tree_path}")
else:
    print("No tree - skipping phylogenetic analysis")
```

---

## See Also

- Full documentation: `TREE_HANDLING_STRATEGIES.md`
- Tree building config: `config.yaml` → `preprocessing.rebuild_tree`
- Phylo diversity config: `config.yaml` → `phylogenetic_diversity`
