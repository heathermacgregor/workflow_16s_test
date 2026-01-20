# ML Visualization and Batch Effect Analysis Guide
**Created:** 2026-01-12  
**Purpose:** Generate comprehensive, interpretable ML outputs with strategy comparison and batch effect analysis

---

## Overview

The enhanced ML visualization framework provides:

1. **Strategy Comparison Dashboards** - Compare baseline vs agnostic vs group_validated approaches
2. **Group Fingerprint Plots** - Visualize biomarkers for different facilities/types/status
3. **Batch Effect Impact Analysis** - Show how batch corrections affect feature importance
4. **Multi-Group Comparisons** - Heatmaps comparing features across multiple groupings
5. **Interactive HTML Reports** - Publication-ready visualizations with interpretations

---

## Quick Start

### Option 1: Full Pipeline with ML Focus

Use the standard config but ensure ML is enabled:

```bash
cd /usr2/people/macgregor/amplicon/workflow_16s
bash run.sh --config config/config.yaml
```

### Option 2: ML & Reporting Only (Recommended)

Use the specialized ML-only config to skip all preprocessing:

```bash
cd /usr2/people/macgregor/amplicon/workflow_16s
bash run.sh --config config/config_ml_only.yaml
```

This will:
1. Load existing processed data (`final_processed_adata.h5ad`)
2. Run CatBoost with all 3 strategies (baseline, agnostic, group_validated)
3. Generate comprehensive ML visualizations
4. Create interactive HTML report

---

## Output Structure

```
project_01/
├── 04_analysis/
│   └── testing_4/
│       ├── catboost_feature_selection/
│       │   ├── baseline/                    # Strategy 1: Batch included
│       │   │   └── Genus_facility_match/
│       │   │       └── shap/
│       │   │           ├── best_model.cbm
│       │   │           ├── top_features.csv
│       │   │           ├── results_summary.json
│       │   │           └── figs/
│       │   │               ├── shap.summary.bar.20.html
│       │   │               ├── shap.summary.beeswarm.20.html
│       │   │               ├── shap.summary.heatmap.20.html
│       │   │               ├── shap.summary.force.20.html
│       │   │               └── shap.summary.waterfall.20.html
│       │   │
│       │   ├── agnostic/                    # Strategy 2: Batch excluded
│       │   │   └── ... (same structure)
│       │   │
│       │   └── group_validated/             # Strategy 3: Group-aware CV
│       │       └── ... (same structure)
│       │
│       └── ml_visualizations/               # **NEW** Comprehensive comparisons
│           ├── strategy_comparisons/
│           │   ├── strategy_comparison_dashboard_facility_match.html
│           │   └── strategy_comparison_dashboard_facility_distance_km.html
│           │
│           ├── group_fingerprints/
│           │   ├── agnostic/
│           │   │   ├── fingerprint_facility_match_agnostic.html
│           │   │   ├── fingerprint_facility_type_agnostic.html
│           │   │   └── fingerprint_facility_status_agnostic.html
│           │   │
│           │   └── group_validated/
│           │       └── ... (same structure)
│           │
│           ├── multi_group_comparisons/
│           │   ├── multi_group_comparison_agnostic.html
│           │   └── multi_group_comparison_group_validated.html
│           │
│           └── batch_effects/
│               ├── batch_effect_impact_facility_match.html
│               └── batch_effect_impact_facility_distance_km.html
│
└── 05_reports/
    └── ML_Strategy_Comparison_Report.html   # **NEW** Interactive report
```

---

## Key Visualizations

### 1. Strategy Comparison Dashboard

**File:** `ml_visualizations/strategy_comparisons/strategy_comparison_dashboard_<target>.html`

**Purpose:** Compare all 3 batch correction strategies side-by-side

**Panels:**
- **Model Performance:** Accuracy, MCC, ROC-AUC, F1 across strategies
- **Feature Stability:** Jaccard similarity of top features between strategies
- **Batch Effect Impact:** Heatmap showing batch variable importance per strategy
- **Sample Distribution:** Train/test split sizes

**Interpretation:**
- High feature stability (Jaccard > 0.7) → Robust biomarkers across strategies
- Low batch impact in agnostic/group_validated → Batch correction working
- Similar performance across strategies → Batch not confounding biological signal
- Divergent performance → Batch effects present, use group_validated results

### 2. Group Fingerprint Plots

**Files:** `ml_visualizations/group_fingerprints/<strategy>/fingerprint_<grouping>_<strategy>.html`

**Purpose:** Show unique microbial signatures for each grouping variable

**Shows:**
- Top 30 genera ranked by SHAP importance
- Color-coded by importance magnitude
- Separate plots for each grouping (facility_match, facility_type, etc.)

**Interpretation:**
- High importance → Strongly predictive of group membership
- Shared features across groups → Universal biomarkers
- Unique features → Group-specific signatures

### 3. Multi-Group Comparison Heatmap

**File:** `ml_visualizations/multi_group_comparisons/multi_group_comparison_<strategy>.html`

**Purpose:** Compare features across multiple grouping variables simultaneously

**Shows:**
- Rows: Top genera overall
- Columns: Different grouping variables
- Colors: SHAP importance for each genus-grouping combination

**Interpretation:**
- Bright columns → Grouping variable with strong biomarkers
- Bright rows → "Universal" biomarkers important across multiple comparisons
- Dark cells → Genus not important for that grouping
- Patterns → Related groupings share similar biomarkers

### 4. Batch Effect Impact Plots

**File:** `ml_visualizations/batch_effects/batch_effect_impact_<target>.html`

**Purpose:** Visualize how batch variables are handled across strategies

**Shows:**
- Side-by-side top features for baseline, agnostic, group_validated
- Red bars = Batch-related features (sequencing_center, instrument, etc.)
- Blue bars = Biological features (genera)

**Interpretation:**
- Baseline: High red bars expected (batch included as features)
- Agnostic: No red bars (batch completely excluded)
- Group_validated: Few/no red bars (batch accounted for in CV)
- If agnostic/group_validated still show red → Batch confounded with biology

---

## Understanding the 3 Strategies

### Baseline Strategy
**Approach:** Include batch variables as features alongside microbial abundances

**When to use:**
- Exploratory analysis to see if batch matters
- Dataset/source is biologically meaningful (e.g., different studies = different ecosystems)

**Caveats:**
- Can't distinguish batch effect from real biology
- Model may learn "sequencing_center=X predicts facility_match=True" (spurious)

**Interpretation:**
- If batch features are top-ranked → Batch effects present!
- Use agnostic or group_validated for robust biomarkers

### Agnostic Strategy
**Approach:** Completely exclude batch variables from features

**When to use:**
- You want biomarkers robust across datasets
- Batch is purely technical (not biological)

**Caveats:**
- If batch perfectly confounds biology (e.g., all facility samples from 1 study), can't separate
- May reduce performance if batch has legitimate biological signal

**Interpretation:**
- Features here are "batch-agnostic" biomarkers
- Use these for external validation on new datasets

### Group-Validated Strategy
**Approach:** Use GroupKFold cross-validation to prevent batch leakage

**When to use:**
- Gold standard for preventing batch overfitting
- Want conservative, generalizable biomarkers

**How it works:**
- Train/test splits ensure entire batches are held out (no leakage)
- Forces model to generalize across batches

**Interpretation:**
- Features here are "batch-robust" biomarkers
- Lower performance than baseline is GOOD (means baseline was overfitting to batch)
- Use these for final biomarker list

---

## Recommended Workflow

### 1. Initial Run (All Strategies)
```bash
bash run.sh --config config/config_ml_only.yaml
```

### 2. Review Strategy Comparison Dashboard
**Check:** `ml_visualizations/strategy_comparisons/strategy_comparison_dashboard_facility_match.html`

**Questions:**
- Q: Are performance metrics similar across strategies?
  - **Similar** → Batch not confounding, safe to use any strategy
  - **Baseline >> Others** → Batch overfitting, use group_validated

- Q: Is feature stability high (Jaccard > 0.7)?
  - **Yes** → Robust biomarkers, use intersection of all strategies
  - **No** → Strategy-dependent features, investigate why

- Q: Are batch variables important in baseline?
  - **Yes** → Batch effects present, trust agnostic/group_validated only
  - **No** → Batch not an issue, but still prefer group_validated for rigor

### 3. Review Batch Effect Impact
**Check:** `ml_visualizations/batch_effects/batch_effect_impact_facility_match.html`

**Red flags:**
- Red bars in agnostic/group_validated → Batch confounded with biology
- Different top features across strategies → Unstable biomarkers

**Green flags:**
- All blue bars in agnostic/group_validated → Clean biological signal
- Same top features across strategies → Robust biomarkers

### 4. Identify Final Biomarkers
**Use:** `ml_visualizations/multi_group_comparisons/multi_group_comparison_group_validated.html`

**Criteria:**
1. High importance in group_validated strategy (conservative)
2. Present in top 20 of agnostic strategy (batch-robust)
3. Consistent across multiple grouping variables (universal)

**Export:**
```python
# Load top features from group_validated
import pandas as pd
from pathlib import Path

catboost_dir = Path("project_01/04_analysis/testing_4/catboost_feature_selection")

# Facility match biomarkers
facility_match = pd.read_csv(
    catboost_dir / "group_validated/Genus_facility_match/shap/top_features.csv"
).head(20)

# Facility type biomarkers
facility_type = pd.read_csv(
    catboost_dir / "group_validated/Genus_facility_type/shap/top_features.csv"
).head(20)

# Find shared biomarkers (robust across comparisons)
shared = set(facility_match['feature']) & set(facility_type['feature'])
print(f"Shared biomarkers (n={len(shared)}): {shared}")
```

### 5. Biological Interpretation
**For each biomarker:**
1. Check taxonomy (Kingdom → Species)
2. Look up ecology (habitat, metabolism, radiation tolerance)
3. Review SHAP dependency plots for directionality
4. Cross-reference with statistical tests (PERMANOVA, Kruskal-Wallis)

---

## Troubleshooting

### Issue: All strategies show similar performance
**Cause:** Batch effects minimal or biology dominates  
**Action:** ✅ Good! Use any strategy, prefer group_validated for rigor

### Issue: Baseline >> Agnostic/Group_validated
**Cause:** Batch overfitting or batch confounded with biology  
**Action:** 🚨 Trust only agnostic/group_validated results, investigate confounding

### Issue: Low Jaccard similarity (<0.5)
**Cause:** Unstable feature selection or genuine strategy differences  
**Action:** Increase `num_features` in config, run with more iterations

### Issue: No red bars in baseline batch impact plot
**Cause:** Batch variables not in top features or excluded from analysis  
**Action:** Check `helpers.py:find_plottable_metadata()` - ensure batch cols included

### Issue: Empty/missing strategy results
**Cause:** CatBoost crashed due to bugs (numeric exclusion, parameter conflict, etc.)  
**Action:** Check logs, verify fixes from METADATA_FIXES_2026-01-12.md applied

---

## Configuration Reference

### Enable ML & Reporting Only

**File:** `config/config_ml_only.yaml`

**Key settings:**
```yaml
downstream:
  enabled: True
  load_existing_data: True
  data_file: "03_processed_data/final_processed_adata.h5ad"

ml:
  enabled: True
  load_existing: False  # Force fresh run
  tables:
    clr_transformed:
      enabled: True
      levels: ["genus"]
      methods: ["shap"]

machine_learning:
  batch_covariates:
    enabled: True
    covariate_adjustment:
      enabled: True  # Baseline strategy
    stratified_prediction:
      enabled: True  # Two-stage strategy
    comparison:
      enabled: True  # Compare all strategies

reporting:
  enabled: True
  include_sections:
    - 'machine_learning'
    - 'ml_strategy_comparison'
    - 'ml_group_fingerprints'
    - 'ml_batch_effects'
    - 'synthesis'

synthesis:
  enabled: True
  ml_visualizations: True
  strategy_comparison: True
  group_fingerprints: True
  batch_effect_impact: True
```

### Customize ML Targets

**Add/remove targets:**
```yaml
ml_targets:
  - 'facility_match'
  - 'facility_distance_km'
  - 'facility_type'
  - 'contamination_status'
  - 'env_biome'  # Add environmental comparisons
  - 'country'  # Add geographic comparisons
```

### Adjust Feature Selection

**More features, smaller steps:**
```yaml
ml:
  num_features: 1000  # Increase from 500
  step_size: 50  # Decrease from 100 for finer resolution
```

### Change Batch Columns

**Add/remove batch covariates:**
```yaml
machine_learning:
  batch_covariates:
    covariate_columns:
      - 'batch_original'
      - 'study_accession'
      - 'sequencing_platform'
      # Add project-specific batch variables
      - 'extraction_kit'
      - 'sequencing_date'
```

---

## Python API Usage

### Load and Compare Strategies

```python
from pathlib import Path
import pandas as pd
import json

catboost_dir = Path("project_01/04_analysis/testing_4/catboost_feature_selection")
target = "facility_match"

# Load results for each strategy
strategies = {}
for strategy in ['baseline', 'agnostic', 'group_validated']:
    strategy_dir = catboost_dir / strategy / f"Genus_{target}" / "shap"
    
    # Load summary
    with open(strategy_dir.parent / "results_summary.json") as f:
        strategies[strategy] = {
            'summary': json.load(f),
            'features': pd.read_csv(strategy_dir / "top_features.csv")
        }

# Compare performance
for strat, data in strategies.items():
    scores = data['summary']['test_scores']
    print(f"{strat:20} | Accuracy: {scores['accuracy']:.3f} | MCC: {scores['mcc']:.3f}")

# Find consensus biomarkers (top 20 in all strategies)
top_20_sets = [set(data['features'].head(20)['feature']) for data in strategies.values()]
consensus = set.intersection(*top_20_sets)
print(f"\nConsensus biomarkers (n={len(consensus)}): {consensus}")
```

### Generate Custom Visualizations

```python
from workflow_16s.downstream.ml_visualization import (
    create_strategy_comparison_dashboard,
    create_group_fingerprint_comparison,
    create_batch_effect_impact_plot,
    generate_comprehensive_ml_report
)
from pathlib import Path

# Custom output directory
output_dir = Path("custom_ml_viz")

# Generate full report
report = generate_comprehensive_ml_report(
    catboost_dir=Path("project_01/04_analysis/testing_4/catboost_feature_selection"),
    output_dir=output_dir,
    ml_targets=['facility_match', 'facility_type'],
    grouping_variables=['contamination_status', 'env_biome'],
    strategies=['baseline', 'agnostic', 'group_validated']
)

print(f"Generated {len(report['strategy_comparisons'])} strategy dashboards")
print(f"Generated {sum(len(v) for v in report['group_fingerprints'].values())} fingerprint plots")
```

---

## Integration with Statistical Tests

### Cross-Reference with PERMANOVA

ML biomarkers should align with statistically significant taxa:

```python
# Load ML biomarkers
ml_features = pd.read_csv(
    "project_01/04_analysis/testing_4/catboost_feature_selection/"
    "group_validated/Genus_facility_match/shap/top_features.csv"
).head(20)

# Load statistical results
stats_results = pd.read_csv(
    "project_01/04_analysis/testing_4/plots/stats/"
    "significant_taxa_metadata_associations.csv"
)

# Filter for facility_match correlations
facility_stats = stats_results[
    stats_results['metadata'] == 'facility_match'
].sort_values('p_adj')

# Find overlap
ml_genera = set(ml_features['feature'])
stat_genera = set(facility_stats['taxon'])
overlap = ml_genera & stat_genera

print(f"ML features: {len(ml_genera)}")
print(f"Stat significant: {len(stat_genera)}")
print(f"Overlap: {len(overlap)} ({len(overlap)/len(ml_genera)*100:.1f}%)")
```

**Expected:**
- **High overlap (>70%)** → ML and stats agree, high confidence biomarkers
- **Low overlap (<30%)** → May indicate batch effects or non-linear relationships

---

## Publication-Ready Figures

### Export High-Resolution PNGs

```python
import plotly.io as pio
from pathlib import Path

# Load HTML figure
fig = pio.read_html(
    "ml_visualizations/strategy_comparisons/"
    "strategy_comparison_dashboard_facility_match.html"
)

# Export high-res PNG
fig.write_image(
    "figures/Figure_S1_Strategy_Comparison.png",
    width=1800,
    height=1200,
    scale=3  # 300 DPI equivalent
)
```

### Customize for Publication

```python
import plotly.graph_objects as go

# Update layout for publication
fig.update_layout(
    font_family="Arial",
    font_size=14,
    title_font_size=18,
    legend_font_size=12,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Save as vector format
fig.write_image("figures/Figure_1.svg", width=1800, height=1200)
fig.write_image("figures/Figure_1.pdf", width=1800, height=1200)
```

---

## Next Steps

1. **Run ML with fixed bugs:** All 4 critical bugs are now fixed (see METADATA_FIXES_2026-01-12.md)
2. **Generate visualizations:** Use `config_ml_only.yaml` for fast iteration
3. **Review strategy comparison:** Check if batch effects are problematic
4. **Extract final biomarkers:** Use group_validated strategy for robustness
5. **Validate externally:** Test biomarkers on independent datasets

---

## Files Created

1. **`src/workflow_16s/downstream/ml_visualization.py`** - Visualization module
2. **`config/config_ml_only.yaml`** - ML-only configuration
3. **`src/workflow_16s/downstream/steps/synthesis.py`** - Updated with ML viz integration
4. **`ML_VISUALIZATION_GUIDE.md`** - This documentation

---

## Support

For issues or questions:
1. Check logs in `src/logs/`
2. Review METADATA_FIXES_2026-01-12.md for bug fixes
3. Check CATBOOST_STRATEGY_ANALYSIS.md for strategy details
4. Look at successful run: `src/logs/2026-01-06_201412.log`

**Last successful 3-strategy run:** 2026-01-06 (agnostic strategy completed successfully)  
**Last failed run:** 2026-01-11 (all strategies failed due to num_features parameter conflict - NOW FIXED)
