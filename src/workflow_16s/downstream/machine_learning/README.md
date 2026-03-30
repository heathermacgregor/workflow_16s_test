---

# 🧬 16S Machine Learning Discovery & Forensic Suite

## 🧠 Overview

This module provides a high-integrity, production-ready Machine Learning pipeline designed specifically for identifying robust microbial biomarkers in 16S data. Unlike standard pipelines that optimize for raw accuracy, this suite optimizes for **Scientific Generalizability**—ensuring discovered taxa are biological signals rather than technical artifacts.

### The "Forensic Grade" Difference:

* **Study-Agnostic Discovery:** Prioritizes microbes that persist across independent labs and sequencing centers.
* **Consensus Weighting:** Adjusts feature importance based on cross-study appearance frequency.
* **Defense-in-Depth Validation:** Every discovery must pass a 4-tier audit (Eligibility, Overfitting, Significance, and Stability).
* **Certification Dashboard:** Automatically generates a PASS/FAIL "Executive Summary" for stakeholders.

---

## 🏗️ The 4-Tier Validation Stack

The pipeline implements a hierarchical audit trail to ensure every reported biomarker is mathematically and biologically defensible.

| Tier | Module | Goal | Logic |
| --- | --- | --- | --- |
| **1. Eligibility** | `StudyEligibilityManager` | **Pre-flight Audit** | Prunes studies with  or zero target variance to prevent injecting noise. |
| **2. Internal Audit** | `run_comprehensive_validation` | **Overfitting Guardrail** | Uses Nested CV to measure the "Gap" between training and validation scores. |
| **3. Significance** | `run_shuffle_baseline` | **Chance Elimination** | Calculates an empirical -value by comparing results to 100+ random permutations. |
| **4. Consensus** | `perform_meta_analysis` | **Universal Mapping** | Identifies "Golden Biomarkers" that appear as top predictors in  of independent studies. |

---

## 🚀 Discovery Workflow

The discovery process is orchestrated by `workflows/feature_selection.py`:

1. **Sanitization**: Applies **Batch-Centered CLR** (Z-Centering) to remove constant technical offsets between sequencing facilities.
2. **Agnostic Search**: Trains a model blinded to "Batch IDs" to ensure it learns microbial rules, not lab metadata.
3. **Cross-Validation (LOPOCV)**: Implements "Leave-One-Project-Out" CV—the ultimate test of a model's ability to predict a target in a brand-new, unseen facility.
4. **Robustness Weighting**: Final SHAP importance scores are weighted by **Meta-Analysis Frequency**, surfacing biomarkers that are both powerful and consistent.

---

## ⚙️ Configuration

Control the discovery engine in `config/config_ml_only.yaml`.

```yaml
ml:
  enabled: true
  eligibility_mode: "filter"   # Prune underpowered studies before training
  targets: 
    - "facility_match"         # Forensic Target
    - "ph_h2o"                 # Ecological Baseline
    
  grid_settings:
    levels: ["Genus"]
    fs_strategies: 
      - "agnostic"             # Blinded to Batch ID
      - "lopocv"               # Leave-One-Project-Out (The Gold Standard)
      - "meta_analysis"        # Cross-study consensus

```

---

## 📂 Architecture & Artifacts

### Directory Structure

```text
outputs/machine_learning/
├── Discovery_Executive_Summary.html    <-- The Certification Dashboard
├── agnostic/
│   └── Genus_facility_match/
│       ├── discovery_audit_plot.html   <-- 4-Panel diagnostic report
│       ├── batch_dependency_donut.html <-- Tech vs. Bio signal ratio
│       └── robustness_weighted_features.csv
├── meta_analysis/
│   └── Genus_facility_match/
│       └── biomarker_stability_heatmap.html
├── environmental_baseline/              <-- SoilGrids baseline suite
└── ...

```

### Module Descriptions

* **`workflows/feature_selection.py`**: The Orchestrator. Manages the loop from data prep to final certification.
* **`validation/`**: The Auditor. Houses the eligibility guards, shuffle tests, and quality certification gates.
* **`meta_analysis.py`**: The Consensus Engine. Compares features across independent studies to identify universal biomarkers.
* **`visualization/`**: The Storyteller. Generates the diagnostic audits, stability heatmaps, and batch-dependency plots.

---

## 🛡️ Forensic Integrity Checks

A discovery is considered **"Certified"** only if:

1. **Biological Signal**: MCC Score  (Agnostic).
2. **Statistical Significance**: Permutation -value .
3. **Generalization**: Overfitting Gap .
4. **Consistency**: At least 3 biomarkers verified across independent cohorts.

---