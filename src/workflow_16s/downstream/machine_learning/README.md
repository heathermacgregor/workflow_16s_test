Here is a comprehensive `README.md` for the **Machine Learning Module**. You can place this file in `src/workflow_16s/downstream/machine_learning/README.md`.

---

# Machine Learning & Feature Selection Module

## 🧠 Overview

This module provides a robust, production-ready Machine Learning pipeline designed specifically for high-dimensional microbiome data. It integrates **CatBoost** and **Random Forest** to identify microbial signatures that predict metadata targets (e.g., "Facility Match", "pH", "Treatment Group").

**Key Capabilities:**

* **Dual-Engine Support:** Runs Gradient Boosting (CatBoost) and Bagging (Random Forest) in parallel.
* **Batch Effect Management:** Implements 4 distinct strategies to handle technical noise (e.g., Sequencing Center, Run ID).
* **Strict Targeting:** Focuses computational power only on validated, relevant metadata columns.
* **SHAP-Based Feature Selection:** Reduces tens of thousands of ASVs/Genera to a stable core signature.

---

## 🏗️ Architecture

The pipeline operates in two phases: **Feature Selection (FS)** and **Model Evaluation**.

### 1. Feature Selection (FS)

Before training complex models, the pipeline uses **CatBoost with SHAP values** to identify the top predictive features.

* **Input:** CLR-transformed counts (e.g., Genus level).
* **Process:** Trains a quick model, calculates SHAP importance for every feature, and keeps the top  (default: 20-50).
* **Output:** A reduced feature set used for the final models.

### 2. Batch Control Strategies

Microbiome data is notoriously sensitive to batch effects. This module allows you to run up to 4 concurrent strategies to assess model robustness:

| Strategy | Description | Best For |
| --- | --- | --- |
| **Baseline** | Standard ML. Ignores batch information. | Initial scouting. |
| **Agnostic** | **"The Control".** Removes batch columns from training data entirely. | Verifying signal isn't just batch ID. |
| **Batch-Adjusted** | Includes Batch ID as a covariate (One-Hot Encoded) in the model. | allowing the model to "learn" the noise. |
| **Group-Validated** | **"The Gold Standard".** Uses `GroupKFold` cross-validation, ensuring samples from the same batch never appear in both train and test sets. | Testing true generalizability. |

---

## ⚙️ Configuration

Configure the module in `config/config_ml_only.yaml`.

```yaml
ml:
  enabled: true
  strict_targets: true   # If true, only analyzes specific columns
  targets:               # The specific columns to predict
    - "facility_match"
    - "facility_distance_km"
  
  # Grid Settings (Loop through these combinations)
  grid_settings:
    levels: ["Genus", "Family"]  # Taxonomic levels
    transformations: ["clr"]     # Normalization (clr, binary, log1p)
    fs_strategies:               # Which batch strategies to run during Feature Selection
      - "baseline"
      - "group_validated"

  # Model Hyperparameters
  models:
    enable_catboost: true
    enable_random_forest: true
    n_estimators: 200
    max_depth: 10        # Keep <= 10 for CatBoost on CPU
    
  # Batch Correction Settings
  batch_covariates:
    enabled: true
    covariate_columns: ["sequencing_center", "run_id"]
    covariate_adjustment:
      one_hot_encode: true

```

---

## 🚀 Usage

This module is typically invoked automatically by the main analysis pipeline (`steps/analysis.py`), but it is modular enough to be understood in isolation.

### The Workflow Loop

1. **Data Prep:** `analysis.py` aggregates data (e.g., to Genus) and applies CLR transform.
2. **Selection:** `run_catboost_selection` finds the top features.
3. **Training:** `run_machine_learning_analysis` trains full models using the selected features and the requested batch strategies.
4. **Visualization:** Generates performance reports and SHAP plots.

### Output Directory Structure

Results are saved in `outputs/ml_plots/`:

```text
outputs/ml_plots/
├── Genus/
│   ├── clr/
│   │   ├── facility_match/         # Target Name
│   │   │   ├── catboost_baseline/  # Strategy + Algorithm
│   │   │   │   ├── shap_beeswarm.png
│   │   │   │   ├── confusion_matrix.png
│   │   │   │   └── performance_metrics.csv
│   │   │   ├── rf_group_validated/
│   │   │   └── ...

```

---

## 📂 File Descriptions

* **`main.py`**
* **The Orchestrator.** Contains `run_machine_learning_analysis` and `run_catboost_selection`.
* Handles data preparation, target validation, and looping through strategies.


* **`feature_selection.py`**
* **The Filter.** Runs a lightweight CatBoost regressor/classifier to calculate SHAP values.
* Returns a sorted list of the most important taxa.


* **`batch_control.py`**
* **The Engine.** Contains `run_ml_with_batch_control`.
* Manages the cross-validation logic (StratifiedKFold vs GroupKFold) and covariate injection.
* Handles the actual training of CatBoost and sklearn Random Forest models.


* **`visualization.py`**
* **The Artist.** Generates publication-ready plots:
* **SHAP Beeswarm:** Direction and magnitude of feature impact.
* **Correlation Matrix:** How features relate to each other.
* **Performance Bar Charts:** Compare Accuracy/R2 across batch strategies.
