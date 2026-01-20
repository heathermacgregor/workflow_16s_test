# Example 1. Nuclear Contamination

This repository includes a example [YAML configuration file](https://github.com/heathermacgregor/workflow_16s/blob/main/references/config.yaml) for a bioinformatics pipeline analyzing microbial communities, with a focus on identifying signatures of **nuclear contamination**. The config enables modular control over upstream (preprocessing, QC) and downstream (statistical and ML) steps.

---


## Hardware and Parallelism

```yaml
execution:
  threads: 8   # Number of parallel workers/cores for downstream ingestion and other multithreaded tasks
  cpu_limit: 8 # Global CPU limit for all operations
cpu:
  limit: 8     # Global CPU limit for all operations
```

- `execution.threads` (preferred): Sets the number of parallel workers/cores for downstream ingestion and other parallel tasks.
- `cpu.limit` or `execution.cpu_limit`: Used as fallback if `execution.threads` is not set.
- If none are set, defaults to 8. Max is capped at system CPU count (64).

**Example:**

```yaml
execution:
  threads: 16
```

This will use up to 16 parallel workers for downstream ingestion.

---

## Paths

```yaml
dataset_list: "./datasets.txt"
dataset_info: "./datasets.tsv"
manual_metadata_dir: "../../../manual_metadata"
project_dir: "../../test"
```

- `dataset_list`: Text file with datasets to analyze.
- `dataset_info`: Tab-separated metadata file.
- `manual_metadata_dir`: Folder with manually curated metadata.
- `project_dir`: Output location for all results.

---

## Credentials

```yaml
ena_email: "your_email@example.com"
```

Used to authenticate with the ENA API for metadata retrieval or downloads.

---

## Execution Toggles

```yaml
upstream:
  enabled: False

downstream:
  enabled: True
  find_subsets: False
```

- Enable or disable different parts of the pipeline.
- Upstream handles raw read QC and preprocessing.
- Downstream handles metadata integration, stats, ML, and visualization.

---

## Upstream Processing

Handles data cleaning and 16S rRNA gene analysis using tools like QIIME2 and Cutadapt.

### PCR and BLAST

```yaml
pcr_primers_mode: "manual"
target_subfragment_mode: "any"
blast_db_dir: "./blast/silva_16s/SILVA_16S_db"
```

### Quality Control & Trimming

Includes tools:
- `fastqc`
- `seqkit`
- `validate_16s`
- `cutadapt`

### QIIME2 Integration

Full control over trimming, denoising (e.g., DADA2), taxonomy classification, and filtering.

---

## Downstream Analysis

### Metadata Setup

```yaml
group_column: "nuclear_contamination_status"
group_column_type: "bool"
metadata_id_column: "#sampleid"
dataset_column: "dataset_name"
```

Supports flexible groupings using metadata fields.

---

### Feature Preprocessing

```yaml
features:
  filter: True
  normalize: True
  clr_transform: True
  presence_absence: True
```

Supports multiple transformations on the feature table before downstream analysis.

---

### Nearby Facility Analysis

```yaml
nfc_facilities:
  enabled: True
  databases:
    - name: "GEM"
    - name: "NFCIS"
  max_distance_km: 50
```

Analyzes proximity to nuclear facilities using geospatial data.

---

### Statistical Tests

Supports testing across multiple feature representations:

- **Raw**: Mann-Whitney U, Kruskal-Wallis
- **Presence/Absence**: Fisher's exact
- **Normalized, Filtered, CLR**: t-tests, MWU, Kruskal-Wallis

---

### Alpha Diversity

```yaml
alpha_diversity:
  enabled: True
  metrics:
    - 'shannon'
    - 'observed_features'
    - 'simpson'
    - 'pielou_evenness'
    - 'heip_evenness'
```

Supports visualization, statistical annotation, and optional correlation analysis with metadata.

---

### Beta Diversity & Ordination

Multiple methods (`pca`, `pcoa`, `tsne`, `umap`) and distance metrics (`braycurtis`, `euclidean`, `jaccard`) are supported across feature representations and taxonomic levels.

---

### Functional Prediction

```yaml
faprotax:
  enabled: True
```

Runs FaProTax to infer potential ecological functions from taxonomy.

---

### Mapping & Visualization

```yaml
maps:
  enabled: True
  color_columns:
    - "dataset_name"
    - "nuclear_contamination_status"
```

Maps samples with color overlays using selected metadata columns.

---

### Violin Plots & Feature Maps

```yaml
violin_plots:
  enabled: True
  n: 50
```

Visualize the most important features per group or dataset.

---

### Machine Learning

```yaml
ml:
  enabled: True
  n_threads: 32
  num_features: 500
  step_size: 100
```

#### Methods Supported:

- `rfe`
- `select_k_best`
- `chi_squared`
- `lasso`
- `shap`

Each method can be applied across raw, filtered, normalized, CLR-transformed, and presence/absence data.

---

## Use Case: Nuclear Contamination

This config is optimized for evaluating the impact of nuclear contamination on microbial communities using:

- Group-based comparisons
- Facility proximity analyses
- Geospatial visualizations
- Machine learning feature selection

---

## Getting Started

1. Update paths to match your project structure.
2. Toggle `upstream` or `downstream` as needed.
3. Make sure your environment has all required tools: `QIIME2`, `FaProTax`, `cutadapt`, `scikit-learn`, etc.
4. Launch your pipeline script with the config.

---

## Questions?

Feel free to [open an issue](https://github.com/heathermacgregor/workflow_16s/issues) if you need help adapting the config to your dataset or analysis goals.
