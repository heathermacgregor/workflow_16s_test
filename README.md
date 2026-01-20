workflow_16s
============

A modular, extensible **microbial community analysis pipeline** for 16S rRNA amplicon sequencing data. Designed for reproducible research.

> See an example output file [here](https://heathermacgregor.github.io/workflow_16s_sample/).

---

## Purpose

This repository provides:

1. A configurable **upstream pipeline**:
    1. Retrieval of metadata and raw sequencing data
        1. From the European Nucleotide Archive (ENA) database
        1. Locally
    1. Sample selection based on target region
    1. (Optional) Prediction of PCR primers and target regions
    1. (Optional) Validation of 16S sequences
    1. (Optional) Sequence quality assessment ([SeqKit](https://bioinf.shenwei.me/seqkit/), [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/))
    1. (Optional) Sequence trimming using [CutAdapt](https://cutadapt.readthedocs.io/en/stable/)
    1. [QIIME 2](https://qiime2.org) workflow performed on each dataset
        1. Sequence trimming
        1. Denoising
        1. Quality control
        1. Taxonomic assignment
    1. (Optional) Cleanup of raw data
1. Advanced **downstream analysis**, focusing on the relationship between a group column and the microbial community composition and/or individual taxonomic features:
    1. Alpha diversity metrics
    1. Beta diversity and ordination
    1. Statistical testing
    1. Feature selection ([CatBoost](https://catboost.ai))
    1. Functional prediction ([FAPROTAX](https://pages.uoregon.edu/slouca/LoucaLab/archive/FAPROTAX/lib/php/index.php))
    1. Geospatial mapping
1. Generation of a comprehensive HTML report with interactive visualizations

Originally used for ["Article Title Placeholder"](https://doi.org), the pipeline is fully adaptable to other analyses. It will continue to be updated in order to become more easily applicable to different use cases.

---

## Repository Structure


<pre> 
  workflow_16s/ 
  ├── references/ 
  │ ├── classifier/ 
  │ │ └── silva-138-99-515-806/ 
  │ ├── conda_envs/
  │ ├── manual_metadata/                   # Default directory for manual metadata
  │ ├── config.yaml                        # Default config file
  │ ├── datasets.tsv                       # Default datasets TSV
  │ └── datasets.txt                       # Default datasets TXT
  ├── src/ 
  │ ├── workflow_16s/
  │ │ ├── amplicon_data/                   # Downstream (cross-dataset) analysis
  │ │ ├── ena/                             # Interactions with the ENA API
  │ │ ├── figures/                         # Figure generation
  │ │ ├── function/                        # Functional assignment
  │ │ ├── metadata/                        # Metadata handling
  │ │ ├── models/                          # Machine learning models
  │ │ ├── qiime/                           # Interactions with the QIIME 2 API
  │ │ ├── sequences/                       # Sequence data analysis
  │ │ ├── stats/                           # Statistical analysis
  │ │ ├── utils/                           # Utilities
  │ │ ├── __init__.py 
  │ │ ├── config.py 
  │ │ ├── constants.py                     # Constants and default values 
  │ │ └── logger.py 
  │ ├── __init__.py 
  │ └── run.py 
  ├── README.md                            # YOU ARE HERE
  ├── run.sh                               # Executes the full workflow
  └── setup.sh                             # Set up a conda environment for the workflow
  </pre>

---

## Installation & Setup

```bash
git clone https://github.com/heathermacgregor/workflow_16s.git
cd workflow_16s
bash setup.sh
```

---

## Usage

After creating custom input files ([`config.yaml`](#configuration), [`datasets.txt`](#datasets-txt), [`datasets.tsv`](#datasets-tsv)), run: 

```bash
bash run.sh [--config PATH_TO_CUSTOM_CONFIG_YAML]
```

**`[!] IMPORTANT:`** Make sure you edit `ena_email` in the configuration file so you can access the ENA API.

---
## Input Files

### Configuration

> See a breakdown of the default example [here](https://github.com/heathermacgregor/workflow_16s/blob/main/info/config.md).

**`[!] IMPORTANT:`** Make sure that your config file contains accurate paths for `dataset_list`, `dataset_info`, `manual_metadata_dir`, and `project_dir`.

#### Key YAML Settings

- **Hardware**: CPU thread limits (`cpu.limit`).
- **Paths**: Dataset list, metadata, manual metadata directory, project output path.
- **Credentials**: ENA API email for obtaining sequence metadata or download authorization.
- **Execution flags**: Toggle `upstream` and `downstream` processing; optionally enable subset analysis.

#### Core Pipeline Modules

#### Upstream
- PCR primer control, subfragment targeting, BLAST database location.
- Tools like `fastqc`, `seqkit`, `validate_16s`, and `cutadapt` for preprocessing.
- QIIME2 settings: trimming, denoising (e.g. DADA2), taxonomy classification, filtering, and cleanup.

##### Downstream
- **Metadata grouping**: Define sample groups using columns (e.g. `nuclear_contamination_status`).
- **Feature preprocessing**: Support for filtering, normalization, CLR transformation, and presence/absence conversion.
- **Statistical testing**: Mann‑Whitney U, Kruskal‑Wallis, Fisher’s exact, and t‑tests across multiple feature representations.
- **Diversity analyses**:
  - *Alpha*: Shannon, Simpson, Pielou’s evenness, Heip, etc., with visualization and optional correlation.
  - *Beta/Ordination*: PCoA, PCA, t-SNE, UMAP across metrics like Bray‑Curtis, Jaccard, Euclidean.
- **Functional prediction**: FaProTax integration for ecological role inference.
- **Mapping & visualization**: Geospatial sample maps, violin plots, feature maps colored by metadata.
- **Machine learning**: Feature selection via RFE, chi‑squared, LASSO, SHAP; permutation importance; performance tracking.

### Datasets (TXT)
A TXT file containing a list of datasets. ENA datasets should be listed by their project accession. Local datasets should be listed by the same name used for the subfolder(s) where their metadata and sequencing data are stored.
```
PRJDB15313
PRJDB7915
PRJDB7978
```

### Datasets (TSV)
A TSV file containing pertinent information about datasets. 

**`[!] IMPORTANT:`** ENA datasets do not always clarify the target gene, target region, the PCR primers used, and other experimental details. Although the workflow can attempt to predict the details necessary for conversion of raw data to feature tables, it is highly recommended to manually collect metadata (e.g. from associated publications).

| dataset_id     | metadata_complete | dataset_type | ena_project_accession | ena_project_description | ... |
|----------------|-------------------|--------------|-----------------------|-------------------------|-----|
| ENA_PRJDB15313 | TRUE              | ENA          | PRJDB15313            | Boston, MA              | ... |
| ENA_PRJDB7915  | TRUE              | ENA          | PRJDB7915             | Fukushima (river soils) | ... |
| ENA_PRJDB7978  | TRUE              | ENA          | PRJDB7978             | Fukushima (soil)        | ... |

---

## To-Do
- [ ] Add to HTML report sections on the original datasets, linking to ENA BioProjects when relevant.
- [ ] Add **batch correction methods** that are appropriate for microbial community data (e.g. [ConQuR](https://github.com/wdl2459/ConQuR)).
- [ ] Add support for ASV mode (only genus mode is currently available).
