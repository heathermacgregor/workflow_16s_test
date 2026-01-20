# Installation Guide for workflow_16s

## Package Installation

### Option 1: Development Installation (Recommended for Contributors)

```bash
# Clone the repository
git clone https://github.com/heathermacgregor/workflow_16s.git
cd workflow_16s

# Install in editable mode with all dependencies
pip install -e ".[dev,geo]"
```

### Option 2: User Installation from PyPI (when published)

```bash
pip install workflow-16s
```

### Option 3: Direct from GitHub

```bash
pip install git+https://github.com/heathermacgregor/workflow_16s.git
```

## QIIME 2 Setup (Required for Upstream Pipeline)

QIIME 2 must be installed separately via conda due to complex dependencies:

```bash
# Run the included setup script
bash setup.sh

# Or manually:
conda create -n qiime2-amplicon-2024.10 \
    -c conda-forge -c bioconda -c defaults \
    qiime2 q2cli q2-dada2 q2-feature-table \
    cutadapt seqkit fastqc multiqc
```

## Verifying Installation

```python
import workflow_16s
print(f"Installed version: {workflow_16s.__version__}")

# Test upstream module
from workflow_16s.upstream import UpstreamWorkflow

# Test downstream module  
from workflow_16s.downstream import DownstreamWorkflow
```

## Dependencies

### Core Dependencies (automatically installed)
- numpy, pandas, scipy
- scikit-learn
- scanpy, anndata
- plotly
- catboost
- requests, pyyaml, tqdm, rich

### Optional Dependencies

**Geospatial analysis** (`pip install workflow-16s[geo]`):
- geopandas, shapely, folium

**Development tools** (`pip install workflow-16s[dev]`):
- pytest, pytest-cov
- black, flake8, mypy

### System Requirements
- Python 3.9+
- For QIIME 2: Conda/Mamba package manager
- Recommended: 16+ GB RAM for large datasets

## Building from Source

```bash
# Install build tools
pip install build twine

# Build distribution packages
python -m build

# This creates:
# - dist/workflow_16s-2.0.0-py3-none-any.whl
# - dist/workflow-16s-2.0.0.tar.gz
```

## Publishing to PyPI (Maintainers Only)

```bash
# Test on TestPyPI first
python -m twine upload --repository testpypi dist/*

# Then publish to PyPI
python -m twine upload dist/*
```

## Troubleshooting

**Import errors**: Ensure you're in the correct conda environment
```bash
conda activate workflow_16s  # or qiime2-amplicon-2024.10
```

**QIIME 2 not found**: The package works without QIIME 2, but upstream processing requires it
```bash
# Verify QIIME 2 installation
qiime --version
```

**Permission errors on cluster**: Use `--user` flag
```bash
pip install --user -e .
```
