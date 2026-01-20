#!/bin/bash
set -euo pipefail  # Exit on error, unset variables, and pipe failures

# Get absolute path of the script's directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Conda Installation Checks
echo "🔍 Checking Conda installation..."
if ! command -v conda &>/dev/null; then
    echo "❌ Error: Conda not found. Please install Conda and ensure it's in your PATH."
    exit 1
fi

if ! conda --version &>/dev/null; then
    echo "❌ Error: Conda installation appears corrupted. Please verify your installation."
    exit 1
fi

# Conda Initialization
echo "🔄 Initializing Conda..."
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Ensure base environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️ Warning: No Conda environment active"
    echo "🔄 Activating base environment..."
    conda activate base
    echo "✅ Activated base environment"
fi

# Mamba Setup
echo "🔍 Checking Mamba installation..."
if ! command -v mamba &>/dev/null; then
    echo "🔄 Installing Mamba..."
    conda install -y -c conda-forge mamba
fi

# QIIME2 Environment Setup
QIIME_ENV="qiime2-amplicon-2024.10"
YAML_URL="https://data.qiime2.org/distro/amplicon/${QIIME_ENV}-py310-$(uname -s | tr '[:upper:]' '[:lower:]')-conda.yml"

echo "🔍 Checking for '${QIIME_ENV}' environment..."
if ! conda env list | grep -q "${QIIME_ENV}"; then
    echo "🔄 Creating QIIME2 environment..."
    
    # Handle different OS cases
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) echo "❌ Error: Unsupported operating system"; exit 1 ;;
    esac

    if ! mamba env create -n "${QIIME_ENV}" --file "${YAML_URL}"; then
        echo "⚠️ Remote YAML failed, trying local download..."
        YAML_FILE=$(basename "${YAML_URL}")
        
        # Download YAML with cleanup trap
        trap "rm -f ${YAML_FILE}" EXIT
        echo "🔄 Downloading YAML file..."
        curl -LO "${YAML_URL}" || wget "${YAML_URL}"
        
        if [[ ! -f "${YAML_FILE}" ]]; then
            echo "❌ Error: Failed to download YAML file"
            exit 1
        fi
        
        # Attempt creation with mamba/conda
        if ! mamba env create -n "${QIIME_ENV}" --file "${YAML_FILE}"; then
            echo "⚠️ Mamba failed, trying Conda..."
            conda env create -n "${QIIME_ENV}" --file "${YAML_FILE}" || {
                echo "❌ Error: Failed to create environment"; exit 1
            }
        fi
    fi
    echo "✅ Created QIIME2 environment"
else
    echo "✅ Found existing QIIME2 environment"
fi

# SILVA Classifier Setup
CLASSIFIER_DIR="${SCRIPT_DIR}/references/classifier/silva-138-99-515-806"
mkdir -p "${CLASSIFIER_DIR}"

declare -a SILVA_FILES=(
    "silva-138-99-seqs-515-806.qza"
    "silva-138-99-tax-515-806.qza"
)
QIIME_BASE_URL="https://data.qiime2.org/2024.10/common"
CLASSIFIER_FILE="${CLASSIFIER_DIR}/silva-138-99-515-806-classifier.qza"

echo "🔍 Verifying SILVA classifier components..."
for FILE in "${SILVA_FILES[@]}"; do
    FILE_PATH="${CLASSIFIER_DIR}/${FILE}"
    if [[ ! -f "${FILE_PATH}" ]]; then
        echo "🔄 Downloading ${FILE}..."
        curl -#L "${QIIME_BASE_URL}/${FILE}" -o "${FILE_PATH}" || 
        wget --progress=bar "${QIIME_BASE_URL}/${FILE}" -O "${FILE_PATH}" || {
            echo "❌ Download failed"; exit 1
        }
    fi
done

echo "🔍 Checking classifier artifact..."
if [[ ! -f "${CLASSIFIER_FILE}" ]]; then
    echo "🔄 Attempting Zenodo download..."
    ZENODO_URL="https://zenodo.org/records/15299267/files/silva-138-99-515-806-classifier.qza"
    
    if ! (curl -#L "${ZENODO_URL}" -o "${CLASSIFIER_FILE}" || 
          wget --progress=bar "${ZENODO_URL}" -O "${CLASSIFIER_FILE}"); then
        echo "⚠️ Download failed, generating classifier..."
        conda activate "${QIIME_ENV}"
        qiime feature-classifier fit-classifier-naive-bayes \
            --i-reference-reads "${CLASSIFIER_DIR}/silva-138-99-seqs-515-806.qza" \
            --i-reference-taxonomy "${CLASSIFIER_DIR}/silva-138-99-tax-515-806.qza" \
            --o-classifier "${CLASSIFIER_FILE}"
        conda deactivate
    fi
    echo "✅ Classifier setup complete"
fi

# Workflow Environment
WORKFLOW_ENV="workflow_16s"
ENV_YAML="${SCRIPT_DIR}/references/conda_envs/workflow_16s.yml"

echo "🔍 Checking workflow environment..."
EXISTING_ENV=$(conda env list | awk -v env="${WORKFLOW_ENV}" '$1 == env {print $1}')

if [[ -n "${EXISTING_ENV}" ]]; then
    echo "✅ Found existing workflow environment"
else
    echo "🔄 Creating workflow environment..."
    mamba env create -n "${WORKFLOW_ENV}" --file "${ENV_YAML}" || {
        echo "❌ Error: Failed to create workflow environment"; exit 1
    }
    echo "✅ Created workflow environment"
fi

# Final Checks
echo "🔍 Validating environment setup..."
conda activate "${WORKFLOW_ENV}"

if ! command -v fastqc &>/dev/null; then
    echo "🔄 Installing FastQC..."
    mamba install -y -c bioconda fastqc || {
        echo "❌ Error: FastQC installation failed"; exit 1
    }
fi

conda deactivate
echo "✅ Environment setup completed successfully"
