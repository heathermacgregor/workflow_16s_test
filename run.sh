#!/bin/bash
set -euo pipefail  # Strict error handling

# Configuration
readonly ENV_NAME="workflow_16s"
readonly SCRIPT_DIR=$(dirname "$(realpath "$0")")
readonly PYTHON_SCRIPT="${SCRIPT_DIR}/src/run.py"

# Timestamp toggle (default: off)
ENABLE_TIMESTAMPS=false

# Logging Utilities
log() {
    local message=$1
    if $ENABLE_TIMESTAMPS; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${timestamp} - ${message}"
    else
        echo "${message}"
    fi
}

# Parse command-line options
while getopts ":T" opt; do
    case $opt in
        T)
            ENABLE_TIMESTAMPS=false
            ;;
        \?)
            log "[ ✗ ] 🟥 Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))  # Remove processed options

# Dependency Checks
check_conda() {
    log "【 𖦞 】 🟦 Checking Conda availability..."
    if ! command -v conda &>/dev/null; then
        log "【 ✗ 】 🟥 Critical: Conda not found in PATH!"
        exit 1
    fi
}

# Environment Management
validate_environment() {
    log "【 𖦞 】 🟦 Scanning for Conda environments..."
    
    # Check for exact match first
    if conda env list | grep -qw "^${ENV_NAME}"; then
        log "【 ✓ 】 🟩 Found exact match: ${ENV_NAME}"
        return
    fi

    # Fallback to suffix match
    local alt_env=$(conda env list | awk -v pattern="workflow_16s$" \
        '/^[^#]/ && $1 ~ pattern {print $1; exit}')
    
    if [[ -n "${alt_env}" ]]; then
        log "【 ⚠ 】 🟨 Using alternate environment: ${alt_env}"
        ENV_NAME="${alt_env}"
        return
    fi

    log "【 ✗ 】 🟥 No valid Conda environment found matching:"
    log "            - Exact name: ${ENV_NAME}"
    log "            - Name suffix: workflow_16s"
    exit 1
}

activate_environment() {
    log "【 ↺ 】 🟦 Initializing Conda..."
    source "$(conda info --base)/etc/profile.d/conda.sh"

    log "【 ↺ 】 🟦 Activating Conda environment (${ENV_NAME})..."
    if ! conda activate "${ENV_NAME}"; then
        log "【 ✗ 】 🟥 Failed to activate environment (${ENV_NAME})"
        exit 1
    fi
    log "【 ✓ 】 🟩 Environment activated: ${CONDA_DEFAULT_ENV}"
}

# Script Validation
validate_python_script() {
    log "【 ↺ 】 🟦 Verifying workflow script..."
    if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
        log "【 ✗ 】 🟥 Missing Python script: ${PYTHON_SCRIPT}"
        exit 1
    fi
    log "【 ✓ 】 🟩 Script validated: $(realpath "${PYTHON_SCRIPT}")"
}

# Main Execution
main() {
    check_conda
    validate_environment
    activate_environment
    validate_python_script

    log "【 ↺ 】 🟦 Running workflow script..."
    python "${PYTHON_SCRIPT}"

    log "【 ↺ 】 🟦 Deactivating Conda environment (${ENV_NAME})..."
    conda deactivate
    log "【 ✓ 】 🟩 Workflow completed successfully"
}

# Execute main function
main "$@"
