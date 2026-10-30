#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Determine script directory (Adapter_BBCH) and project root (one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to python in your virtual environment (adjust if your venv has a different name/path)
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"

# Check if the virtual environment Python exists
if [[ ! -x "${VENV_PYTHON}" ]]; then
    # Fallback check for .venv if venv doesn't exist
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
    else
        echo "Error: Virtual environment python not found at ${VENV_PYTHON} or ${PROJECT_ROOT}/.venv/bin/python" >&2
        exit 1
    fi
fi

# Target directory containing YAML configs
CONFIG_DIR="${SCRIPT_DIR}/config"

# Verify config directory exists
if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Error: Config directory not found at ${CONFIG_DIR}" >&2
    exit 1
fi

# Ensure running from the project root
cd "${PROJECT_ROOT}"

echo "Starting training orchestrator from project root: ${PROJECT_ROOT}"
echo "Using interpreter: ${VENV_PYTHON}"

# Enable nullglob so the loop does not run if no matching files are found
shopt -s nullglob
config_files=("${CONFIG_DIR}"/*.yaml "${CONFIG_DIR}"/*.yml)
shopt -u nullglob

if [[ ${#config_files[@]} -eq 0 ]]; then
    echo "No YAML files found in ${CONFIG_DIR}."
    exit 0
fi

# Iterate over each config file
for config_path in "${config_files[@]}"; do
    # Convert absolute path to a relative path starting with Adapter_BBCH/config/...
    rel_config_path="Adapter_BBCH/config/$(basename "${config_path}")"

    echo "=================================================="
    echo "Running configuration: ${rel_config_path}"
    echo "=================================================="

    "${VENV_PYTHON}" Adapter_BBCH/run_adapter_training.py --config "${rel_config_path}"
done

echo "All training configurations completed successfully."