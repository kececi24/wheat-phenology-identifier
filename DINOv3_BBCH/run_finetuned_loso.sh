#!/usr/bin/env bash
set -Eeuo pipefail

# YAML-driven fine-tuning -> cache -> temporal LOSO experiments.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/.env"

exec "${PYTHON:-python3}" "${SCRIPT_DIR}/finetuned_loso_runner.py" "$@"
