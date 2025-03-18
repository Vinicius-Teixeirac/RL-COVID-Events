#!/bin/bash
set -euo pipefail

# (Optional) load config defaults from a separate file.
# source "./lib/config.sh"

# Loads configuration from .env file if available.
CONFIG_FILE="${CONFIG_FILE:-./.env}"
if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

# Sets configurable paths with defaults.
DATASET_DIR="${DATASET_DIR:-DataPreparation/UsageDatasets}"
OUTPUT_DIR="${OUTPUT_DIR:-DatasetEventFeatures}"
LOG_DIR="${LOG_DIR:-logs}"
CRITDD_RESULTS="${CRITDD_RESULTS:-CritddResults}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-40}"

# Exports them for use in parallel subshells.
export DATASET_DIR OUTPUT_DIR LOG_DIR CRITDD_RESULTS MAX_PARALLEL_JOBS

# Creates necessary directories.
mkdir -p "$LOG_DIR" "$CRITDD_RESULTS"

# Source our modular functions.
source "./Lib/logging.sh"
source "./Lib/dependencies.sh"
source "./Lib/checking.sh"
source "./Lib/preprocessing.sh"
source "./Lib/hyperparams.sh"
source "./Lib/graph_generation.sh"
source "./Lib/evaluation.sh"
source "./Lib/process_dataset.sh"

# Centralized summary log and lock file.
MAIN_LOG="$LOG_DIR/script_summary.log"
PREPROC_LOG="$LOG_DIR/pre_processing.log"
LOCK_FILE="/tmp/script_log.lock"

# Catch errors.
trap 'log_msg "ERROR" "Error: Command '\''$BASH_COMMAND'\'' failed on line ${LINENO}. Exiting..."; exit 1' ERR

# Check required dependencies.
check_dependencies

# Initialize PYTHONPATH safely.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Run preprocessing.
run_preprocessing

# Gather dataset files after preprocessing.
shopt -s nullglob
datasets=( "$DATASET_DIR"/*.pkl )
shopt -u nullglob

if [ ${#datasets[@]} -eq 0 ]; then
  log_msg "ERROR" "No .pkl files found in $DATASET_DIR after preprocessing."
  exit 1
fi

# Process all datasets in parallel.
printf "%s\n" "${datasets[@]}" | parallel -j "$MAX_PARALLEL_JOBS" process_dataset {}

log_msg "INFO" "All processes completed successfully!"