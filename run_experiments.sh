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

###############################################################################
# Granular Thresholds
#   - Group 1: <= 10 MB  => 100 jobs
#   - Group 2: <= 20 MB  => 80 jobs
#   - Group 3: <= 30 MB  => 40 jobs
#   - Group 4: <= 40 MB  => 25 jobs
#   - Group 5: <= 50 MB  => 15 jobs
#   - Group 6:  > 50 MB  =>  5 jobs
###############################################################################

threshold1=$((10 * 1024 * 1024))  # 10 MB
threshold2=$((20 * 1024 * 1024))  # 20 MB
threshold3=$((30 * 1024 * 1024))  # 30 MB
threshold4=$((40 * 1024 * 1024))  # 40 MB
threshold5=$((50 * 1024 * 1024))  # 50 MB

# Initialize arrays for each group.
group1=()  # <= threshold1 => 100 jobs
group2=()  # > threshold1 && <= threshold2 => 80 jobs
group3=()  # > threshold2 && <= threshold3 => 40 jobs
group4=()  # > threshold3 && <= threshold4 => 25 jobs
group5=()  # > threshold4 && <= threshold5 => 15 jobs
group6=()  # > threshold5 => 5 jobs

# Categorize datasets by file size.
for dataset in "${datasets[@]}"; do
  size=$(stat -c%s "$dataset")
  
  if   [ "$size" -le "$threshold1" ]; then
    group1+=("$dataset")
  elif [ "$size" -le "$threshold2" ]; then
    group2+=("$dataset")
  elif [ "$size" -le "$threshold3" ]; then
    group3+=("$dataset")
  elif [ "$size" -le "$threshold4" ]; then
    group4+=("$dataset")
  elif [ "$size" -le "$threshold5" ]; then
    group5+=("$dataset")
  else
    group6+=("$dataset")
  fi
done

# Now process each group using the specified parallel job counts.
if [ ${#group1[@]} -gt 0 ]; then
  printf "%s\n" "${group1[@]}" | parallel -j 100 process_dataset {}
fi

if [ ${#group2[@]} -gt 0 ]; then
  printf "%s\n" "${group2[@]}" | parallel -j 80 process_dataset {}
fi

if [ ${#group3[@]} -gt 0 ]; then
  printf "%s\n" "${group3[@]}" | parallel -j 40 process_dataset {}
fi

if [ ${#group4[@]} -gt 0 ]; then
  printf "%s\n" "${group4[@]}" | parallel -j 25 process_dataset {}
fi

if [ ${#group5[@]} -gt 0 ]; then
  printf "%s\n" "${group5[@]}" | parallel -j 15 process_dataset {}
fi

if [ ${#group6[@]} -gt 0 ]; then
  printf "%s\n" "${group6[@]}" | parallel -j 5 process_dataset {}
fi

log_msg "INFO" "All processes completed successfully!"
