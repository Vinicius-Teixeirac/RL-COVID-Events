#!/bin/bash
set -euo pipefail

# Loads configuration from .env file if available.
CONFIG_FILE="${CONFIG_FILE:-./.env}"
if [ -f "$CONFIG_FILE" ]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

# Sets configurable paths with defaults.
DATASET_DIR="${DATASET_DIR:-DataPreparation/UsageDatasets}"
LOG_DIR="${LOG_DIR:-logs}"
CRITDD_RESULTS="${CRITDD_RESULTS:-CritddResults}"

export DATASET_DIR LOG_DIR CRITDD_RESULTS 

# Starting the necessary directories.
mkdir -p "$LOG_DIR" "$DATASET_DIR" "$CRITDD_RESULTS" 

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
# Sort datasets by file size and assign parallel jobs:
#   - First file:    110 jobs  
#   - Second file:   90 jobs  
#   - Third file:    40 jobs  
#   - Fourth file:   30 jobs  
#   - Fifth file:    20 jobs  
#   - Sixth file:    5 jobs  
#   - Additional files: Default to 5 jobs  
#  
# Note: These are experimental values based on a 128-core, 126 GB machine. You may consider using as much cores as possible without running out of RAM
###############################################################################

# Sort datasets by file size (ascending order).
# Note: Adjust stat flags if you're on a non-GNU system.
mapfile -t sorted_datasets < <(
  for dataset in "${datasets[@]}"; do
    size=$(stat -c%s "$dataset")
    echo "$size:$dataset"
  done | sort -n | cut -d: -f2-
)

# Define desired parallel job counts for the first 6 datasets.
job_counts=(110 90 40 30 20 5)

for ((i = 0; i < ${#sorted_datasets[@]}; i++)); do
  dataset="${sorted_datasets[$i]}"
  
  if [ "$i" -lt "${#job_counts[@]}" ]; then
    jobs="${job_counts[$i]}"
  else
    jobs="${job_counts[-1]}"
  fi

  log_msg "INFO" "Processing $dataset with $jobs parallel jobs."

  MAX_PARALLEL_JOBS="$jobs"
  export MAX_PARALLEL_JOBS

  # Pass dataset to parallel 
  parallel -j 1 process_dataset ::: "$dataset"
done

log_msg "INFO" "All processes completed successfully!"