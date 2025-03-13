#!/bin/bash
set -euo pipefail

###############################################
# Logging function.
###############################################
log_msg() {
  local level="$1"
  shift
  local message="$*"
  if [ -n "${MAIN_LOG-}" ]; then
    {
      flock -x 200
      echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] - $message"
    } >> "$MAIN_LOG" 2>&1 200>"$LOCK_FILE"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] - $message" >&2
  fi
}

###############################################
# Returns a log file name based on parameters.
###############################################
get_log_file() {
  local base_dir="$1"
  local prefix="$2"
  local dataset_stem="$3"
  shift 3
  local params="$*"
  echo "${base_dir}/${prefix}_${dataset_stem}_${params}.log"
}

###############################################
# Executes a command if the log file is missing or does not contain the success marker.
# Usage: execute_if_not_done <log_file> <marker> <command> [args...]
###############################################
execute_if_not_done() {
  local log_file="$1"
  local marker="${2:-}"
  shift 2
  if [ ! -s "$log_file" ] || { [ -n "$marker" ] && ! grep -Fq "$marker" "$log_file"; }; then
    if ! "$@" > "$log_file" 2>&1; then
      log_msg "ERROR" "Command failed: $*"
      return 1
    fi
  fi
}

###############################################
# Checks required dependencies.
###############################################
check_dependencies() {
  local missing_deps=()
  for dep in python jupyter papermill parallel bc; do
    if ! type -p "$dep" &>/dev/null; then
      missing_deps+=("$dep")
    fi
  done
  if (( ${#missing_deps[@]} )); then
    echo "Error: Missing dependencies: ${missing_deps[*]}. Please install them before running the script." >&2
    exit 1
  fi
}

###############################################
# Runs preprocessing if not already done.
###############################################
run_preprocessing() {
  if [ -s "$PREPROC_LOG" ] && grep -Fq "Preprocessing completed successfully" "$PREPROC_LOG"; then
      log_msg "INFO" "Preprocessing already completed. Skipping preprocessing step."
  else
      log_msg "INFO" "Starting preprocessing..."
      if ! jupyter nbconvert --to notebook --execute --inplace DataPreparation/pre_processing_datasets.ipynb > /dev/null 2> "$PREPROC_LOG"; then
        log_msg "ERROR" "Preprocessing failed. Check $PREPROC_LOG for details."
        exit 1
      fi
      echo "Preprocessing completed successfully" >> "$PREPROC_LOG"
      log_msg "INFO" "Preprocessing completed successfully."
  fi
}

###############################################
# Runs evaluation for a given method.
###############################################
run_evaluation() {
  local method="$1"
  local dataset_path="$2"
  local dataset_stem="$3"
  local base_log_dir="$4"  # e.g. "$LOG_DIR/${dataset_stem}"
  local log_eval
  log_eval=$(get_log_file "$base_log_dir/Evaluation" "eval_${method//+/_}" "$dataset_stem")
  mkdir -p "$base_log_dir/Evaluation"

  local comp_folder n_jobs
  case "$method" in
    PCA)
      comp_folder="./GeneratedGraphs/PCA"
      n_jobs=25
      ;;
    TSNE)
      comp_folder="./GeneratedGraphs/TSNE"
      n_jobs=25
      ;;
    "TSNE+PCA")
      comp_folder="./GeneratedGraphs/TSNE_PCA"
      n_jobs=25
      ;;
    UMAP)
      comp_folder="./GeneratedGraphs/UMAP"
      n_jobs=25
      ;;
    *)
      log_msg "ERROR" "Unknown evaluation method: $method"
      return 1
      ;;
  esac

  execute_if_not_done "$log_eval" "evaluation completed successfully" \
    python GraphEvaluation/evaluation_main.py \
      --dataset_path "$dataset_path" \
      --reference_folder "./GeneratedGraphs/Consistency" \
      --comparison_folder "$comp_folder" \
      --method "$method" \
      --output_dir "./EvaluationResults" \
      --n_jobs "$n_jobs" || \
      log_msg "ERROR" "Evaluation for $method failed for dataset: $dataset_stem"
}

###############################################
# Generates critical difference diagram.
###############################################
generate_critdd() {
  local dataset_stem="$1"
  local dataset_path="$2"
  local critdd_log="$LOG_DIR/${dataset_stem}/critdd_${dataset_stem}.log"
  mkdir -p "$LOG_DIR/${dataset_stem}"
  execute_if_not_done "$critdd_log" "critical difference diagrams generated successfully" \
    papermill GraphEvaluation/critdd_template.ipynb \
      "$CRITDD_RESULTS/critdd_${dataset_stem}.ipynb" \
      -p dataset "$dataset_stem" || \
      log_msg "ERROR" "Critical difference diagram generation failed for dataset: $dataset_stem"
}

###############################################
# Generates hyperparameter groups.
# Outputs 7 lines (one per hyperparameter group).
###############################################
generate_hyperparams() {
  local num_rows="$1"
  local num_parts=10
  local min_value=1
  local max_value
  max_value=$(echo "scale=0; sqrt($num_rows)" | bc)
  (( max_value < min_value )) && max_value=$min_value
  local interval=0
  (( num_parts > 1 )) && interval=$(( (max_value - min_value) / (num_parts - 1) ))
  local ks=() kg=() kt=() kpca=() ktsne=() kumap=() umap_neighbors=()
  local i value
  for (( i=0; i<num_parts; i++ )); do
    if (( i == num_parts - 1 )); then
      value="$max_value"
    else
      value=$(( min_value + i * interval ))
    fi
    ks+=("$value")
    kg+=("$value")
    kt+=("$value")
    kpca+=("$value")
    ktsne+=("$value")
    kumap+=("$value")
    umap_neighbors+=("$value")
  done
  echo "${ks[*]}"
  echo "${kg[*]}"
  echo "${kt[*]}"
  echo "${kpca[*]}"
  echo "${ktsne[*]}"
  echo "${kumap[*]}"
  echo "${umap_neighbors[*]}"
}

###############################################
# Functions to run graph generation commands in parallel.
# These helper functions are exported so GNU parallel can call them.
###############################################

# Consistency Graphs.
run_consistency_cmd() {
  local ks="$1" kg="$2" kt="$3"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/Consistency" "consistency" "$dataset_stem" "$ks" "$kg" "$kt")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphEvaluation/consistency_main.py --hyperparameters "$ks" "$kg" "$kt" --dataset_path "$dataset_path"
}
export -f run_consistency_cmd

run_consistency_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/Consistency"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_consistency_cmd ::: "${ks_values[@]}" ::: "${kg_values[@]}" ::: "${kt_values[@]}"
}

# PCA Graphs.
run_pca_cmd() {
  local kpca="$1" whiten="$2"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/PCA" "pca" "$dataset_stem" "$kpca" "$whiten")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/pca_main.py --hyperparameters "$kpca" 2 --whiten "$whiten" --dataset_path "$dataset_path"
}
export -f run_pca_cmd

run_pca_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/PCA"
  mkdir -p "$log_dir"
  local whiten_arr=(0 1)
  parallel -j "$MAX_PARALLEL_JOBS" run_pca_cmd ::: "${kpca_values[@]}" ::: "${whiten_arr[@]}"
}

# t-SNE Graphs.
run_tsne_cmd() {
  local ktsne="$1" ppxty="$2" init="$3" apply_pca="$4"
  local prefix
  if [ "$apply_pca" -eq 1 ]; then
    prefix="tsne_pca"
  else
    prefix="tsne"
  fi
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/TSNE" "$prefix" "$dataset_stem" "$ktsne" "$ppxty" "$init")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/tsne_main.py \
      --hyperparameters "$ktsne" "$ppxty" 2 42 \
      --apply_pca "$apply_pca" \
      --initialization "$init" \
      --dataset_path "$dataset_path"
}
export -f run_tsne_cmd

run_tsne_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/TSNE"
  mkdir -p "$log_dir"
  local ppxty_values=(5 10 20 30 40 50)
  local init_arr=("pca" "spectral")
  
  # Without PCA.
  parallel -j "$MAX_PARALLEL_JOBS" run_tsne_cmd ::: "${ktsne_values[@]}" ::: "${ppxty_values[@]}" ::: "${init_arr[@]}" ::: 0;
  # With PCA.
  parallel -j "$MAX_PARALLEL_JOBS" run_tsne_cmd ::: "${ktsne_values[@]}" ::: "${ppxty_values[@]}" ::: "${init_arr[@]}" ::: 1;
}

run_compute_knn() {
  local dataset_stem="$1"
  local knn_dir="./precomputed_knn"
  mkdir -p "$knn_dir"
  local knn_file="${knn_dir}/${dataset_stem}_knn.pkl"
  
  if [ ! -f "$knn_file" ]; then
    log_msg "INFO" "Computing precomputed k-NN for dataset: $dataset_stem"
    # Use the last element in umap_neighbors_values as max_neighbors
    local max_neighbors="${umap_neighbors_values[-1]}"
    python compute_knn.py --dataset_name "$dataset_stem" --max_neighbors "$max_neighbors" --random_state 42 --output_path "$knn_file"
  else
    log_msg "INFO" "Precomputed k-NN already exists: $knn_file"
  fi
  echo "$knn_file"
}

# UMAP Graphs.
run_umap_cmd() {
  local kumap="$1" umap_neighbors="$2" min_dist="$3" init="$4"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/UMAP" "umap" "$dataset_stem" "$kumap" "$umap_neighbors" "$min_dist" "$init")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/umap_main.py \
      --int_hyperparameters "$kumap" "$umap_neighbors" 2 42 \
      --float_hyperparameters "$min_dist" \
      --initialization "$init" \
      --dataset_path "$dataset_path" \
      --precomputed_knn_path "$PRECOMPUTED_KNN"
}

export -f run_umap_cmd

run_umap_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/UMAP"
  mkdir -p "$log_dir"
  local min_dist_values=(0.0 0.25 0.5 0.75 0.99)
  local init_arr=("pca" "spectral")
  # # Defines a UMAP-specific parallel job count, since its implementation uses one core only:
  # local MAX_UMAP_JOBS="${MAX_UMAP_JOBS:-100}"
  parallel -j "$MAX_PARALLEL_JOBS" run_umap_cmd ::: "${kumap_values[@]}" ::: "${umap_neighbors_values[@]}" ::: "${min_dist_values[@]}" ::: "${init_arr[@]}"
}


export -f run_consistency_cmd run_consistency_graphs \
  run_pca_cmd run_pca_graphs \
  run_tsne_cmd run_tsne_graphs \
  run_umap_cmd run_umap_graphs 

###############################################
# Process a single dataset.
###############################################
process_dataset() {
  local dataset_path="$1"
  local dataset_name
  dataset_name=$(basename "$dataset_path")
  dataset_stem="${dataset_name%.pkl}"
  export dataset_stem dataset_path
  log_msg "INFO" "Starting processing for dataset: $dataset_stem"
  mkdir -p "$LOG_DIR/$dataset_stem"

  # 1. Generates Event Features.
  if ! python DataPreparation/event_features.py \
         --dataset "$dataset_path" \
         --output_dir "$OUTPUT_DIR" \
         > "$LOG_DIR/$dataset_stem/event_features.log" 2>&1; then
    log_msg "ERROR" "Feature generation failed for dataset: $dataset_stem"
    return 1
  fi

  # 2. Determines number of rows.
  local num_rows
  num_rows=$(python -c "import pandas as pd; print(len(pd.read_pickle('$dataset_path')))" 2>> "$LOG_DIR/$dataset_stem/num_rows.err")
  if [ -z "$num_rows" ]; then
    log_msg "ERROR" "Could not determine number of rows for dataset: $dataset_stem"
    return 1
  fi

  # 3. Generates hyperparameter groups.
  local hypergroups
  mapfile -t hypergroups < <(generate_hyperparams "$num_rows")
  if [ "${#hypergroups[@]}" -ne 7 ]; then
    log_msg "ERROR" "Hyperparameter generation failed for dataset: $dataset_stem"
    return 1
  fi

  # Reads hyperparameter groups into arrays.
  IFS=' ' read -r -a ks_values             <<< "${hypergroups[0]}"
  IFS=' ' read -r -a kg_values             <<< "${hypergroups[1]}"
  IFS=' ' read -r -a kt_values             <<< "${hypergroups[2]}"
  IFS=' ' read -r -a kpca_values           <<< "${hypergroups[3]}"
  IFS=' ' read -r -a ktsne_values          <<< "${hypergroups[4]}"
  IFS=' ' read -r -a kumap_values          <<< "${hypergroups[5]}"
  IFS=' ' read -r -a umap_neighbors_values <<< "${hypergroups[6]}"

  PRECOMPUTED_KNN=$(run_compute_knn "$dataset_stem")
  export PRECOMPUTED_KNN

  ###############################################
  # Runs graph generation steps.
  ###############################################
  run_consistency_graphs
  run_pca_graphs
  run_tsne_graphs
  run_umap_graphs

  ###############################################
  # Evaluation Steps.
  ###############################################
  local log_eval_base="$LOG_DIR/${dataset_stem}"
  parallel -j 4 run_evaluation {1} "$dataset_path" "$dataset_stem" "$log_eval_base" ::: "PCA" "TSNE" "TSNE+PCA" "UMAP"
  
  ###############################################
  # Generates critical difference diagram.
  ###############################################
  generate_critdd "$dataset_stem" "$dataset_path"

  log_msg "INFO" "Finished processing dataset: $dataset_stem"
}
export -f process_dataset
export -f log_msg generate_hyperparams get_log_file execute_if_not_done run_evaluation generate_critdd

###############################################
# Main Script Execution
###############################################
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -h, --help         Show this help message.

Description:
  This script processes datasets by generating event features, computing hyperparameters,
  generating graphs using various methods, and evaluating the results.
EOF
  exit 0
fi

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

# Centralized summary log and lock file.
MAIN_LOG="$LOG_DIR/script_summary.log"
PREPROC_LOG="$LOG_DIR/pre_processing.log"
LOCK_FILE="/tmp/script_log.lock"

# Catches errors and log them.
trap 'log_msg "ERROR" "Error: Command '\''$BASH_COMMAND'\'' failed on line ${LINENO}. Exiting..."; exit 1' ERR

# Checks dependencies.
check_dependencies

# Initializes PYTHONPATH safely.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Runs preprocessing to get the datasets.
run_preprocessing

# Gathers dataset files AFTER preprocessing.
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

