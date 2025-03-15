#!/bin/bash

process_dataset() {
  local dataset_path="$1"
  local dataset_name
  dataset_name=$(basename "$dataset_path")
  dataset_stem="${dataset_name%.pkl}"
  export dataset_stem dataset_path
  log_msg "INFO" "Starting processing for dataset: $dataset_stem"
  mkdir -p "$LOG_DIR/$dataset_stem"

  # 1. Generate event features.
  if ! python DataPreparation/event_features.py \
         --dataset "$dataset_path" \
         --output_dir "$OUTPUT_DIR" \
         > "$LOG_DIR/$dataset_stem/event_features.log" 2>&1; then
    log_msg "ERROR" "Feature generation failed for dataset: $dataset_stem"
    return 1
  fi

  # 2. Determine number of rows.
  local num_rows
  num_rows=$(python -c "import pandas as pd; print(len(pd.read_pickle('$dataset_path')))" 2>> "$LOG_DIR/$dataset_stem/num_rows.err")
  if [ -z "$num_rows" ]; then
    log_msg "ERROR" "Could not determine number of rows for dataset: $dataset_stem"
    return 1
  fi

  # 3. Generate hyperparameter groups.
  local hypergroups
  mapfile -t hypergroups < <(generate_hyperparams "$num_rows")
  if [ "${#hypergroups[@]}" -ne 7 ]; then
    log_msg "ERROR" "Hyperparameter generation failed for dataset: $dataset_stem"
    return 1
  fi

  IFS=' ' read -r -a ks_values             <<< "${hypergroups[0]}"
  IFS=' ' read -r -a kg_values             <<< "${hypergroups[1]}"
  IFS=' ' read -r -a kt_values             <<< "${hypergroups[2]}"
  IFS=' ' read -r -a kpca_values           <<< "${hypergroups[3]}"
  IFS=' ' read -r -a ktsne_values          <<< "${hypergroups[4]}"
  IFS=' ' read -r -a kumap_values          <<< "${hypergroups[5]}"
  IFS=' ' read -r -a umap_neighbors_values <<< "${hypergroups[6]}"

  local last_index=$((${#umap_neighbors_values[@]} - 1))
  local max_neighbors="${umap_neighbors_values[$last_index]}"

  PRECOMPUTED_KNN=$(run_compute_knn "$dataset_stem" "$max_neighbors")
  export PRECOMPUTED_KNN

  # Run graph generation steps.
  run_consistency_graphs
  run_pca_graphs
  run_tsne_graphs
  run_umap_graphs

  # Run evaluations.
  local log_eval_base="$LOG_DIR/${dataset_stem}"
  parallel -j 4 run_evaluation {1} "$dataset_path" "$dataset_stem" "$log_eval_base" ::: "PCA" "TSNE" "TSNE+PCA" "UMAP"
  
  # Generate critical difference diagram.
  generate_critdd "$dataset_stem" "$dataset_path"

  log_msg "INFO" "Finished processing dataset: $dataset_stem"
}
