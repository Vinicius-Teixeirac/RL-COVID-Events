process_dataset() {
  # Foundational variables: inputs and basic setup
  local dataset_path="$1"
  local dataset_name
  dataset_name=$(basename "$dataset_path")
  local dataset_stem="${dataset_name%.pkl}"
  export dataset_stem dataset_path
  
  local output_dir="DatasetEventFeatures"
  
  log_msg "INFO" "Starting processing for dataset: $dataset_stem"
  mkdir -p "$LOG_DIR/$dataset_stem"

  # 1. Generate event features.
  if ! python DataPreparation/event_features.py \
         --dataset_path "$dataset_path" \
         --output_dir "$output_dir" \
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

  # 3. Generate hyperparameter values.
  local k_values
  IFS=' ' read -r -a k_values <<< "$(generate_hyperparams "$num_rows")"

  # Use the same k values for all methods.
  ks_values=("${k_values[@]}")
  kg_values=("${k_values[@]}")
  kt_values=("${k_values[@]}")
  kica_values=("${k_values[@]}")
  kisomap_values=("${k_values[@]}")
  klle_values=("${k_values[@]}")
  kpca_values=("${k_values[@]}")
  krp_values=("${k_values[@]}")
  kspectral_values=("${k_values[@]}")
  ktsne_values=("${k_values[@]}")
  kumap_values=("${k_values[@]}")

  export ks_values kg_values kt_values kica_values kisomap_values klle_values kpca_values krp_values kspectral_values ktsne_values kumap_values

  local n_neighbors=()
  for k in "${k_values[@]}"; do
    n_neighbors+=("$(( k + 5 ))")
  done

  local last_index=$((${#n_neighbors[@]} - 1))
  local max_neighbors="${n_neighbors[$last_index]}"

  export n_neighbors 

  PRECOMPUTED_KNN_EUCLIDEAN=$(run_compute_knn "$dataset_stem" "$max_neighbors" "euclidean")
  PRECOMPUTED_KNN_COSINE=$(run_compute_knn "$dataset_stem" "$max_neighbors" "cosine")
  
  export PRECOMPUTED_KNN_EUCLIDEAN PRECOMPUTED_KNN_COSINE

  # Run graph generation steps.
  run_consistency_graphs
  run_ica_graphs
  run_isomap_graphs   
  run_lle_graphs      
  run_pca_graphs
  run_rp_graphs        
  run_spectral_graphs 
  run_tsne_graphs
  run_umap_graphs

  # Run evaluations.
  local log_eval_base="$LOG_DIR/${dataset_stem}"
  parallel -j "${EVAL_PARALLEL_JOBS:-9}" run_evaluation {1} "$dataset_path" "$dataset_stem" "$log_eval_base" \
    ::: "ICA" "Isomap" "LLE" "PCA" "RandomProjection" "Spectral" "TSNE" "TSNE+PCA" "UMAP"     

  # Generate critical difference diagram.
  generate_critdd "$dataset_stem" "$dataset_path"

  log_msg "INFO" "Finished processing dataset: $dataset_stem"
}

export -f process_dataset
