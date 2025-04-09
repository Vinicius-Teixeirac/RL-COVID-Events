#!/bin/bash

# Consistency
run_consistency_cmd() {
  local ks="$1" kg="$2" kt="$3"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/Consistency" "consistency" "$dataset_stem" "$ks" "$kg" "$kt")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphEvaluation/consistency_main.py --k_s "$ks" --k_g "$kg" --k_t "$kt" --dataset_name "$dataset_stem"
}

run_consistency_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/Consistency"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_consistency_cmd ::: "${ks_values[@]}" ::: "${kg_values[@]}" ::: "${kt_values[@]}"
}

# ICA
run_ica_cmd() {
  local k_ica="$1" whiten="$2" fun="$3"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/ICA" "ica" "$dataset_stem" "$k_ica" "$whiten" "$fun")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/ICA/ica_main.py --k_ica "$k_ica" --whiten "$whiten" --fun "$fun" --dataset_name "$dataset_stem"
}

run_ica_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/ICA"
  mkdir -p "$log_dir"
  local whiten_arr=("unit-variance" "arbitrary-variance") # no whitten = false since it will ignore n_components
  local fun_arr=("logcosh" "exp" "cube")
  parallel -j "$MAX_PARALLEL_JOBS" run_ica_cmd ::: "${kica_values[@]}" ::: "${whiten_arr[@]}" ::: "${fun_arr[@]}"
}

# Isomap
run_isomap_cmd() {
  local k_isomap="$1" n_neighbors="$2"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/Isomap" "isomap" "$dataset_stem" "$k_isomap" "$n_neighbors")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/Isomap/isomap_main.py --k_isomap "$k_isomap" --n_neighbors "$n_neighbors" --dataset_name "$dataset_stem"
}

run_isomap_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/Isomap"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_isomap_cmd ::: "${kisomap_values[@]}" ::: "${n_neighbors[@]}"
}

# Locally Linear Embedding
run_lle_cmd() {
  local k_lle="$1" n_neighbors="$2" method="$3"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/LLE" "lle" "$dataset_stem" "$k_lle" "$n_neighbors" "$method")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/LLE/lle_main.py --k_lle "$k_lle" --n_neighbors "$n_neighbors" --method "$method" --dataset_name "$dataset_stem"
}

run_lle_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/LLE"
  mkdir -p "$log_dir"
  local methods_arr=("standard" "modified") # no hessian neither ltsa
  parallel -j "$MAX_PARALLEL_JOBS" run_lle_cmd ::: "${klle_values[@]}" ::: "${n_neighbors[@]}" ::: "${methods_arr[@]}"
}

# PCA
run_pca_cmd() {
  local kpca="$1" whiten="$2"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/PCA" "pca" "$dataset_stem" "$kpca" "$whiten")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/PCA/pca_main.py --k_pca "$kpca" --whiten "$whiten" --dataset_name "$dataset_stem"
}

run_pca_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/PCA"
  mkdir -p "$log_dir"
  local whiten_arr=(0 1)
  parallel -j "$MAX_PARALLEL_JOBS" run_pca_cmd ::: "${kpca_values[@]}" ::: "${whiten_arr[@]}"
}

# Gaussian Random Projection
run_rp_cmd() {
  local k_rp="$1"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/RandomProjection" "randomprojection" "$dataset_stem" "$k_rp")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/RandomProjection/random_main.py --k_rp "$k_rp" --dataset_name "$dataset_stem"
}

run_rp_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/RandomProjection"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_rp_cmd ::: "${krp_values[@]}"
}

# Laplacian Eigenmaps (Spectral Decomposition)
run_spectral_cmd() {
  local k_spectral="$1" n_neighbors="$2"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/Spectral" "spectral" "$dataset_stem" "$k_spectral" "$n_neighbors")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/SpectralEmbedding/spectral_main.py --k_spectral "$k_spectral" --n_neighbors "$n_neighbors" --dataset_name "$dataset_stem"
}

run_spectral_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/Spectral"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_spectral_cmd ::: "${kspectral_values[@]}" ::: "${n_neighbors[@]}"
}

# t-SNE
run_tsne_cmd() {
  local ktsne="$1" ppxty="$2" init="$3" metric="$4" apply_pca="$5"
  local prefix
  if [ "$apply_pca" -eq 1 ]; then
    prefix="tsne_pca"
  else
    prefix="tsne"
  fi
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/TSNE" "$prefix" "$dataset_stem" "$ktsne" "$ppxty" "$init" "$metric")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/TSNE/tsne_main.py \
      --k_tsne "$ktsne" \
      --perplexity "$ppxty" \
      --initialization "$init" \
      --metric "$metric" \
      --apply_pca "$apply_pca" \
      --dataset_name "$dataset_stem"
}

run_tsne_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/TSNE"
  mkdir -p "$log_dir"
  local ppxty_values=(5 10 20 30 40 50)
  local init_arr=("pca" "spectral")
  local metric_arr=("euclidean" "cosine")
  parallel -j "$MAX_PARALLEL_JOBS" run_tsne_cmd ::: "${ktsne_values[@]}" ::: "${ppxty_values[@]}" ::: "${init_arr[@]}" ::: "${metric_arr[@]}" ::: 0;
  parallel -j "$MAX_PARALLEL_JOBS" run_tsne_cmd ::: "${ktsne_values[@]}" ::: "${ppxty_values[@]}" ::: "${init_arr[@]}" ::: "${metric_arr[@]}" ::: 1;
}


# UMAP
run_umap_cmd() {
  local kumap="$1" n_neighbors="$2" min_dist="$3" init="$4" metric="$5"
  
  local preknn
  # Select the appropriate precomputed k-NN file based on the metric.
  if [ "$metric" = "euclidean" ]; then
    preknn="$PRECOMPUTED_KNN_EUCLIDEAN"
  elif [ "$metric" = "cosine" ]; then
    preknn="$PRECOMPUTED_KNN_COSINE"
  fi

  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/UMAP" "umap" "$dataset_stem" "$kumap" "$n_neighbors" "$min_dist" "$init" "$metric")
  
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/UMAP/umap_main.py \
      --k_umap "$kumap" \
      --n_neighbors "$n_neighbors" \
      --min_dist "$min_dist" \
      --initialization "$init" \
      --metric "$metric" \
      --dataset_name "$dataset_stem" \
      --precomputed_knn_path "$preknn" 
}

run_umap_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/UMAP"
  mkdir -p "$log_dir"
  local min_dist_values=(0.0 0.25 0.5 0.99)
  local init_arr=("pca" "spectral")
  local metric_arr=("euclidean" "cosine")
  
  parallel -j "$MAX_PARALLEL_JOBS" run_umap_cmd ::: "${kumap_values[@]}" ::: "${n_neighbors[@]}" ::: "${min_dist_values[@]}" ::: "${init_arr[@]}" ::: "${metric_arr[@]}"
}

# Compute precomputed k-NN
run_compute_knn() {
  local dataset_stem="$1"
  local max_neighbors="$2"
  local metric="$3"
  local knn_dir="./PrecomputedKNNs"
  mkdir -p "$knn_dir"
  local knn_file="${knn_dir}/${dataset_stem}_${metric}_knn.pkl"
  local knn_log="$LOG_DIR/${dataset_stem}/knn_${dataset_stem}_${max_neighbors}_${metric}.log"
  mkdir -p "$LOG_DIR/${dataset_stem}"
  if [ ! -f "$knn_file" ]; then
    log_msg "INFO" "Computing precomputed k-NN for dataset: $dataset_stem with max_neighbors=$max_neighbors. Log: $knn_log"
    if ! python Utils/NeighborhoodUtils/compute_knn.py \
           --dataset_name "$dataset_stem" \
           --max_neighbors "$max_neighbors" \
           --metric "$metric" \
           --random_state 42 \
           --output_path "$knn_file" \
           > "$knn_log" 2>&1; then
      log_msg "ERROR" "k-NN computation failed for dataset: $dataset_stem. Check log: $knn_log"
      return 1
    fi
    log_msg "INFO" "k-NN computed successfully for dataset: $dataset_stem"
  else
    log_msg "INFO" "Precomputed k-NN already exists: $knn_file"
  fi
  echo "$knn_file"
}

export -f run_consistency_cmd run_consistency_graphs run_pca_cmd run_pca_graphs run_tsne_cmd run_tsne_graphs run_umap_cmd run_umap_graphs run_compute_knn run_ica_cmd run_ica_graphs run_isomap_cmd run_isomap_graphs run_lle_cmd run_lle_graphs run_rp_cmd run_rp_graphs run_spectral_cmd run_spectral_graphs
