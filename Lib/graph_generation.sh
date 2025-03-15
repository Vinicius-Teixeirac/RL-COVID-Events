#!/bin/bash

run_consistency_cmd() {
  local ks="$1" kg="$2" kt="$3"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/Consistency" "consistency" "$dataset_stem" "$ks" "$kg" "$kt")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphEvaluation/consistency_main.py --hyperparameters "$ks" "$kg" "$kt" --dataset_name "$dataset_stem"
}

run_consistency_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/Consistency"
  mkdir -p "$log_dir"
  parallel -j "$MAX_PARALLEL_JOBS" run_consistency_cmd ::: "${ks_values[@]}" ::: "${kg_values[@]}" ::: "${kt_values[@]}"
}

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

run_tsne_cmd() {
  local ktsne="$1" ppxty="$2" init="$3" metric"$4" apply_pca="$5"
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
      --ktsne "$ktsne" \
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

run_umap_cmd() {
  local kumap="$1" umap_neighbors="$2" min_dist="$3" init="$4" metric="$5" preknn="$6"
  local log_file
  log_file=$(get_log_file "$LOG_DIR/${dataset_stem}/UMAP" "umap" "$dataset_stem" "$kumap" "$umap_neighbors" "$min_dist" "$init")
  execute_if_not_done "$log_file" "Edges saved successfully" \
    python GraphGeneration/UMAP/umap_main.py \
      --k_umap "$kumap" \
      --n_neighbors "$umap_neighbors" \
      --min_dist "$min_dist" \
      --initialization "$init" \
      --metric "$metric" \
      --dataset_name "$dataset_stem" \
      --precomputed_knn_path "$preknn"
}

run_umap_graphs() {
  local log_dir="$LOG_DIR/${dataset_stem}/UMAP"
  mkdir -p "$log_dir"
  local min_dist_values=(0.0 0.25 0.5 0.75 0.99)
  local init_arr=("pca" "spectral")
  local metric_arr=("euclidean" "cosine")
  parallel -j "$MAX_PARALLEL_JOBS" run_umap_cmd ::: "${kumap_values[@]}" ::: "${umap_neighbors_values[@]}" ::: "${min_dist_values[@]}" ::: "${init_arr[@]}" ::: "${metric_arr[@]}" ::: "$PRECOMPUTED_KNN"
}

run_compute_knn() {
  local dataset_stem="$1"
  local max_neighbors="$2"
  local knn_dir="PrecomputedKNNs"
  mkdir -p "$knn_dir"
  local knn_file="${knn_dir}/${dataset_stem}_knn.pkl"
  local knn_log="$LOG_DIR/${dataset_stem}/knn_${dataset_stem}_${max_neighbors}.log"
  mkdir -p "$LOG_DIR/${dataset_stem}"
  if [ ! -f "$knn_file" ]; then
    log_msg "INFO" "Computing precomputed k-NN for dataset: $dataset_stem with max_neighbors=$max_neighbors. Log: $knn_log"
    if ! python Utils/NeighborhoodUtils/compute_knn.py \
           --dataset_name "$dataset_stem" \
           --max_neighbors "$max_neighbors" \
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
