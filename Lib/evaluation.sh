#!/bin/bash

run_evaluation() {
  local method="$1"
  local dataset_path="$2"
  local dataset_stem="$3"
  local base_log_dir="$4"
  local log_eval
  log_eval=$(get_log_file "$base_log_dir/Evaluation" "eval_${method//+/_}" "$dataset_stem")
  mkdir -p "$base_log_dir/Evaluation"
  local comp_folder n_jobs
  case "$method" in
    ICA)
      comp_folder="./GeneratedGraphs/ICA"
      n_jobs=-1
      ;;
    Isomap)
      comp_folder="./GeneratedGraphs/Isomap"
      n_jobs=-1
      ;;
    LLE)
      comp_folder="./GeneratedGraphs/LLE"
      n_jobs=-1
      ;;
    PCA)
      comp_folder="./GeneratedGraphs/PCA"
      n_jobs=-1
      ;;
    RandomProjection)
      comp_folder="./GeneratedGraphs/RandomProjection"
      n_jobs=-1
      ;;
    Spectral)
      comp_folder="./GeneratedGraphs/Spectral"
      n_jobs=-1
      ;;
    TSNE)
      comp_folder="./GeneratedGraphs/TSNE"
      n_jobs=-1
      ;;
    "TSNE+PCA")
      comp_folder="./GeneratedGraphs/TSNE_PCA"
      n_jobs=-1
      ;;
    UMAP)
      comp_folder="./GeneratedGraphs/UMAP"
      n_jobs=-1
      ;;
    *)
      log_msg "ERROR" "Unknown evaluation method: $method"
      return 1
      ;;
  esac

  execute_if_not_done "$log_eval" "evaluation completed successfully" \
    python GraphEvaluation/evaluation_main.py \
      --dataset_name "$dataset_stem" \
      --reference_folder "./GeneratedGraphs/Consistency" \
      --comparison_folder "$comp_folder" \
      --method "$method" \
      --output_dir "./EvaluationResults" \
      --n_jobs "$n_jobs" || \
      log_msg "ERROR" "Evaluation for $method failed for dataset: $dataset_stem"
}


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

export -f run_evaluation generate_critdd