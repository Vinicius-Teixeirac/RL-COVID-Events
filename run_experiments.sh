#!/bin/bash
set -euo pipefail  # Ensures the script exits on errors, unset variables, and pipeline failures

# Initializes PYTHONPATH safely
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p logs

# Defines where the datasets are stored.
dataset_dir="DataPreparation/UsageDatasets"
datasets=("$dataset_dir"/*.pkl)

# Defines a maximum number of parallel jobs.
MAX_PARALLEL_JOBS=10
job_count=0

###############################################
# Given the number of rows in the dataset, it generates
# a set of hyperparameter values for seven groups:
# ks, kg, kt, kpca, ktsne, kumap, and umap_neighbors.
# It outputs each group on its own line.
###############################################
generate_hyperparams() {
  local num_rows=$1
  local num_parts=10
  local min_value=3
  local max_value
  max_value=$(echo "scale=0; sqrt($num_rows)" | bc)

  # Ensures max_value is at least min_value.
  if (( max_value < min_value )); then
    max_value=$min_value
  fi

  # Calculates interval if more than one part.
  local interval=0
  if (( num_parts > 1 )); then
    interval=$(( (max_value - min_value) / (num_parts - 1) ))
  fi

  # Initializes arrays for each hyperparameter group.
  local ks=() kg=() kt=() kpca=() ktsne=() kumap=() umap_neighbors=()

  for (( i=0; i<num_parts; i++ )); do
    local value
    if (( i == num_parts - 1 )); then
      value=$max_value
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

  # Outputs each group on its own line.
  # (The order is: ks, kg, kt, kpca, ktsne, kumap, umap_neighbors)
  echo "${ks[*]}"
  echo "${kg[*]}"
  echo "${kt[*]}"
  echo "${kpca[*]}"
  echo "${ktsne[*]}"
  echo "${kumap[*]}"
  echo "${umap_neighbors[*]}"
}

###############################################
# Main loop: iterates over each dataset.
###############################################
for dataset_path in "${datasets[@]}"; do
  {
    dataset_name=$(basename "$dataset_path")
    dataset_stem="${dataset_name%.pkl}"

    # 1. Generate representations for the dataset.
    python DataPreparation/generate_representations.py \
      --dataset "$dataset_path" \
      --output_dir DatasetRepresentations

    # 2. Determine number of rows (used for computing hyperparameters).
    num_rows=$(python -c "import pandas as pd; print(len(pd.read_pickle('$dataset_path')))")

    # 3. Read the hyperparameters (7 lines, one per group).
    readarray -t hypergroups < <(generate_hyperparams "$num_rows")

    # 4. Split each hyperparameter group into arrays.
    IFS=' ' read -r -a ks_values             <<< "${hypergroups[0]}"
    IFS=' ' read -r -a kg_values             <<< "${hypergroups[1]}"
    IFS=' ' read -r -a kt_values             <<< "${hypergroups[2]}"
    IFS=' ' read -r -a kpca_values           <<< "${hypergroups[3]}"
    IFS=' ' read -r -a ktsne_values          <<< "${hypergroups[4]}"
    IFS=' ' read -r -a kumap_values          <<< "${hypergroups[5]}"
    IFS=' ' read -r -a umap_neighbors_values <<< "${hypergroups[6]}"

    # --- Consistency Graphs ---
    for ks in "${ks_values[@]}"; do
      for kg in "${kg_values[@]}"; do
        for kt in "${kt_values[@]}"; do
          log_file="logs/consistency_${dataset_stem}_${ks}_${kg}_${kt}.log"
          if [ ! -s "$log_file" ] || ! grep -q "Edges saved successfully" "$log_file"; then
            python GraphEvaluation/generate_consistency_graphs.py \
              --hyperparameters "$ks" "$kg" "$kt" \
              --dataset_path "$dataset_path" > "$log_file" 2>&1 &
            job_count=$((job_count + 1))
            if (( job_count >= MAX_PARALLEL_JOBS )); then
              wait
              job_count=0
            fi
          fi
        done
      done
    done

    # --- PCA Graphs ---
    desired_dimensionality=2
    whiten=(0 1)
    for kpca in "${kpca_values[@]}"; do
      for value in "${whiten[@]}"; do
        log_file="logs/pca_${dataset_stem}_${kpca}_${value}.log"
        if [ ! -s "$log_file" ] || ! grep -q "Edges saved successfully" "$log_file"; then
          python GraphGeneration/generate_pca_graphs.py \
            --hyperparameters "$kpca" "$desired_dimensionality" \
            --whiten "$value" \
            --dataset_path "$dataset_path" > "$log_file" 2>&1 &
          job_count=$((job_count + 1))
          if (( job_count >= MAX_PARALLEL_JOBS )); then
            wait
            job_count=0
          fi
        fi
      done
    done

    # --- t-SNE Graphs ---
    ppxty_values=(5 10 20 30 40 50)
    rnd_state=42
    for ktsne in "${ktsne_values[@]}"; do
      for ppxty in "${ppxty_values[@]}"; do
        log_file="logs/tsne_${dataset_stem}_${ktsne}_${ppxty}.log"
        if [ ! -s "$log_file" ] || ! grep -q "Edges saved successfully" "$log_file"; then
          python GraphGeneration/generate_tsne_graphs.py \
            --hyperparameters "$ktsne" "$ppxty" "$desired_dimensionality" "$rnd_state" \
            --dataset_path "$dataset_path" > "$log_file" 2>&1 &
          job_count=$((job_count + 1))
          if (( job_count >= MAX_PARALLEL_JOBS )); then
            wait
            job_count=0
          fi
        fi
      done
    done

    # --- UMAP Graphs ---
    min_dist_values=(0.0 0.1 0.25 0.5 0.8 0.99)
    for kumap in "${kumap_values[@]}"; do
      for umap_neighbors in "${umap_neighbors_values[@]}"; do
        for min_dist in "${min_dist_values[@]}"; do
          log_file="logs/umap_${dataset_stem}_${kumap}_${umap_neighbors}_${min_dist}.log"
          if [ ! -s "$log_file" ] || ! grep -q "Edges saved successfully" "$log_file"; then
            python GraphGeneration/generate_umap_graphs.py \
              --int_hyperparameters "$kumap" "$umap_neighbors" "$desired_dimensionality" "$rnd_state" \
              --float_hyperparameters "$min_dist" \
              --dataset_path "$dataset_path" > "$log_file" 2>&1 &
            job_count=$((job_count + 1))
            if (( job_count >= MAX_PARALLEL_JOBS )); then
              wait
              job_count=0
            fi
          fi
        done
      done
    done

    # Waits for all graph generation processes for this dataset
    wait

    # --- Evaluation step for each method in parallel ---
    # evaluate_methods.py for PCA, t-SNE, and UMAP work individually
    {
      log_file="logs/eval_pca_${dataset_stem}.log"
      if [ ! -s "$log_file" ] || ! grep -q "evaluation completed successfully" "$log_file"; then
        python GraphEvaluation/evaluate_methods.py \
          --dataset_path "$dataset_path" \
          --reference_folder "./GeneratedGraphs/Consistency" \
          --comparison_folder "./GeneratedGraphs/PCA" \
          --method "PCA" \
          --output_dir "./EvaluationResults" \
          --n_jobs 1 > "$log_file" 2>&1
      fi
      echo "PCA evaluation done for $dataset_stem" >> logs/success.log
    } &

    {
      log_file="logs/eval_tsne_${dataset_stem}.log"
      if [ ! -s "$log_file" ] || ! grep -q "evaluation completed successfully" "$log_file"; then
        python GraphEvaluation/evaluate_methods.py \
          --dataset_path "$dataset_path" \
          --reference_folder "./GeneratedGraphs/Consistency" \
          --comparison_folder "./GeneratedGraphs/TSNE" \
          --method "TSNE" \
          --output_dir "./EvaluationResults" \
          --n_jobs 30 > "$log_file" 2>&1
      fi
      echo "TSNE evaluation done for $dataset_stem" >> logs/success.log
    } &

    {
      log_file="logs/eval_umap_${dataset_stem}.log"
      if [ ! -s "$log_file" ] || ! grep -q "evaluation completed successfully" "$log_file"; then
        python GraphEvaluation/evaluate_methods.py \
          --dataset_path "$dataset_path" \
          --reference_folder "./GeneratedGraphs/Consistency" \
          --comparison_folder "./GeneratedGraphs/UMAP" \
          --method "UMAP" \
          --output_dir "./EvaluationResults" \
          --n_jobs 90 > "$log_file" 2>&1
      fi
      echo "UMAP evaluation done for $dataset_stem" >> logs/success.log
    } &

    # Waits for the three evaluations to finish
    wait

    echo "Process for dataset $dataset_stem completed successfully!" >> logs/success.log
  } &

  # Throttles the outer loop as well.
  job_count=$((job_count + 1))
  if (( job_count >= MAX_PARALLEL_JOBS )); then
    wait
    job_count=0
  fi
done

# Ensures all datasets complete before finishing script
wait

echo "All processes completed successfully!" >> logs/success.log
