#!/bin/bash

mkdir -p logs

dataset_dir="DataPreparation/UsageDatasets"

datasets=("$dataset_dir"/*.pkl)

ppxty_values=(5 10 20 30 40 50)
min_dist_values=(0.0 0.1 0.25 0.5 0.8 0.99)
desired_dimensionality=2
rnd_state=42

MAX_PARALLEL_JOBS=10

generate_hyperparams() {
  local num_rows=$1
  local num_parts=10
  local min_value=2
  local max_value=$(echo "sqrt($num_rows)" | bc)

  local ks=()
  local kg=()
  local kt=()
  local kpca=()
  local ktsne=()
  local kumap=()
  local umap_neighbors_values=()

  local interval=$(( (max_value + 1 - min_value) / (num_parts - 1) ))

  for (( i=0; i<num_parts; i++ )); do
    value=$(( min_value + i * interval ))
    ks+=("$value")
    kg+=("$value")
    kt+=("$value")
    kpca+=("$value")
    ktsne+=("$value")
    kumap+=("$value")
    umap_neighbors_values+=("$value")
  done

  ks[-1]=$max_value
  kg[-1]=$max_value
  kt[-1]=$max_value
  kpca[-1]=$max_value
  ktsne[-1]=$max_value
  kumap[-1]=$max_value
  umap_neighbors_values[-1]=$max_value

  echo "${ks[@]}" "${kg[@]}" "${kt[@]}" "${kpca[@]}" "${ktsne[@]}" "${kumap[@]}" "${umap_neighbors_values[@]}"
}


job_count=10

for dataset_name in "${datasets[@]}"; do
  {
    python DataPreparation/generate_representations.py "$dataset_name"

    num_rows=$(python -c "import pandas as pd; print(len(pd.read_pickle('$dataset_name')))")
    read -r -a hyperparams < <(generate_hyperparams "$num_rows")

    ks_values=("${hyperparams[@]:0:100}")
    kg_values=("${hyperparams[@]:0:100}")
    kt_values=("${hyperparams[@]:0:100}")
    kpca_values=("${hyperparams[@]:0:100}")
    ktsne_values=("${hyperparams[@]:0:100}")
    kumap_values=("${hyperparams[@]:0:100}")
    umap_neighbors_values=("${hyperparams[@]:0:100}")

    for ks in "${ks_values[@]}"; do
      for kg in "${kg_values[@]}"; do
        for kt in "${kt_values[@]}"; do
          log_file="logs/consistency_$(basename "$dataset_name" .pkl)_${ks}_${kg}_${kt}.log"
          if [[ ! -s "$log_file" || "$(cat "$log_file")" != "success" ]]; then
            python generate_consistency_graphs.py $ks $kg $kt $dataset_name > "$log_file" 2>&1 &
            job_count=$((job_count + 1))
            [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0
          fi
        done
      done
    done

    for kpca in "${kpca_values[@]}"; do
      log_file="logs/pca_$(basename "$dataset_name" .pkl)_${kpca}.log"
      if [[ ! -s "$log_file" || "$(cat "$log_file")" != "success" ]]; then
        python generate_pca_graphs.py $kpca $desired_dimensionality $dataset_name > "$log_file" 2>&1 &
        job_count=$((job_count + 1))
        [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0
      fi
    done

    for ktsne in "${ktsne_values[@]}"; do
      for ppxty in "${ppxty_values[@]}"; do
        log_file="logs/tsne_$(basename "$dataset_name" .pkl)_${ktsne}_${ppxty}.log"
        
        if [[ ! -s "$log_file" || "$(tail -n 1 "$log_file")" != "success" ]]; then
          python generate_tsne_graphs.py $ktsne $ppxty $desired_dimensionality $rnd_state $dataset_name > "$log_file" 2>&1 &
          job_count=$((job_count + 1))
          [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0
        fi
      done
    done

    for kumap in "${kumap_values[@]}"; do
      for umap_neighbors in "${umap_neighbors_values[@]}"; do
        for min_dist in "${min_dist_values[@]}"; do
          log_file="logs/umap_$(basename "$dataset_name" .pkl)_${kumap}_${umap_neighbors}_${min_dist}.log"

          if [[ ! -s "$log_file" || "$(tail -n 1 "$log_file")" != "success" ]]; then
            python generate_umap_graphs.py $kumap $umap_neighbors $desired_dimensionality $rnd_state $min_dist $dataset_name > "$log_file" 2>&1 &
            job_count=$((job_count + 1))
            [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0
          fi
        done
      done
    done



  } &
  
  job_count=$((job_count + 1))
  [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0

done

wait




    # for kumap in "${kumap_values[@]}"; do
    #   for umap_neighbors in "${umap_neighbors_values[@]}"; do
    #     for min_dist in "${min_dist_values[@]}"; do
    #       log_file="logs/CUMAP/consistency_umap_$(basename "$dataset_name" .pkl)_${kumap}_${umap_neighbors}_${min_dist}.log"
    #       python generate_df_cumap.py $kumap $umap_neighbors $min_dist $desired_dimensionality $rnd_state $dataset_name > "$log_file" 2>&1 &
    #       job_count=$((job_count + 1))
    #       [[ $job_count -ge $MAX_PARALLEL_JOBS ]] && wait && job_count=0
    #     done
    #   done
    # done