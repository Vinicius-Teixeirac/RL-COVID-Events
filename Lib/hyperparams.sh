#!/bin/bash

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
    umap_neighbors+=("$(value+1)")
  done
  echo "${ks[*]}"
  echo "${kg[*]}"
  echo "${kt[*]}"
  echo "${kpca[*]}"
  echo "${ktsne[*]}"
  echo "${kumap[*]}"
  echo "${umap_neighbors[*]}"
}
