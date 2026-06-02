#!/bin/bash

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
  
  local k_values=()
  local i value
  for (( i=0; i<num_parts; i++ )); do
    if (( i == num_parts - 1 )); then
      value="$max_value"
    else
      value=$(( min_value + i * interval ))
    fi
    k_values+=("$value")
  done
  
  echo "${k_values[*]}"
}

export -f generate_hyperparams