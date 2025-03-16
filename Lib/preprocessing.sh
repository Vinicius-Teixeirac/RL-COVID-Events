#!/bin/bash

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

export -f run_preprocessing