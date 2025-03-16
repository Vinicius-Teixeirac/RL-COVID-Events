#!/bin/bash

log_msg() {
  local level="$1"
  shift
  local message="$*"
  if [ -n "${MAIN_LOG-}" ]; then
    {
      flock -x 200
      echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] - $message"
    } >> "$MAIN_LOG" 2>&1 200>"$LOCK_FILE"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] - $message" >&2
  fi
}

get_log_file() {
  local base_dir="$1"
  local prefix="$2"
  local dataset_stem="$3"
  shift 3
  local params="$*"
  echo "${base_dir}/${prefix}_${dataset_stem}_${params}.log"
}

export -f log_msg get_log_file
