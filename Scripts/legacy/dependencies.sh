#!/bin/bash

check_dependencies() {
  local missing_deps=()
  for dep in python jupyter papermill parallel bc; do
    if ! type -p "$dep" &>/dev/null; then
      missing_deps+=("$dep")
    fi
  done
  if (( ${#missing_deps[@]} )); then
    echo "Error: Missing dependencies: ${missing_deps[*]}. Please install them before running the script." >&2
    exit 1
  fi
}

export -f check_dependencies