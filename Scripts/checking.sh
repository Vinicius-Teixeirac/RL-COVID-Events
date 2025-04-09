# Executes a command if the log file is missing or does not contain the success marker.

execute_if_not_done() {
  local log_file="$1"
  local marker="${2:-}"
  shift 2
  if [ ! -s "$log_file" ] || { [ -n "$marker" ] && ! grep -Fq "$marker" "$log_file"; }; then
    if ! "$@" > "$log_file" 2>&1; then
      log_msg "ERROR" "Command failed: $*"
      return 1
    fi
  fi
}

export -f execute_if_not_done