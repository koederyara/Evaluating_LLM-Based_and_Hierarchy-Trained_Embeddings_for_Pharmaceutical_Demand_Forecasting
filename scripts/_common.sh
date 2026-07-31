#!/usr/bin/env bash
# Shared helpers for the four numbered stages. Sourced, never executed directly.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Prefer the project venv, fall back to whatever python is on PATH.
if   [[ -x ".venv/Scripts/python.exe" ]]; then PY=".venv/Scripts/python.exe"   # Windows
elif [[ -x ".venv/bin/python" ]];         then PY=".venv/bin/python"           # Unix
else PY="python"; fi

export PYTHONUTF8=1     # rho/delta prints would crash under Windows cp1252
export MPLBACKEND=Agg   # headless: plt.show() becomes a no-op

STAMP="$(date +%Y%m%d_%H%M%S)"
FAILURES=0
STAGE_START=$(date +%s)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# run <label> <command...> - logs, and keeps going on failure so one broken step
# does not discard the hours already spent on the others.
run() {
  local label="$1"; shift
  local log="$LOG_DIR/${label}.log"
  echo "----------------------------------------------------------------------"
  echo "[$(ts)] >>> $label"
  echo "    cmd: $*"
  echo "    log: $log"
  local start; start=$(date +%s)

  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}

  local elapsed=$(( $(date +%s) - start ))
  printf "[%s] <<< exit %d in %dh%02dm%02ds\n\n" "$(ts)" "$status" \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  [[ $status -ne 0 ]] && FAILURES=$((FAILURES + 1))
  return 0
}

# require <path> <hint> - abort early if an earlier stage did not run.
require() {
  if [[ ! -e "$1" ]]; then
    echo "ERROR: missing prerequisite: $1"
    echo "       $2"
    exit 1
  fi
}

finish() {
  local total=$(( $(date +%s) - STAGE_START ))
  echo "======================================================================"
  printf "[%s] DONE in %dh%02dm  (%d failure(s))\n" "$(ts)" \
    $((total / 3600)) $(((total % 3600) / 60)) "$FAILURES"
  echo "logs: $LOG_DIR"
  exit $(( FAILURES > 0 ? 1 : 0 ))
}
