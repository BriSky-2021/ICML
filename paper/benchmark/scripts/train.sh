#!/usr/bin/env bash
# Train algorithms (Multi_Task_Run_v5) with multiple seeds. Run from repo root.
# Usage:
#   bash paper/benchmark/scripts/train.sh
#       # run all algorithms × all seeds (default device 0)
#   bash paper/benchmark/scripts/train.sh onestep
#       # run algorithm onestep for all seeds
#   bash paper/benchmark/scripts/train.sh onestep 0
#       # run onestep for all seeds on GPU 0
#   bash paper/benchmark/scripts/train.sh onestep 0 42
#       # single run: onestep, GPU 0, seed 42

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-paper/benchmark/flows/config.yaml}"
DEVICE_ID="${DEVICE_ID:-0}"

# Algorithms to train (comma-separated)
Algorithms="qmix_cql,discrete_bc,iqlcql_marl,onestep,cql,multitask_offline_cpq,cdt,multitask_bc,random_osrl"
Algorithms_heuristic="cql_heuristic,multitask_cpq_heuristic"
# Seeds for multi-seed runs
SEEDS="${SEEDS:-42 1024 2026}"

run_one() {
  local algo="$1"
  local dev="$2"
  local seed="$3"
  echo "========== Training: algo=$algo device_id=$dev seed=$seed =========="
  python3 paper/benchmark/flows/Multi_Task_Run_v5.py "$algo" "$dev" "$seed"
}

if [[ $# -ge 3 ]]; then
  # Single run: algo device_id seed
  run_one "$1" "$2" "$3"
elif [[ $# -eq 2 ]]; then
  # One algorithm, one device, all seeds
  for seed in $SEEDS; do
    run_one "$1" "$2" "$seed"
  done
elif [[ $# -eq 1 ]]; then
  # One algorithm, default device, all seeds
  for seed in $SEEDS; do
    run_one "$1" "$DEVICE_ID" "$seed"
  done
else
  # All algorithms × all seeds
  ALL_ALGOS="${Algorithms},${Algorithms_heuristic}"
  for algo in ${ALL_ALGOS//,/ }; do
    for seed in $SEEDS; do
      run_one "$algo" "$DEVICE_ID" "$seed"
    done
  done
fi

echo "Done."
