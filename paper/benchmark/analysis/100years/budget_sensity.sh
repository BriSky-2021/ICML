#!/usr/bin/env bash
# Budget factor sensitivity: for each of 6 algorithms, run 100-year at 0.25, 0.5, 1.0, 2.0, 4.0, then budget_analysis_polish.
# Run from repo root: bash paper/benchmark/analysis/100years/budget_sensity.sh
# Optional: DEVICE_ID=0 bash paper/benchmark/analysis/100years/budget_sensity.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-paper/benchmark/flows/config.yaml}"
TEST_BUFFER="${TEST_BUFFER:-paper/dataset/data/episodes/test_buffer.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/benchmark/analysis/100years/results}"
DEVICE_ID="${DEVICE_ID:-3}"

ALGORITHMS="cql cql_heuristic multitask_cpq qmix_cql discrete_bc iqlcql_marl"
BUDGET_FACTORS="0.25 0.5 1.0 2.0 4.0"

for algo in $ALGORITHMS; do
  echo "---------- Algorithm: $algo ----------"
  for bf in $BUDGET_FACTORS; do
    python3 paper/benchmark/analysis/100years/evaluate_with_budget_bc_100_years.py \
      --test_buffer "$TEST_BUFFER" \
      --config "$CONFIG" \
      --device_id "$DEVICE_ID" \
      --output_dir "$OUTPUT_DIR" \
      --budget_factor "$bf" \
      --target_algorithms "$algo"
  done
  python3 paper/benchmark/analysis/100years/budget_analysis_polish.py \
    --algorithm "$algo" \
    --results_dir "$OUTPUT_DIR"
done

echo "Done. Results in $OUTPUT_DIR"
