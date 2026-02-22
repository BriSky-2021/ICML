#!/usr/bin/env bash
# 100-year simulation: (1) run 6 algorithms at default budget, then analysis_polish;
# (2) for each algorithm, run budget factors 0.25, 0.5, 1.0, 2.0, 4.0, then budget_analysis_polish.
# Run from repo root.
# Usage: bash paper/benchmark/scripts/100years.sh
#        DEVICE_ID=0 bash paper/benchmark/scripts/100years.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-paper/benchmark/flows/config.yaml}"
TEST_BUFFER="${TEST_BUFFER:-paper/dataset/data/episodes/test_buffer.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/benchmark/analysis/100years/results}"
DEVICE_ID="${DEVICE_ID:-0}"

# Six algorithms for 100-year experiments
ALGORITHMS="cql cql_heuristic multitask_cpq qmix_cql discrete_bc iqlcql_marl"
# Budget factors for sensitivity experiments (per-algorithm)
BUDGET_FACTORS="0.25 0.5 1.0 2.0 4.0"

run_100years() {
  local algo="$1"
  local bf="${2:-1.0}"
  python3 paper/benchmark/analysis/100years/evaluate_with_budget_bc_100_years.py \
    --test_buffer "$TEST_BUFFER" \
    --config "$CONFIG" \
    --device_id "$DEVICE_ID" \
    --output_dir "$OUTPUT_DIR" \
    --budget_factor "$bf" \
    --target_algorithms $algo
}

echo "=============================================="
echo "Part 1: 100-year run for 6 algorithms (budget_factor=1.0), then analysis_polish"
echo "=============================================="
run_100years "$ALGORITHMS" 1.0
python3 paper/benchmark/analysis/100years/analysis_polish.py \
  --results_dir "$OUTPUT_DIR" \
  --algorithms $ALGORITHMS

echo ""
echo "=============================================="
echo "Part 2: Budget factor sensitivity (0.25, 0.5, 1.0, 2.0, 4.0) per algorithm, then budget_analysis_polish"
echo "=============================================="
for algo in $ALGORITHMS; do
  echo "---------- Algorithm: $algo ----------"
  for bf in $BUDGET_FACTORS; do
    run_100years "$algo" "$bf"
  done
  python3 paper/benchmark/analysis/100years/budget_analysis_polish.py \
    --algorithm "$algo" \
    --results_dir "$OUTPUT_DIR"
done

echo "Done. Results in $OUTPUT_DIR"
