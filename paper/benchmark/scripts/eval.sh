#!/usr/bin/env bash
# Budget-constrained evaluation (BC-based budget, then evaluate target algorithms).
# Run from repo root. By default evaluates all algorithms from TARGET_ALGORITHMS.
# Usage:
#   bash paper/benchmark/scripts/eval.sh
#   bash paper/benchmark/scripts/eval.sh --device_id 0

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-paper/benchmark/flows/config.yaml}"
TEST_BUFFER="${TEST_BUFFER:-paper/dataset/data/episodes/test_buffer.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/benchmark/budget_constrained_results}"
DEVICE_ID="${DEVICE_ID:-0}"

# All algorithms from evaluate_with_budget_bc_v4_no_limit_three.py TARGET_ALGORITHMS (lines 51-66)
TARGET_ALGORITHMS="qmix_cql iqlcql_marl multitask_offline_cpq random_osrl onestep cql discrete_bc multitask_bc onestep_heuristic cql_heuristic multitask_cpq_heuristic cdt heuristic"

python3 paper/benchmark/flows/evaluate_with_budget_bc_v4_no_limit_three.py \
  --test_buffer "$TEST_BUFFER" \
  --config "$CONFIG" \
  --device_id "$DEVICE_ID" \
  --output_dir "$OUTPUT_DIR" \
  --target_algorithms $TARGET_ALGORITHMS \
  "$@"
