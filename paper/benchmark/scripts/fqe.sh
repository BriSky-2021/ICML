#!/usr/bin/env bash
# FQE (Fitted Q Evaluation) in batch mode: find latest models per algorithm, run FQE, save per-algorithm results.
# Run from repo root. By default runs FQE for all target algorithms listed below.
# Usage:
#   bash paper/benchmark/scripts/fqe.sh
#   bash paper/benchmark/scripts/fqe.sh --device_id 0

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-paper/benchmark/flows/config.yaml}"
TEST_BUFFER="${TEST_BUFFER:-paper/dataset/data/episodes/test_buffer.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-paper/benchmark/fqe_results}"
DEVICE_ID="${DEVICE_ID:-0}"

# Algorithms to run FQE on (aligned with eval / paper experiments)
TARGET_ALGORITHMS="qmix_cql iqlcql_marl multitask_offline_cpq random_osrl onestep cql discrete_bc multitask_bc onestep_heuristic cql_heuristic multitask_cpq_heuristic heuristic"

python3 paper/benchmark/flows/evaluate_fqe_v3.py --batch \
  --test_buffer "$TEST_BUFFER" \
  --config "$CONFIG" \
  --device_id "$DEVICE_ID" \
  --output_dir "$OUTPUT_DIR" \
  --target_algorithms $TARGET_ALGORITHMS \
  "$@"
