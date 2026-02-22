#!/usr/bin/env bash
# Generate action analysis figure from budget-constrained eval results.
# Budget sensitivity and 100-year figures are produced by 100years.sh / budget_sensity.sh.
# Run from repo root. Expects eval results in paper/benchmark/budget_constrained_results/.
# Usage: bash paper/benchmark/scripts/draw.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

python3 paper/benchmark/flows/action_analysis.py
