#!/usr/bin/env bash
# Download NBI bridge data and run the full paper/dataset processing pipeline.
# Run from the repository root: bash paper/dataset/scripts/get_data.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

DOWNLOAD_DIR="$REPO_ROOT/paper/dataset/download"
PROCESS_DIR="$REPO_ROOT/paper/dataset/process"
DATA_DIR="$REPO_ROOT/paper/dataset/data"

echo "=============================================="
echo "Paper dataset: download + full processing"
echo "REPO_ROOT=$REPO_ROOT"
echo "=============================================="

# ---------------------------------------------------------------------------
# Step 0: Create directories
# ---------------------------------------------------------------------------
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$DATA_DIR/processed"
mkdir -p "$DATA_DIR/regions/region_data"
mkdir -p "$DATA_DIR/transition_metrics"
mkdir -p "$DATA_DIR/episodes"

# ---------------------------------------------------------------------------
# Step 1: Download raw NBI data (FHWA) into paper/dataset/download/NBIDATA/
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Downloading raw NBI data (CA, 1992-2023)..."
cd "$DOWNLOAD_DIR"
python3 Downloadv1.py
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Step 2: Extract and merge to output-1992-2023.xlsx in download/
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Extracting and merging to Excel..."
cd "$DOWNLOAD_DIR"
python3 "ExtractData - V1.py"
cd "$REPO_ROOT"

if [[ ! -f "$DOWNLOAD_DIR/output-1992-2023.xlsx" ]]; then
  echo "Error: expected $DOWNLOAD_DIR/output-1992-2023.xlsx not found after extract."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Clean and verify bridge data -> data/processed/cleaned_bridge_data_verified.csv
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Cleaning and verifying bridge data..."
python3 "$PROCESS_DIR/data_processor.py"

if [[ ! -f "$DATA_DIR/processed/cleaned_bridge_data_verified.csv" ]]; then
  echo "Error: expected $DATA_DIR/processed/cleaned_bridge_data_verified.csv not found."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 4a: Build regions -> data/regions/regions.json, region_data/*, connectivity_data.pkl
# ---------------------------------------------------------------------------
echo ""
echo "[4a/5] Building regions (region splitter)..."
python3 "$PROCESS_DIR/region_splitter_udelta_4level.py"

# ---------------------------------------------------------------------------
# Step 4b: Build transition matrices -> data/transition_metrics/*.npy, *.npz
# ---------------------------------------------------------------------------
echo ""
echo "[4b/5] Building transition matrices..."
python3 "$PROCESS_DIR/transition_metrics_builder_copy_without_faling.py"

# ---------------------------------------------------------------------------
# Step 5: Generate episodes and buffers -> data/episodes/*.pt, *.json, *.pkl
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Generating episodes and train/test buffers..."
python3 "$PROCESS_DIR/episode_generator.py"

echo ""
echo "=============================================="
echo "Done. Outputs:"
echo "  - $DATA_DIR/processed/cleaned_bridge_data_verified.csv"
echo "  - $DATA_DIR/regions/regions.json, region_data/, connectivity_data.pkl"
echo "  - $DATA_DIR/transition_metrics/*.npy, all_transition_matrices.npz"
echo "  - $DATA_DIR/episodes/train_buffer.pt, test_buffer.pt, *_env_info.json"
echo "=============================================="
