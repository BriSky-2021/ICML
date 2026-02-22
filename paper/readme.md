# InfraRL: A Benchmark for Constrained Resource Allocation in Large-Scale Infrastructure Asset Management

This directory contains the dataset pipeline and benchmark code for the paper. All scripts are intended to be run from the **repository root**.

## Abstract

Optimizing maintenance strategies for large-scale infrastructure is a critical sequential decision-making problem, exemplified by the high-stakes domain of bridge management. While Reinforcement Learning (RL) offers a theoretical framework for such problems, practical deployment necessitates offline constrained RL—learning policies solely from static historical datasets under rigid budgetary limits without dangerous on-policy exploration. However, current research is hindered by benchmarks that fail to capture the confluence of distributional shift and hard constraints typical of real-world assets. We introduce **InfraRL**, a high-fidelity benchmark that uses bridge maintenance as a rigorous testbed for general infrastructure asset management challenges. Constructed from the U.S. National Bridge Inventory, InfraRL defines a rigorous offline task for optimizing maintenance strategies under hard budgetary constraints. We benchmark a diverse suite of baselines, ranging from industry-standard heuristics to SOTA single-agent and multi-agent offline RL algorithms. Through a comprehensive evaluation protocol, we analyze performance across structural utility, constraint adherence, and behavioral fidelity, revealing critical trade-offs between safety and long-term efficiency.

---

## Recommended workflow (script order)

After data is ready (`get_data.sh`), run the benchmark in this order:

1. **train** → train all algorithms (multi-seed)
2. **eval** → budget-constrained evaluation (writes results used by draw and FQE)
3. **fqe** → FQE batch evaluation (uses trained models and test buffer)
4. **100years** → 100-year simulation + budget factor sensitivity + analysis/figures
5. **draw** → action analysis figure (uses eval results; budget sensitivity and 100-year figures are already produced by step 4)

```bash
bash paper/dataset/scripts/get_data.sh    # once
bash paper/benchmark/scripts/train.sh
bash paper/benchmark/scripts/eval.sh
bash paper/benchmark/scripts/fqe.sh
bash paper/benchmark/scripts/100years.sh
bash paper/benchmark/scripts/draw.sh
```

---

## Structure

```
paper/
├── readme.md                 # This file
├── dataset/                  # Data download and processing
│   ├── download/            # NBI raw data download (FHWA) and extract
│   ├── process/             # Clean, region split, transition matrices, episode generation
│   ├── data/                # Processed outputs (episodes, regions, transition_metrics)
│   └── scripts/
│       └── get_data.sh      # One-shot: download + full processing pipeline
└── benchmark/               # Training and evaluation
    ├── flows/               # Main scripts (train, eval, FQE, draw) and config
    ├── algos/               # Algorithm implementations
    ├── analysis/            # 100-year simulation and analysis
    └── scripts/
        ├── train.sh         # Train algorithms (multi-seed)
        ├── eval.sh          # Budget-constrained evaluation
        ├── fqe.sh           # FQE batch evaluation
        ├── draw.sh          # Report and plots
        └── 100years.sh      # 100-year simulation
```

---

## Installation

We recommend **Python 3.8** and **PyTorch 2.4.1 with CUDA 12**. The following steps assume a Linux environment; adjust for Windows/macOS if needed.

### 1. Python 3.8

Ensure Python 3.8 is available:

```bash
python3 --version   # should be 3.8.x
```

Using a virtual environment is recommended:

```bash
python3.8 -m venv venv
source venv/bin/activate   # Linux/macOS; on Windows: venv\Scripts\activate
```

### 2. PyTorch 2.4.1 (CUDA 12)

Install PyTorch with CUDA 12 support first. For **Python 3.8** and **CUDA 12.1**:

```bash
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

- For **CUDA 12.4**: use `cu124` instead of `cu121` in the index URL.
- For **CPU-only** (no GPU):  
  `pip install torch==2.4.1`

Verify:

```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 3. Other dependencies

From the repository root, install the remaining packages:

```bash
pip install -r paper/requirements.txt
```

Or install manually:

```bash
pip install "numpy>=1.20,<2" "pandas>=1.3" "PyYAML>=5.4" "matplotlib>=3.4" "seaborn>=0.11" "scipy>=1.7" "tqdm>=4.60" "openpyxl>=3.0" "requests>=2.26"
```

**`requirements.txt` vs `requirements-frozen.txt`**

- **`paper/requirements.txt`** uses version ranges (e.g. `numpy>=1.20,<2`). Use it for a first install; it does not pin exact versions.
- **`paper/requirements-frozen.txt`** lists the exact versions we tested (e.g. `numpy==1.24.4`). For strict reproducibility, install with:
  ```bash
  pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121   # or your PyTorch variant
  pip install -r paper/requirements-frozen.txt
  ```
- To regenerate `requirements-frozen.txt` from your current environment (run from repo root with the paper env activated):
  ```bash
  pip freeze | grep -iE "^(torch|numpy|pandas|pyyaml|matplotlib|seaborn|scipy|tqdm|openpyxl|requests)=" > paper/requirements-frozen.txt
  ```

### 5. Optional

- **CUDA**: Omit `+cu121` and use CPU-only PyTorch if no GPU; training and evaluation will be slower.
- **Reproducibility**: Scripts set seeds where applicable; for strict reproducibility use fixed `SEEDS` and `DEVICE_ID` and the same PyTorch/CUDA version.

---

## 1. Data Pipeline

From the repo root, run the full data pipeline (download NBI data, clean, build regions, transition matrices, and episode buffers):

```bash
bash paper/dataset/scripts/get_data.sh
```

This will:

1. Download raw NBI bridge data (FHWA) for California (1992–2023) into `paper/dataset/download/NBIDATA/`.
2. Merge and extract to `paper/dataset/download/output-1992-2023.xlsx`.
3. Clean and verify → `paper/dataset/data/processed/cleaned_bridge_data_verified.csv`.
4. Build regions → `paper/dataset/data/regions/` (regions.json, region_data/, connectivity_data.pkl).
5. Build transition matrices → `paper/dataset/data/transition_metrics/`.
6. Generate train/test episodes and buffers → `paper/dataset/data/episodes/` (train_buffer.pt, test_buffer.pt, env info, etc.).

**Note:** Step 1 requires network access and may take a while. The benchmark expects the episode buffers and env info under `paper/dataset/data/episodes/` (paths are set in `paper/benchmark/flows/config.yaml`).

---

## 2. Benchmark: Training

Train all configured algorithms with multiple seeds (default seeds: 42, 1024, 2026):

```bash
bash paper/benchmark/scripts/train.sh
```

- **One algorithm, all seeds:**  
  `bash paper/benchmark/scripts/train.sh onestep`
- **One algorithm, GPU 0, all seeds:**  
  `bash paper/benchmark/scripts/train.sh onestep 0`
- **Single run (algorithm, device_id, seed):**  
  `bash paper/benchmark/scripts/train.sh onestep 0 42`

Override seeds or device via environment variables:

```bash
SEEDS="42 1024" DEVICE_ID=1 bash paper/benchmark/scripts/train.sh
```

Models and training results are written under `paper/benchmark/saved_models/` and `paper/benchmark/training_results/` (paths used by the training script).

---

## 3. Benchmark: Budget-Constrained Evaluation

Run budget-constrained evaluation (BC-generated budget, then evaluate target algorithms):

```bash
bash paper/benchmark/scripts/eval.sh
```

Optional arguments (e.g. device, target algorithms):

```bash
bash paper/benchmark/scripts/eval.sh --device_id 0 --target_algorithms onestep cql multitask_bc
```

Defaults: test buffer `paper/dataset/data/episodes/test_buffer.pt`, config `paper/benchmark/flows/config.yaml`, output `paper/benchmark/budget_constrained_results/`.

---

## 4. Benchmark: FQE (Fitted Q Evaluation)

Run FQE in batch mode (latest model per algorithm, then FQE and save results per algorithm):

```bash
bash paper/benchmark/scripts/fqe.sh
```

Example with custom options:

```bash
bash paper/benchmark/scripts/fqe.sh --target_algorithms onestep cql multitask_offline_cpq --device_id 0
```

Results go to `paper/benchmark/fqe_results/` by default.

---

## 5. Benchmark: 100-Year Simulation

Run the 100-year long-horizon simulation and budget factor sensitivity (6 algorithms × factors 0.25, 0.5, 1.0, 2.0, 4.0), then run analysis and budget-sensitivity figures:

```bash
bash paper/benchmark/scripts/100years.sh
```

Output directory default: `paper/benchmark/analysis/100years/results/`. Budget sensitivity and 100-year plots are generated inside this step (via `analysis_polish.py` and `budget_analysis_polish.py`).

---

## 6. Benchmark: Draw (action analysis)

Generate the **action analysis** figure from budget-constrained evaluation results. Run after **eval** (and optionally after 100years). Budget sensitivity and 100-year figures are produced by `100years.sh`, not by this script.

```bash
bash paper/benchmark/scripts/draw.sh
```

Reads from `paper/benchmark/budget_constrained_results/` (eval output) and writes the action distribution figure there (or as configured in `action_analysis.py`).

---

## Configuration

- **Benchmark config:** `paper/benchmark/flows/config.yaml`  
  Data paths, training hyperparameters, algorithm list, and hardware settings. Data paths point to `paper/dataset/data/episodes/` by default.
- **Dataset:** No separate config file; paths are fixed in the dataset scripts and `get_data.sh`.

---

## Outputs Summary

| Step        | Main outputs |
|------------|--------------|
| get_data   | `paper/dataset/data/episodes/*.pt`, `*_env_info.json`; `regions/`; `transition_metrics/`; `processed/*.csv` |
| train      | `paper/benchmark/saved_models/`, `paper/benchmark/training_results/`, `paper/benchmark/metrics_results/` |
| eval       | `paper/benchmark/budget_constrained_results/` |
| fqe        | `paper/benchmark/fqe_results/` |
| 100years   | `paper/benchmark/analysis/100years/results/` (includes budget sensitivity and 100-year figures) |
| draw       | Action analysis figure (reads/writes under `paper/benchmark/budget_constrained_results/` or script default) |

All commands above are assumed to be run from the repository root.
