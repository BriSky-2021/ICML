import os
import glob
import json
import re
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR_DEFAULT = "paper/benchmark/analysis/100years/results"


def find_all_result_files_for_algo(algorithm_name: str, results_dir: str) -> List[str]:
    """
    Find all result files for an algorithm in the given directory.

    Naming (must match evaluation script):
        {algo}_budget_constrained_eval[_bfX_Y]_YYYYMMDD_HHMMSS.json
    """
    pattern = os.path.join(results_dir, f"{algorithm_name}_budget_constrained_eval*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No result files for algorithm {algorithm_name} in {results_dir}, "
            f"pattern: {os.path.basename(pattern)}"
        )
    print(f"[Info] Found {len(files)} result file(s) for {algorithm_name}")
    return files


def parse_budget_factor_from_filename(filename: str) -> float:
    """
    Parse budget_factor from filename.

    Convention:
      - No _bfX_Y -> budget_factor=1.0
      - With _bfX_Y, '_' in X_Y is decimal point, e.g.:
            *_bf0_5_*.json -> 0.5, *_bf2_0_*.json -> 2.0
    """
    base = os.path.basename(filename)
    # Match e.g. algo_budget_constrained_eval_bf0_5_20260127_123456.json -> 0.5
    # Use strict regex: bf<int> or bf<int>_<frac> to avoid matching date/time.
    m = re.search(r"_bf([0-9]+(?:_[0-9]+)?)_", base)
    if not m:
        return 1.0
    raw = m.group(1)  # e.g. "0_5" or "0_25"
    try:
        value = float(raw.replace("_", "."))
        return value
    except ValueError:
        return 1.0


def load_result(filepath: str) -> Dict:
    """Load a single result file; return aggregated result dict (first element if list)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Result file {filepath} is an empty list")
        return data[0]
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError(f"Result file {filepath} top-level is neither list nor dict")


def extract_time_series(result: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a single result extract:
      - years: year array
      - avg_health_per_year: average health per year (over all episodes)
      - cumulative_cost_per_year: cumulative average cost per year
    """
    episode_metrics = result.get("episode_metrics", [])
    if not episode_metrics:
        raise ValueError("Result has no episode_metrics, cannot do per-year analysis")

    max_years = max(len(ep.get("yearly_avg_health", [])) for ep in episode_metrics)
    years = np.arange(1, max_years + 1, dtype=int)

    avg_health_per_year: List[float] = []
    avg_cost_per_year: List[float] = []

    for year_idx in range(max_years):
        health_vals = []
        cost_vals = []
        for ep in episode_metrics:
            yearly_health = ep.get("yearly_avg_health", [])
            yearly_costs = ep.get("yearly_costs", [])
            if year_idx < len(yearly_health):
                health_vals.append(yearly_health[year_idx])
            if year_idx < len(yearly_costs):
                cost_vals.append(yearly_costs[year_idx])

        avg_health_per_year.append(float(np.mean(health_vals)) if health_vals else 0.0)
        avg_cost_per_year.append(float(np.mean(cost_vals)) if cost_vals else 0.0)

    avg_health_per_year = np.asarray(avg_health_per_year, dtype=float)
    avg_cost_per_year = np.asarray(avg_cost_per_year, dtype=float)
    cumulative_cost_per_year = np.cumsum(avg_cost_per_year)

    return years, avg_health_per_year, cumulative_cost_per_year


def plot_health_vs_budget_factor(
    bf_series: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
    algo_name: str,
) -> None:
    """Health trajectory comparison across budget factors."""
    plt.figure(figsize=(10, 6))
    for bf, (years, health, _) in sorted(bf_series.items(), key=lambda x: x[0]):
        label = f"bf={bf:g}"
        plt.plot(years, health, marker="o", linewidth=2, label=label)

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average health level", fontsize=12)
    plt.title(f"Health trajectory vs budget factor ({algo_name})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[Info] Health trajectory comparison saved to: {output_path}")
    plt.close()


def plot_cumulative_cost_vs_budget_factor(
    bf_series: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
    algo_name: str,
) -> None:
    """Cumulative cost trajectory comparison across budget factors."""
    plt.figure(figsize=(10, 6))
    for bf, (years, _, cum_cost) in sorted(bf_series.items(), key=lambda x: x[0]):
        label = f"bf={bf:g}"
        plt.plot(years, cum_cost, marker="o", linewidth=2, label=label)

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Cumulative cost", fontsize=12)
    plt.title(f"Cumulative spending vs budget factor ({algo_name})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[Info] Cumulative cost comparison saved to: {output_path}")
    plt.close()


def plot_efficiency_vs_budget_factor(
    bf_series: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
    algo_name: str,
) -> None:
    """
    Efficiency comparison across budget factors.

    Same composite score as in analysis_polish (lower is better):
        score = max(health_start - health_end, 0) * total_cost
    """
    bfs = []
    scores = []

    for bf, (_, health, cum_cost) in sorted(bf_series.items(), key=lambda x: x[0]):
        if health.size == 0 or cum_cost.size == 0:
            score = 0.0
        else:
            health_drop = max(float(health[0] - health[-1]), 0.0)
            total_cost = float(cum_cost[-1])
            if total_cost <= 0.0:
                score = 0.0
            else:
                score = health_drop * total_cost
        bfs.append(bf)
        scores.append(score)

    x = np.arange(len(bfs))

    plt.figure(figsize=(8, 6))
    plt.bar(x, scores, color="steelblue", alpha=0.8)

    plt.xticks(x, [f"{bf:g}" for bf in bfs])
    plt.xlabel("Budget factor", fontsize=12)
    plt.ylabel("Score (health drop × total cost)", fontsize=12)
    plt.title(f"Cost-health score vs budget factor ({algo_name})", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[Info] Cost-health efficiency comparison saved to: {output_path}")
    plt.close()


def plot_cost_and_final_health_vs_budget_factor(
    bf_series: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
    algo_name: str,
) -> None:
    """
    On one figure: bar chart of total cost vs budget factor, line of final average health vs budget factor.
    """
    bfs: List[float] = []
    total_costs: List[float] = []
    final_healths: List[float] = []

    for bf, (_, health, cum_cost) in sorted(bf_series.items(), key=lambda x: x[0]):
        if health.size == 0 or cum_cost.size == 0:
            bfs.append(bf)
            total_costs.append(0.0)
            final_healths.append(0.0)
        else:
            bfs.append(bf)
            total_costs.append(float(cum_cost[-1]))
            final_healths.append(float(health[-1]))

    x = np.arange(len(bfs))

    # Type1 serif font (e.g. STIXGeneral), larger font size; fallback to default if missing
    plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans"]
    base_fontsize = 16

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart: total cost (fill #C4D9ED, same edge)
    bar = ax1.bar(
        x,
        total_costs,
        width=0.5,  # narrower bars
        color="#C4D9ED",
        edgecolor="#C4D9ED",
        linewidth=1.0,
        alpha=0.9,
        label="Total cost",
    )
    ax1.set_xlabel("Budget factor", fontsize=base_fontsize, color="black")
    ax1.set_ylabel("Total cost", fontsize=base_fontsize, color="black")
    ax1.tick_params(axis="y", labelsize=base_fontsize - 2, colors="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{bf:g}" for bf in bfs], fontsize=base_fontsize - 2, color="black")

    # Line: final health (shared x, right y-axis)
    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        final_healths,
        color="#1F6EB5",
        marker="o",
        markersize=7,
        linewidth=2.5,
        label="Final health",
    )[0]
    ax2.set_ylabel("Final average health", fontsize=base_fontsize, color="black")
    ax2.tick_params(axis="y", labelsize=base_fontsize - 2, colors="black")

    # Axes: black, thicker spines and ticks
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.8)
        ax.tick_params(width=1.5, colors="black")

    # Grid below bars
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, color="gray", zorder=0)
    for rect in bar:
        rect.set_zorder(2)

    # Combined legend (bar + line), inside upper left
    handles = [bar, line]
    labels = ["Total cost", "Final health"]
    ax1.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=base_fontsize - 2,
        frameon=False,
    )

    plt.title(
        f"Total cost & final health vs budget factor ({algo_name})",
        fontsize=base_fontsize + 2,
        color="black",
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[Info] Cost & final health comparison saved to: {output_path}")
    plt.close()


def main():
    """
    CLI example: compare one algorithm across budget factors, e.g.:

        python -m paper.benchmark.analysis.100years.budget_analysis_polish --algorithm cql

    Optional: --results_dir, --output_prefix (auto includes algo name and timestamp).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compare one algorithm across budget factors")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="qmix_cql",# discrete_bc
        help="Algorithm name to analyze, e.g. cql",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR_DEFAULT,
        help=f"Results directory (default: {RESULTS_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Output figure filename prefix (default: auto with algo name and timestamp)",
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    algo = args.algorithm
    os.makedirs(results_dir, exist_ok=True)

    # Collect results for this algorithm across budget factors
    files = find_all_result_files_for_algo(algo, results_dir)
    bf_series: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for fp in files:
        bf = parse_budget_factor_from_filename(fp)
        result = load_result(fp)
        years, avg_health, cum_cost = extract_time_series(result)
        bf_series[bf] = (years, avg_health, cum_cost)

    if not bf_series:
        print(f"[Warn] No budget-factor data parsed from files for: {algo}")
        return

    # Print key metrics per budget factor
    print(f"\n========== Summary for algorithm: {algo} ==========")
    for bf, (_, health, cum_cost) in sorted(bf_series.items(), key=lambda x: x[0]):
        if health.size == 0 or cum_cost.size == 0:
            print(f"- bf={bf:g}: no valid data")
            continue
        h0 = float(health[0])
        hT = float(health[-1])
        total_cost = float(cum_cost[-1])
        health_drop = max(h0 - hT, 0.0)
        score = health_drop * total_cost if total_cost > 0 else 0.0
        print(
            f"- bf={bf:g}: initial_health = {h0:.4f}, "
            f"final_health = {hT:.4f}, "
            f"total_cost = ${total_cost:.2f}, "
            f"health_drop = {health_drop:.4f}, "
            f"score = {score:.4f}"
        )

    # Unified output prefix
    if args.output_prefix is not None:
        prefix = os.path.join(results_dir, args.output_prefix)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(results_dir, f"{algo}_bf_compare_{ts}")

    health_fig_path = prefix + "_health_trajectory.png"
    cost_fig_path = prefix + "_cumulative_cost.png"
    eff_fig_path = prefix + "_cost_efficiency.png"
    combo_fig_path = prefix + "_cost_and_final_health.png"

    plot_health_vs_budget_factor(bf_series, health_fig_path, algo)
    plot_cumulative_cost_vs_budget_factor(bf_series, cost_fig_path, algo)
    plot_efficiency_vs_budget_factor(bf_series, eff_fig_path, algo)
    plot_cost_and_final_health_vs_budget_factor(bf_series, combo_fig_path, algo)


if __name__ == "__main__":
    main()

