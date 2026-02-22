import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR_DEFAULT = "paper/benchmark/analysis/100years/results"

# Default algorithm list for comparison (edit as needed)
DEFAULT_ALGORITHMS: List[str] = [
    "cql",
    "cql_heuristic",
    "multitask_offline_cpq",
    "random_osrl",
    "onestep",
    "qmix_cql",
    "iqlcql_marl",
    "discrete_bc",
    "multitask_bc",
    #"cdt",
    #"onestep_heuristic",
    # To add more algorithms, append names here; names must match those used when saving results.
]


def find_latest_result_file(algorithm_name: str, results_dir: str) -> str:
    """
    Find the latest result file for an algorithm in the given results directory.
    
    File naming (must match evaluation script):
        {algo_name}_budget_constrained_eval_YYYYMMDD_HHMMSS.json
    """
    pattern = os.path.join(results_dir, f"{algorithm_name}_budget_constrained_eval_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No result file for algorithm {algorithm_name} in {results_dir}, "
                                f"pattern: {os.path.basename(pattern)}")
    # Latest by modification time
    latest = max(files, key=os.path.getmtime)
    print(f"[Info] Using result file for {algorithm_name}: {latest}")
    return latest


def load_result(filepath: str) -> Dict:
    """
    Load a single result file.
    Evaluation script saves a list; we take the first element as the aggregated result.
    """
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
    From a single algorithm result, extract:
      - average health per year (over all episodes)
      - average total cost per year (over all episodes)
      - cumulative average cost per year (cumsum of average cost)
    """
    episode_metrics = result.get("episode_metrics", [])
    if not episode_metrics:
        raise ValueError("Result has no episode_metrics, cannot do per-year analysis")

    # Longest year length
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

        # If no data for this year in any episode, use 0 to avoid errors
        if health_vals:
            avg_health_per_year.append(float(np.mean(health_vals)))
        else:
            avg_health_per_year.append(0.0)

        if cost_vals:
            avg_cost_per_year.append(float(np.mean(cost_vals)))
        else:
            avg_cost_per_year.append(0.0)

    avg_health_per_year = np.asarray(avg_health_per_year, dtype=float)
    avg_cost_per_year = np.asarray(avg_cost_per_year, dtype=float)
    cumulative_cost_per_year = np.cumsum(avg_cost_per_year)

    return years, avg_health_per_year, cumulative_cost_per_year


def plot_health_curves(
    algo_series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    """
    Plot average health vs year for all algorithms on one figure (conference-style colors/layout).
    """
    # Use Type1 font (e.g. STIXGeneral), fallback to default if missing
    plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans"]
    base_fontsize = 16

    # Color cycle: blue, dark blue, wine, orange, light blue, gray, gold, teal, purple
    color_cycle = [
        "#6AACD6",
        "#062F67",
        "#BB484F",
        "#FB9170",
        "#C4D9ED",
        "#4A4A4A",
        "#F4C15D",
        "#2F9E8F",  # teal
        "#7C5C9B",  # soft purple
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (algo, (years, health, _)) in enumerate(sorted(algo_series.items(), key=lambda x: x[0])):
        color = color_cycle[idx % len(color_cycle)]
        # Marker every 5 points
        markevery = 5
        ax.plot(
            years,
            health,
            marker="s",  # square
            markersize=3,
            markevery=markevery,
            linewidth=2.0,
            label=algo,
            color=color,
        )

    ax.set_xlabel("Year", fontsize=base_fontsize, color="black")
    ax.set_ylabel("Average health level", fontsize=base_fontsize, color="black")
    ax.tick_params(axis="both", labelsize=base_fontsize - 2, colors="black", width=1.4)

    # Axis style and grid
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="gray")

    ax.set_title(
        "Health trajectory over years (algorithm comparison)",
        fontsize=base_fontsize + 2,
        color="black",
    )

    # Legend inside upper right, no frame
    ax.legend(
        loc="upper right",
        fontsize=base_fontsize - 3,
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[Info] Health trajectory saved to: {output_path}")
    plt.close(fig)


def plot_cumulative_cost_curves(
    algo_series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    """
    Plot cumulative cost over years for all algorithms on one figure (conference-style).
    """
    plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans"]
    base_fontsize = 16

    color_cycle = [
        "#6AACD6",
        "#062F67",
        "#BB484F",
        "#FB9170",
        "#C4D9ED",
        "#4A4A4A",
        "#F4C15D",
        "#2F9E8F",
        "#7C5C9B",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (algo, (years, _, cum_cost)) in enumerate(sorted(algo_series.items(), key=lambda x: x[0])):
        color = color_cycle[idx % len(color_cycle)]
        markevery = 5
        ax.plot(
            years,
            cum_cost,
            marker="s",  # square
            markersize=3,
            markevery=markevery,
            linewidth=2.0,
            label=algo,
            color=color,
        )

    ax.set_xlabel("Year", fontsize=base_fontsize, color="black")
    ax.set_ylabel("Cumulative cost", fontsize=base_fontsize, color="black")
    ax.tick_params(axis="both", labelsize=base_fontsize - 2, colors="black", width=1.4)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="gray")

    ax.set_title(
        "Cumulative spending over years (algorithm comparison)",
        fontsize=base_fontsize + 2,
        color="black",
    )

    ax.legend(
        loc="upper left",
        fontsize=base_fontsize - 3,
        frameon=False,
        ncol=1,
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[Info] Cumulative cost curve saved to: {output_path}")
    plt.close(fig)


def plot_cost_efficiency_curves(
    algo_series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    """
    Bar chart of cost-health score per algorithm.

    Goal: less health drop and less total cost are better.
    Composite score (lower is better):

        score = max(health_start - health_end, 0) * total_cost

    - health_start - health_end > 0 is health drop (lower better)
    - total_cost is total spending over the horizon (lower better)
    - Product penalizes both high health drop and high spending.
    If total_cost <= 0, score = 0.
    """
    algos = []
    scores = []

    for algo, (_, health, cum_cost) in algo_series.items():
        if health.size == 0 or cum_cost.size == 0:
            score = 0.0
        else:
            # Health drop (negative treated as 0 = no drop or improvement)
            delta_health_raw = float(health[0] - health[-1])
            health_drop = max(delta_health_raw, 0.0)
            total_cost = float(cum_cost[-1])
            if total_cost <= 0.0:
                score = 0.0
            else:
                score = health_drop * total_cost
        algos.append(algo)
        scores.append(score)

    x = np.arange(len(algos))

    plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans"]
    base_fontsize = 16

    bar_colors = [
        "#6AACD6",
        "#062F67",
        "#BB484F",
        "#FB9170",
        "#C4D9ED",
        "#4A4A4A",
        "#F4C15D",
        "#2F9E8F",
        "#7C5C9B",
    ]
    colors = [bar_colors[i % len(bar_colors)] for i in range(len(algos))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, scores, color=colors, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=base_fontsize - 2, color="black")
    ax.set_ylabel("Score (health drop × total cost)", fontsize=base_fontsize, color="black")
    ax.tick_params(axis="y", labelsize=base_fontsize - 2, colors="black", width=1.4)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="gray", zorder=0)
    for b in bars:
        b.set_zorder(2)

    ax.set_title(
        "Cost-health score (lower is better)",
        fontsize=base_fontsize + 2,
        color="black",
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[Info] Cost-efficiency curve saved to: {output_path}")
    plt.close(fig)


def main():
    """
    CLI usage:

    - Compare cql and cql_heuristic:
        python -m paper.benchmark.analysis.100years.analysis_polish --algorithms cql cql_heuristic

    - Custom results dir:
        python -m paper.benchmark.analysis.100years.analysis_polish --algorithms cql --results_dir path/to/results
    """
    import argparse

    parser = argparse.ArgumentParser(description="100-year simulation result analysis and visualization")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Algorithm names to analyze, e.g. cql cql_heuristic; default: DEFAULT_ALGORITHMS",
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
        help="Output figure filename prefix (default: auto with algorithm names and timestamp)",
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    algo_series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for algo in args.algorithms:
        filepath = find_latest_result_file(algo, results_dir)
        result = load_result(filepath)
        years, avg_health, cum_cost = extract_time_series(result)
        algo_series[algo] = (years, avg_health, cum_cost)

    # Print key metrics per algorithm: initial health, final health, total cost
    print("\n========== Summary per algorithm ==========")
    for algo, (_, health, cum_cost) in algo_series.items():
        if health.size == 0 or cum_cost.size == 0:
            print(f"- {algo}: no valid data")
            continue
        health_start = float(health[0])
        health_end = float(health[-1])
        total_cost = float(cum_cost[-1])
        print(
            f"- {algo}: "
            f"initial_health = {health_start:.4f}, "
            f"final_health = {health_end:.4f}, "
            f"total_cost = ${total_cost:.2f}"
        )

    # Unified output prefix
    if args.output_prefix is not None:
        prefix = os.path.join(results_dir, args.output_prefix)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_tag = "_".join(args.algorithms)
        prefix = os.path.join(results_dir, f"{algo_tag}_{ts}")

    health_fig_path = prefix + "_health_trajectory.png"
    cost_fig_path = prefix + "_cumulative_cost.png"
    efficiency_fig_path = prefix + "_cost_efficiency.png"

    plot_health_curves(algo_series, health_fig_path)
    plot_cumulative_cost_curves(algo_series, cost_fig_path)
    plot_cost_efficiency_curves(algo_series, efficiency_fig_path)


if __name__ == "__main__":
    main()

