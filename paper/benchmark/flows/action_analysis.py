import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Eval output dir (same as eval.sh --output_dir default)
RESULTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "budget_constrained_results",
))

# data集中原始动作分布（用户提供的数值）
DATASET_ACTION_DISTRIBUTION = {
    "no_action": 99.08,
    "minor_repair": 0.41,
    "major_repair": 0.42,
    "replacement": 0.10,
}

ACTION_ID_TO_NAME = {
    "0": "no_action",
    "1": "minor_repair",
    "2": "major_repair",
    "3": "replacement",
}

ACTION_NAMES = ["no_action", "minor_repair", "major_repair", "replacement"]

# at这里定义要分析的算法前缀列表，按需修改
# 名称需要与结果文件前缀一致，如：
#   cql -> cql_budget_constrained_eval_YYYYMMDD_HHMMSS.json
#   cql_heuristic -> cql_heuristic_budget_constrained_eval_YYYYMMDD_HHMMSS.json
#   onestep_heuristic -> onestep_heuristic_budget_constrained_eval_YYYYMMDD_HHMMSS.json
ALGOS: List[str] = [
    "cdt",
    "cql",
    "cql_heuristic",
    "onestep",
    "onestep_heuristic",
    "multitask_bc",
    "random_osrl",
    "multitask_offline_cpq",
    "qmix_cql",
    "iqlcql_marl",
    "discrete_bc"
]


def find_latest_result_file(results_dir: str, algo_prefix: str) -> str:
    """在结果目录中找到给定算法前缀的最新结果文件。"""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"结果目录不存在: {results_dir}")

    candidates: List[str] = []
    for fname in os.listdir(results_dir):
        if (
            fname.startswith(f"{algo_prefix}_budget_constrained_eval_")
            and fname.endswith(".json")
        ):
            candidates.append(os.path.join(results_dir, fname))

    if not candidates:
        raise FileNotFoundError(
            f"在目录 {results_dir} 中未找到算法 '{algo_prefix}' 的结果文件。"
        )

    # 文件名中包含 YYYYMMDD_HHMMSS，按文件名排序即可近似按时间排序
    candidates.sort()
    return candidates[-1]


def load_action_distributions(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """从结果 JSON 文件中读取原始与最终动作分布。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        record = data[-1]
    else:
        record = data

    orig = record.get("original_action_distribution", {})
    final = record.get("final_action_distribution", {})

    # 只保留 0-3 四种动作，并补齐缺失项为 0
    def normalize(dist: Dict[str, float]) -> Dict[str, float]:
        norm: Dict[str, float] = {}
        for aid, name in ACTION_ID_TO_NAME.items():
            value = float(dist.get(aid, 0.0))
            norm[name] = value
        return norm

    return normalize(orig), normalize(final)


def print_action_table(
    algo_to_distributions: Dict[str, Tuple[Dict[str, float], Dict[str, float]]]
) -> None:
    """在控制台打印各算法的动作分布表格。"""
    header = (
        f"{'Algorithm':<30}"
        f"{'no_action(orig)':>18}{'no_action(final)':>18}"
        f"{'minor_repair(orig)':>20}{'minor_repair(final)':>20}"
        f"{'major_repair(orig)':>20}{'major_repair(final)':>20}"
        f"{'replacement(orig)':>20}{'replacement(final)':>20}"
    )
    print(header)
    print("-" * len(header))

    for algo, (orig, final) in algo_to_distributions.items():
        row = (
            f"{algo:<30}"
            f"{orig['no_action']:>18.2f}{final['no_action']:>18.2f}"
            f"{orig['minor_repair']:>20.2f}{final['minor_repair']:>20.2f}"
            f"{orig['major_repair']:>20.2f}{final['major_repair']:>20.2f}"
            f"{orig['replacement']:>20.2f}{final['replacement']:>20.2f}"
        )
        print(row)

    print("\n原始数据集动作分布：")
    print(
        "  no_action {:.2f}%  minor_repair {:.2f}%  major_repair {:.2f}%  replacement {:.2f}%".format(
            DATASET_ACTION_DISTRIBUTION["no_action"],
            DATASET_ACTION_DISTRIBUTION["minor_repair"],
            DATASET_ACTION_DISTRIBUTION["major_repair"],
            DATASET_ACTION_DISTRIBUTION["replacement"],
        )
    )


def plot_action_distributions(
    algo_to_distributions: Dict[str, Tuple[Dict[str, float], Dict[str, float]]]
) -> None:
    """绘制并保存动作分布对比“单一表格图”：左侧为原始分布列，右侧为限制后分布列，最后一行为原始数据集分布。"""
    algos = list(algo_to_distributions.keys())
    num_algos = len(algos)

    # 一个 axes + 一个表格，列为 [action, 0,1,2,3 | 0,1,2,3]
    fig, ax = plt.subplots(1, 1, figsize=(4 + 0.5 * num_algos, 4))

    # 颜色映射（浅蓝色）
    cmap = plt.cm.Blues
    vmin, vmax = 0.0, 100.0

    # 显示用算法名称映射
    algo_display_name = {
        "multitask_bc": "single_bc",
        "random_osrl": "random",
        "multitask_offline_cpq": "cpq",
        "discrete_bc": "multi_bc",
    }

    # first构造完整表格内容：首行为表头，其余行为算法 + dataset
    header_row = ["action", "0", "1", "2", "3", "0", "1", "2", "3"]

    body_rows: List[List[str]] = []
    body_values: List[List[float]] = []

    for algo in algos:
        orig, final = algo_to_distributions[algo]
        display_name = algo_display_name.get(algo, algo)
        row_vals = [orig[a] for a in ACTION_NAMES] + [final[a] for a in ACTION_NAMES]
        row = [display_name] + [f"{v:.2f}" for v in row_vals]
        body_rows.append(row)
        body_values.append(row_vals)

    dataset_vals_left = [DATASET_ACTION_DISTRIBUTION[a] for a in ACTION_NAMES]
    dataset_vals_right = dataset_vals_left  # 没有“限制后”版本，这里复用方便位置对齐
    dataset_vals = dataset_vals_left + dataset_vals_right
    dataset_row = ["dataset"] + [f"{v:.2f}" for v in dataset_vals]
    body_rows.append(dataset_row)
    body_values.append(dataset_vals)

    # 最终 cellText：表头 + 内容
    cell_text: List[List[str]] = [header_row] + body_rows

    table = ax.table(
        cellText=cell_text,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.6, 1.1)

    # 扩充最左侧（算法名称）这一列的单元格宽度，避免名称被截断
    total_rows = len(cell_text)
    # if存在至少一行内容，就以第 1 行第 0 列的宽度为基准
    if total_rows > 1:
        base_width = table[(1, 0)].get_width()
        new_width = base_width * 2.8
        for i in range(total_rows):
            cell = table[(i, 0)]
            cell.set_width(new_width)

    # according to数值设置背景色（浅），只对数据单元格着色（从第 1 行、第 1 列开始）
    n_rows = len(cell_text) - 1  # not算表头
    n_cols = 8  # 8 个数值列
    for i in range(n_rows):
        for j in range(n_cols):
            val = body_values[i][j]
            norm_v = (val - vmin) / (vmax - vmin)
            norm_v = max(0.0, min(1.0, norm_v))
            norm_v = 0.2 + 0.5 * norm_v
            color = cmap(norm_v)
            # +1 是跳过表头行，+1 是跳过算法名这一列
            table[(i + 1, j + 1)].set_facecolor(color)

    # 表头加粗
    for j in range(len(header_row)):
        header_cell = table[(0, j)]
        header_cell.get_text().set_fontweight("bold")

    ax.set_axis_off()

    # 上方文字说明：左半 Original，右半 Constrained
    fig.text(0.25, 0.95, "Original", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.text(0.75, 0.95, "Constrained", ha="center", va="center", fontsize=10, fontweight="bold")

    # save到结果目录而不是展示
    output_path = os.path.join(RESULTS_DIR, "action_distribution_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    algo_to_distributions: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}

    for algo in ALGOS:
        try:
            path = find_latest_result_file(RESULTS_DIR, algo)
        except FileNotFoundError as e:
            print(f"[警告] {e}")
            continue

        orig, final = load_action_distributions(path)
        algo_to_distributions[algo] = (orig, final)

    if not algo_to_distributions:
        print("未成功加载任何算法的结果，请检查 --algos 参数与结果目录。")
        return

    print_action_table(algo_to_distributions)
    plot_action_distributions(algo_to_distributions)


if __name__ == "__main__":
    main()
