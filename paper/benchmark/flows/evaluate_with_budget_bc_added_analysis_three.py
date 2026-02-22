import os
import json
import glob
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_algorithm_name(filename):
    """
    从文件名中提取算法名称
    格式: {algorithm_name}_budget_constrained_eval_{timestamp}.json
    """
    basename = os.path.basename(filename)
    if '_budget_constrained_eval_' in basename:
        algo_name = basename.split('_budget_constrained_eval_')[0]
        return algo_name
    return None


def find_latest_files(results_dir, k=3):
    """
    找到每个算法最新的 k 个 JSON 文件
    
    Returns:
        dict: {algorithm_name: [file_path1, file_path2, ...]}  # by时间从新到旧
    """
    pattern = os.path.join(results_dir, '*_budget_constrained_eval_*.json')
    all_files = glob.glob(pattern)
    
    # 只处理文件名中包含 "latest" 的文件
    all_files = [f for f in all_files if 'latest' in os.path.basename(f).lower()]
    
    # by「评估结果中的 algorithm_name 字段」分组，
    # if不存在该字段则退回用文件名解析的算法名
    algorithm_files = defaultdict(list)
    for file_path in all_files:
        algo_name = None
        try:
            result_data = load_results_from_json(file_path)
            if isinstance(result_data, dict):
                algo_name = result_data.get('algorithm_name', None)
        except Exception:
            algo_name = None
        
        if not algo_name:
            algo_name = extract_algorithm_name(file_path)
        
        if algo_name:
            algorithm_files[algo_name].append(file_path)
    
    # for每个算法，找到最新的 k 个 文件（按修改时间）
    latest_files = {}
    for algo_name, files in algorithm_files.items():
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        latest_k = files_sorted[: max(int(k), 0)]
        latest_files[algo_name] = latest_k
        if latest_k:
            print(f"找到 {algo_name} 的最新 {len(latest_k)} 个文件:")
            for idx, fp in enumerate(latest_k, start=1):
                print(f"  [{idx}] {os.path.basename(fp)}")
    
    return latest_files


def load_results_from_json(file_path):
    """
    从JSON文件加载结果
    
    Returns:
        list: 结果对象列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ifdata是列表，返回第一个元素（通常只有一个结果）
    if isinstance(data, list):
        return data[0] if len(data) > 0 else None
    return data


def generate_comparison_table(results_dir='paper/benchmark/budget_constrained_results'):
    """
    生成综合评价表格
    
    Args:
        results_dir: 结果文件目录
    
    Returns:
        pd.DataFrame: 包含所有算法评价指标的DataFrame
    """
    print(f"扫描目录: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"错误: 目录不存在: {results_dir}")
        return None
    
    # 找到每个算法的最新多个文件（默认取3个）
    latest_files = find_latest_files(results_dir, k=3)
    
    if not latest_files:
        print("错误: 未找到任何结果文件")
        return None
    
    # 收集所有算法的结果（对每个算法，基于最新的多个评估文件计算三个指标的均值和方差）
    results = []
    
    for algo_name, file_paths in latest_files.items():
        if not file_paths:
            continue
        try:
            budget_list = []
            improve_list = []
            unit_cost_benefit_1000_list = []
            display_name = algo_name

            for file_path in file_paths:
                result_data = load_results_from_json(file_path)

                if result_data is None:
                    print(f"警告: {algo_name} 的文件 {os.path.basename(file_path)} 为空，跳过该文件")
                    continue

                # use algorithm_name 字段（如果存在），否则使用从文件名提取的名称
                display_name = result_data.get('algorithm_name', algo_name)

                # -------- 1) Budget Ratio --------
                budget_utilization = float(result_data.get('budget_utilization', 0.0))
                budget_list.append(budget_utilization)

                # -------- 2) Improve vs (total_health_gain_vs_history) --------
                improve_vs = float(result_data.get('total_health_gain_vs_history', 0.0))
                improve_list.append(improve_vs)

                # -------- 3) 单位成本收益 * 1000 --------
                total_health_gain_vs_nothing = float(result_data.get('total_health_gain_vs_nothing', 0.0))
                mean_step_cost = float(result_data.get('mean_step_cost', 0.0))
                num_steps = float(result_data.get('num_steps', 0.0))

                if mean_step_cost > 0 and num_steps > 0:
                    unit_cost_benefit = total_health_gain_vs_nothing / mean_step_cost / num_steps
                else:
                    unit_cost_benefit = 0.0
                unit_cost_benefit_1000 = unit_cost_benefit * 1000.0
                unit_cost_benefit_1000_list.append(unit_cost_benefit_1000)

            if not budget_list:
                print(f"警告: {algo_name} 没有有效的结果，跳过")
                continue

            # convert to数组，计算均值和方差（只对这三个值计算方差）
            budget_arr = np.asarray(budget_list, dtype=np.float64)
            improve_arr = np.asarray(improve_list, dtype=np.float64)
            unit_cost_benefit_1000_arr = np.asarray(unit_cost_benefit_1000_list, dtype=np.float64)

            result_row = {
                '算法名称': display_name,
                # Budget Ratio
                'budget_utilization': float(np.mean(budget_arr)),
                'budget_utilization_var': float(np.var(budget_arr)),
                # Improve vs
                'total_health_gain_vs_history': float(np.mean(improve_arr)),
                'total_health_gain_vs_history_var': float(np.var(improve_arr)),
                # 单位成本收益 * 1000
                '单位成本收益_1000': float(np.mean(unit_cost_benefit_1000_arr)),
                '单位成本收益_1000_var': float(np.var(unit_cost_benefit_1000_arr)),
            }

            results.append(result_row)
            print(f"成功加载: {display_name}（基于 {len(budget_list)} 个评估文件）")

        except Exception as e:
            print(f"错误: 加载 {algo_name} 的文件失败: {e}")
            continue
    
    if not results:
        print("错误: 没有成功加载任何结果")
        return None
    
    # createDataFrame
    df = pd.DataFrame(results)
    
    # by算法名称排序（可选）
    df = df.sort_values('算法名称')
    
    return df


def compute_pareto_front(df, obj1_col='budget_utilization', obj2_col='total_health_gain_vs_history'):
    """
    计算帕累托前沿
    
    Args:
        df: DataFrame containing the data
        obj1_col: 第一个目标列名（越大越好）
        obj2_col: 第二个目标列名（越大越好）
    
    Returns:
        pd.DataFrame: 帕累托前沿的点
        np.ndarray: 布尔数组，标记哪些点是帕累托最优的
    """
    if df is None or df.empty:
        return None, None
    
    # get目标值
    obj1 = df[obj1_col].values
    obj2 = df[obj2_col].values
    
    # init帕累托最优标记
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    
    # for于每个点，检查是否有其他点在两个目标上都优于或等于它
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # if点j在两个目标上都优于或等于点i，则点i不是帕累托最优
            if obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and (obj1[j] > obj1[i] or obj2[j] > obj2[i]):
                is_pareto[i] = False
                break
    
    # 提取帕累托前沿点
    pareto_df = df[is_pareto].copy()
    
    # byobj1排序（用于绘图）
    pareto_df = pareto_df.sort_values(obj1_col)
    
    print(f"\n帕累托前沿包含 {len(pareto_df)} 个点:")
    for idx, row in pareto_df.iterrows():
        print(f"  - {row['算法名称']}: budget_utilization={row[obj1_col]:.4f}, "
              f"total_health_gain_vs_history={row[obj2_col]:.2f}")
    
    return pareto_df, is_pareto


def plot_pareto_frontier(df, is_pareto, output_file=None, 
                         obj1_col='budget_utilization', 
                         obj2_col='total_health_gain_vs_history'):
    """
    绘制帕累托前沿图
    
    Args:
        df: DataFrame containing all algorithms
        is_pareto: 布尔数组，标记哪些点是帕累托最优的
        output_file: 输出图片文件路径（可选）
        obj1_col: 第一个目标列名
        obj2_col: 第二个目标列名
    """
    if df is None or df.empty:
        print("没有数据可绘图")
        return
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # get所有点的坐标
    x_all = df[obj1_col].values
    y_all = df[obj2_col].values
    
    # 绘制所有点
    ax.scatter(x_all, y_all, c='lightgray', s=100, alpha=0.6, 
               label='All Algorithms', marker='o', edgecolors='gray', linewidths=1)
    
    # 绘制帕累托前沿点
    if is_pareto is not None:
        x_pareto = df[obj1_col].values[is_pareto]
        y_pareto = df[obj2_col].values[is_pareto]
        
        # byx坐标排序以绘制帕累托前沿线
        pareto_indices = np.argsort(x_pareto)
        x_pareto_sorted = x_pareto[pareto_indices]
        y_pareto_sorted = y_pareto[pareto_indices]
        
        # 绘制帕累托前沿线
        ax.plot(x_pareto_sorted, y_pareto_sorted, 'r-', linewidth=2, 
                alpha=0.7, label='Pareto Frontier', zorder=2)
        
        # 绘制帕累托前沿点
        ax.scatter(x_pareto, y_pareto, c='red', s=150, alpha=0.8, 
                   label='Pareto Optimal', marker='*', edgecolors='darkred', 
                   linewidths=2, zorder=3)
        
        # 标注帕累托前沿点的算法名称
        pareto_df = df[is_pareto]
        for idx, row in pareto_df.iterrows():
            ax.annotate(row['算法名称'], 
                       (row[obj1_col], row[obj2_col]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 标注所有算法名称（可选，如果点不多）
    for idx, row in df.iterrows():
        if is_pareto is None or not is_pareto[idx]:
            ax.annotate(row['算法名称'], 
                       (row[obj1_col], row[obj2_col]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.6)
    
    # 设置标签和标题
    ax.set_xlabel('Budget Utilization', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Health Gain vs History', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier Analysis: Budget Utilization vs Health Gain', 
                fontsize=14, fontweight='bold', pad=20)
    
    # add网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # add图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # save图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n帕累托前沿图已保存到: {output_file}")
    else:
        # if没有指定输出文件，显示图片
        plt.show()
    
    plt.close()


def print_table(df, output_file=None):
    """
    打印表格
    
    Args:
        df: DataFrame
        output_file: 可选，输出文件路径
    """
    if df is None or df.empty:
        print("没有数据可显示")
        return
    
    # build用于展示的 DataFrame，把均值和方差合成为 X±Y 的字符串
    df_display = df.copy()
    metric_configs = [
        # (均值列名, 方差列名)
        ('budget_utilization', 'budget_utilization_var'),
        ('total_health_gain_vs_history', 'total_health_gain_vs_history_var'),
        ('单位成本收益_1000', '单位成本收益_1000_var'),
    ]
    for mean_col, var_col in metric_configs:
        if mean_col in df_display.columns and var_col in df_display.columns:
            means = df_display[mean_col].astype(float)
            vars_ = df_display[var_col].astype(float)
            stds = np.sqrt(vars_)
            # direct覆盖均值列为 “均值±标准差”，只删除方差列
            df_display[mean_col] = [f"{m:.6f}±{s:.6f}" for m, s in zip(means, stds)]
            df_display = df_display.drop(columns=[var_col])
    
    print("\n" + "="*80)
    print("算法综合评价表（均值±标准差）")
    print("="*80)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}' if abs(x) < 1 else f'{x:.2f}')
    
    print(df_display.to_string(index=False))
    print("="*80)
    
    # if指定了输出文件，保存为CSV（也使用 X±Y 的形式）
    if output_file:
        df_display.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
    
    # 同时保存为Markdown表格
    if output_file:
        md_file = output_file.replace('.csv', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# algorithm综合评价表（均值±标准差）\n\n")
            f.write(df_display.to_markdown(index=False))
        print(f"Markdown表格已保存到: {md_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成算法综合评价表格和帕累托分析')
    parser.add_argument('--results_dir', type=str, 
                       default='paper/benchmark/budget_constrained_results',
                       help='结果文件目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出CSV文件路径（可选）')
    parser.add_argument('--plot_output', type=str, default=None,
                       help='输出图片文件路径（可选，如 pareto_frontier.png）')
    
    args = parser.parse_args()
    
    # 生成表格
    df = generate_comparison_table(args.results_dir)
    
    # print表格
    if df is not None:
        output_file = args.output
        if output_file is None:
            # default输出文件名
            dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(args.results_dir, f'comparison_table_{dt_str}.csv')
        
        print_table(df, output_file)
        
        # compute帕累托前沿
        pareto_df, is_pareto = compute_pareto_front(df)
        
        # 绘制帕累托前沿图
        plot_output = args.plot_output
        if plot_output is None:
            # default输出文件名
            dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_output = os.path.join(args.results_dir, f'pareto_frontier_{dt_str}.png')
        
        plot_pareto_frontier(df, is_pareto, plot_output)
    else:
        print("生成表格失败")


if __name__ == "__main__":
    main()