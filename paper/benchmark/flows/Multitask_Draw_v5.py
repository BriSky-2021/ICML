import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib.patches import Polygon
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

def load_training_histories(training_results_dir="training_results", 
                            multi_file_algos=None, 
                            num_files=3):
    """
    加载所有算法的训练历史数据。
    对于指定的算法，加载最近的 `num_files` 个历史记录以计算方差。
    
    Args:
        training_results_dir: 训练结果目录
        multi_file_algos: 一个算法名称列表，用于加载多个最近的文件。
        num_files: 为指定算法加载的最近文件数量。
    
    Returns:
        dict: {algorithm_name: [list_of_training_histories]}
    """
    # multi_file_algos 支持三种用法：
    # 1) None / []：仅加载每个算法最新 1 个文件（旧行为）
    # 2) list[str]：仅对列表中的算法加载最近 num_files 个文件
    # 3) "ALL"：对所有算法都加载最近 num_files 个文件（避免 onestep 这种“明明很多结果却只统计 1 次”）
    load_all_latest_n = (multi_file_algos == "ALL")
    if multi_file_algos is None:
        multi_file_algos = []

    if not os.path.exists(training_results_dir):
        print(f"Directory {training_results_dir} does not exist!")
        return {}
    
    training_histories = {}
    
    # get所有训练历史文件
    pattern = os.path.join(training_results_dir, "*_training_history_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No training history files found in directory {training_results_dir}!")
        return {}
    
    # by算法名称对所有文件进行分组
    algo_files = defaultdict(list)
    for file_path in files:
        filename = os.path.basename(file_path)
        # 解析文件名: algo_name_training_history_timestamp.json
        parts = filename.replace('_training_history_', '|').replace('.json', '').split('|')
        if len(parts) == 2:
            algo_name = parts[0]
            timestamp = parts[1]
            algo_files[algo_name].append((file_path, timestamp))
    
    # 为每个算法加载最新的N个文件
    for algo_name, file_list in algo_files.items():
        # by时间戳排序，最新的在前
        sorted_files = sorted(file_list, key=lambda x: x[1], reverse=True)
        
        if load_all_latest_n or (algo_name in multi_file_algos):
            files_to_load = sorted_files[:num_files]
            print(f"Algorithm {algo_name}: found {len(file_list)} files, will load latest {len(files_to_load)}.")
        else:
            files_to_load = sorted_files[:1]
        
        loaded_histories = []
        for file_path, timestamp in files_to_load:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_histories.append(json.load(f))
                print(f"  - Successfully loaded history for {algo_name} from timestamp {timestamp}")
            except Exception as e:
                print(f"  - Failed to load history for {algo_name} from {file_path}: {e}")
        
        if loaded_histories:
            training_histories[algo_name] = loaded_histories
            
    return training_histories

def calculate_improvement_percentages(final_data, 
                                    do_nothing_baseline=-0.0242,  # 什么都不做的基准健康变化
                                    historical_baseline=-0.0180,   # 历史策略的基准健康变化
                                    historical_improvement_vs_nothing=None):  # 历史相对于什么都不做的改善
    """
    Calculate improvement percentages and efficiency metrics
    
    Args:
        final_data: Final evaluation data dictionary
        do_nothing_baseline: Health change when doing nothing (negative deterioration)
        historical_baseline: Health change with historical policy (negative deterioration)
        historical_improvement_vs_nothing: Historical improvement vs doing nothing
    
    Returns:
        dict: Enhanced final data with improvement percentages
    """
    enhanced_data = {}
    
    # compute历史策略相对于什么都不做的改善
    if historical_improvement_vs_nothing is None:
        historical_improvement_vs_nothing = historical_baseline - do_nothing_baseline
    
    for algo_name, data in final_data.items():
        enhanced_metrics = data.copy()
        
        # 基础健康指标
        health_gain_vs_nothing = data['health_gain_vs_nothing']
        health_gain_vs_history = data['health_gain_vs_history']
        health_gain_absolute = data['health_gain_absolute']
        
        # compute相对于"什么都不做"的提升比例
        # 公式: (算法健康改善 - 什么都不做的健康变化) / |什么都不做的健康变化| * 100%
        if do_nothing_baseline != 0:
            improvement_vs_nothing_pct = (health_gain_vs_nothing / abs(do_nothing_baseline)) * 100
        else:
            improvement_vs_nothing_pct = 0.0
        
        # compute相对于"历史策略"的提升比例  
        # 公式: (算法健康改善 - 历史策略健康改善) / |历史策略健康变化| * 100%
        if historical_baseline != 0:
            improvement_vs_historical_pct = (health_gain_vs_history / abs(historical_baseline)) * 100
        else:
            improvement_vs_historical_pct = 0.0
        
        # compute每1000美元的健康提升值（基于相对于什么都不做的提升）
        #bridge_avg_cost = data.get('total_cost', 0) / data.get('total_evaluations', 1) if data.get('total_evaluations', 0) > 0 else data.get('total_cost', 0)
        bridge_avg_cost=data.get('bridge_avg_cost', 1000)

        if bridge_avg_cost > 0:
            health_improvement_per_1M_dollars = (health_gain_vs_nothing * 1000*1000) / bridge_avg_cost
        else:
            health_improvement_per_1M_dollars = 0.0
        
        # compute成本效益比（健康提升/预算使用比例）
        budget_ratio = data.get('budget_usage_ratio', 1.0)
        if budget_ratio > 0:
            cost_effectiveness_ratio = improvement_vs_nothing_pct / budget_ratio
        else:
            cost_effectiveness_ratio = 0.0
        
        # add新的计算指标
        enhanced_metrics.update({
            # 提升比例指标
            'improvement_vs_nothing_pct': improvement_vs_nothing_pct,
            'improvement_vs_historical_pct': improvement_vs_historical_pct,
            
            # 效率指标
            'health_improvement_per_1M_dollars': health_improvement_per_1M_dollars,
            'bridge_avg_cost': bridge_avg_cost,
            'cost_effectiveness_ratio': cost_effectiveness_ratio,
            
            # 基准值（用于参考）
            'do_nothing_baseline': do_nothing_baseline,
            'historical_baseline': historical_baseline,
            'historical_improvement_vs_nothing': historical_improvement_vs_nothing,
            
            # 综合效率评分（结合提升比例和成本效益）
            'efficiency_score': (improvement_vs_nothing_pct * 0.4 + 
                               improvement_vs_historical_pct * 0.3 + 
                               cost_effectiveness_ratio * 0.3)
        })
        
        enhanced_data[algo_name] = enhanced_metrics
    
    return enhanced_data

def extract_final_metrics(training_histories):
    """
    从训练历史中提取最终评估指标。
    如果为某个算法提供了多个历史记录，则计算平均值和标准差。
    
    Args:
        training_histories: 字典 {algorithm_name: [list_of_histories]}
    
    Returns:
        dict: {algorithm_name: {metric_name: value, metric_name_std: value, ...}}
    """
    # 此字典将存储每个算法每次运行的指标列表
    # 例如: {'algo1': [{'metricA': 10}, {'metricA': 12}], 'algo2': [{'metricA': 15}]}
    raw_final_metrics_lists = defaultdict(list)

    for algo_name, histories in training_histories.items():
        for history in histories:
            if 'eval_metrics' not in history or not history['eval_metrics']:
                print(f"Algorithm {algo_name}: history file has no evaluation data, skipping.")
                continue
            
            # get最后一个评估记录
            last_eval = history['eval_metrics'][-1]
            metrics = last_eval['metrics']
            
            # 提取桥梁维护分析的关键指标
            final_metrics = {
                'epoch': last_eval['epoch'],
                'behavioral_similarity': metrics.get('behavioral_similarity_mean', 0.0),
                'total_cost': metrics.get('mean_total_cost', 0.0),
                'budget_usage_ratio': metrics.get('budget_usage_ratio', 0.0),
                'violation_rate': metrics.get('violation_rate_mean', 0.0),
                'health_gain_absolute': metrics.get('bridge_avg_health_gain_absolute', 0.0),
                'health_gain_vs_history': metrics.get('bridge_avg_health_gain_vs_history', 0.0),
                'health_gain_vs_nothing': metrics.get('bridge_avg_health_gain_vs_nothing', 0.0),
                'health_gain_normalized': metrics.get('bridge_avg_health_gain_normalized', 0.0),
                'health_gain_per_1000_dollars': metrics.get('health_gain_per_1000_dollars', 0.0),
                'comprehensive_score': metrics.get('comprehensive_score', 0.0),
                'total_evaluations': metrics.get('total_evaluations', 0),
                'behavioral_similarity_r': metrics.get('behavioral_similarity_mean_r', None),
                'total_cost_r': metrics.get('mean_total_cost_r', None),
                'budget_usage_ratio_r': metrics.get('budget_usage_ratio_r', None),
                'health_gain_absolute_r': metrics.get('bridge_avg_health_gain_absolute_r', None),
                'health_gain_vs_history_r': metrics.get('bridge_avg_health_gain_vs_history_r', None),
                'health_gain_vs_nothing_r': metrics.get('bridge_avg_health_gain_vs_nothing_r', None),
                'comprehensive_score_r': metrics.get('comprehensive_score_r', None),
            }
            raw_final_metrics_lists[algo_name].append(final_metrics)

    # 为每次独立运行计算增强的改进/效率指标
    enhanced_metrics_lists = defaultdict(list)
    for algo_name, metrics_list in raw_final_metrics_lists.items():
        for metrics in metrics_list:
            # calculate_improvement_percentages 函数适用于字典的字典结构。
            # 我们包装单个指标字典以按原样使用该函数。
            temp_dict = {"temp_algo": metrics}
            enhanced_dict = calculate_improvement_percentages(temp_dict)
            enhanced_metrics_lists[algo_name].append(enhanced_dict["temp_algo"])

    # 聚合指标（平均值和标准差）以用于最终输出
    aggregated_data = {}
    for algo_name, enhanced_list in enhanced_metrics_lists.items():
        if not enhanced_list:
            print(f"Algorithm {algo_name}: No valid metrics found after processing.")
            continue

        # use pandas DataFrame 进行简单的聚合
        df = pd.DataFrame(enhanced_list)
        
        # get所有数值列的平均值和标准差
        # for于单次运行的算法，其标准差为NaN，用fillna(0)处理
        mean_metrics = df.mean().to_dict()
        std_metrics = df.std().fillna(0).to_dict()
        
        # will它们合并到一个字典中
        final_metrics_agg = {}
        for key in mean_metrics.keys():
            final_metrics_agg[key] = mean_metrics[key]
            final_metrics_agg[f"{key}_std"] = std_metrics[key]
            
        # 同时存储运行次数
        final_metrics_agg['num_runs'] = len(enhanced_list)
        
        aggregated_data[algo_name] = final_metrics_agg
        print(f"Algorithm {algo_name}: aggregated metrics from {len(enhanced_list)} run(s).")
    
    return aggregated_data

def calculate_pareto_frontier(x_vals, y_vals, maximize_x=False, maximize_y=True):
    """
    Calculate Pareto frontier points
    
    Args:
        x_vals: x values (budget usage ratio)
        y_vals: y values (health improvement percentage)
        maximize_x: whether to maximize x (False for budget - lower is better)
        maximize_y: whether to maximize y (True for health - higher is better)
    
    Returns:
        list: indices of points on Pareto frontier
    """
    points = list(zip(x_vals, y_vals))
    n_points = len(points)
    pareto_indices = []
    
    for i in range(n_points):
        is_pareto = True
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                x_better = (points[j][0] > points[i][0]) if maximize_x else (points[j][0] < points[i][0])
                y_better = (points[j][1] > points[i][1]) if maximize_y else (points[j][1] < points[i][1])
                
                x_equal = points[j][0] == points[i][0]
                y_equal = points[j][1] == points[i][1]
                
                # j dominates i if j is better in at least one dimension and not worse in any
                if (x_better and (y_better or y_equal)) or (y_better and (x_better or x_equal)):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices

def get_contrasting_text_color(background_color):
    """
    Get contrasting text color (black or white) based on background color brightness
    """
    if isinstance(background_color, str):
        # Convert hex to RGB if needed
        if background_color.startswith('#'):
            background_color = background_color[1:]
            rgb = tuple(int(background_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Named color - use black as default
            return 'white'
    else:
        # RGB tuple
        rgb = background_color[:3] if len(background_color) > 3 else background_color
        rgb = tuple(int(c * 255) if c <= 1 else int(c) for c in rgb)
    
    # Calculate brightness using relative luminance formula
    brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return 'black' if brightness > 0.5 else 'white'

def plot_pareto_analysis(final_data, save_path=None, figsize=(22, 14)):
    """
    使用正确的历史基线绘制改进百分比的帕累托分析图。
    为具有多次运行的算法添加误差棒。
    """
    if not final_data:
        print("No data to plot!")
        return
    
    # Set font support
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create subplots with better spacing
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], width_ratios=[3, 1, 1], 
                         hspace=0.35, wspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, :])  # Main Pareto plot spans full width
    ax_legend = fig.add_subplot(gs[1, 0])  # Algorithm legend
    ax_pareto_info = fig.add_subplot(gs[1, 1])  # Pareto info
    ax_table = fig.add_subplot(gs[2, :])   # Performance table spans full width
    
    fig.suptitle('Bridge Maintenance Strategy: Pareto Analysis with Improvement Percentages', 
                fontsize=20, fontweight='bold')
    
    # Prepare data
    algo_names = list(final_data.keys())
    budget_ratios = [d['budget_usage_ratio'] for d in final_data.values()]
    improvement_percentages = [d['improvement_vs_nothing_pct'] for d in final_data.values()]
    
    # compute历史基准点的正确值
    sample_data = next(iter(final_data.values()))
    historical_budget_ratio = 1.0
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    # Calculate Pareto frontier (minimize budget, maximize improvement percentage)
    pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_percentages, 
                                             maximize_x=False, maximize_y=True)
    
    # IMPORTANT: 不要用 zip(algo_names, colors) 这种写法，否则当算法数量 > 颜色数量时会被截断，导致“找不全/画不全”。
    # here按算法数量动态生成颜色，并对 marker 做循环使用，保证任意数量算法都能绘制出来。
    palette = sns.color_palette("tab20", n_colors=max(len(algo_names), 1))
    colors = [tuple(c) for c in palette]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '8', 'P', 'X', '1', '2', '3', '4']
    
    # Plot all algorithms on main plot
    for i, algo_name in enumerate(algo_names):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        size = 400 if i in pareto_indices else 200
        alpha = 1.0 if i in pareto_indices else 0.8
        
        budget_mean = budget_ratios[i]
        health_mean = improvement_percentages[i]
        data = final_data[algo_name]
        
        # Add error bars for algorithms with multiple runs
        if data.get('num_runs', 1) > 1:
            budget_std = data.get('budget_usage_ratio_std', 0)
            health_std = data.get('improvement_vs_nothing_pct_std', 0)
            if budget_std > 0 or health_std > 0:
                ax_main.errorbar(budget_mean, health_mean, xerr=budget_std, yerr=health_std,
                                 fmt='none', ecolor='gray', capsize=5, alpha=0.6, zorder=3)

        # Plot the point
        ax_main.scatter(budget_mean, health_mean, 
                        color=color, marker=marker, s=size, alpha=alpha,
                        edgecolors='black', linewidths=3 if i in pareto_indices else 2,
                        label=f'{i+1:2d}. {algo_name}', zorder=5)
        
        text_color = get_contrasting_text_color(color)
        ax_main.annotate(f'{i+1}', (budget_mean, health_mean), 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color=text_color, zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.1", facecolor=color, 
                                edgecolor='black', linewidth=1, alpha=0.9))
    
    # Add historical baseline point
    ax_main.scatter(historical_budget_ratio, historical_improvement_pct, 
                   color='red', s=600, marker='*', alpha=1.0,
                   edgecolors='darkred', linewidths=4, zorder=6)
    ax_main.annotate('H', (historical_budget_ratio, historical_improvement_pct), 
                    ha='center', va='center', fontsize=16, fontweight='bold', 
                    color='white', zorder=10,
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor='red', 
                            edgecolor='darkred', linewidth=2))
    
    # Draw Pareto frontier
    if len(pareto_indices) > 1:
        pareto_x = [budget_ratios[i] for i in pareto_indices]
        pareto_y = [improvement_percentages[i] for i in pareto_indices]
        pareto_points = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted, pareto_y_sorted = zip(*pareto_points)
        ax_main.plot(pareto_x_sorted, pareto_y_sorted, 'r-', linewidth=3, alpha=0.8,
                    label='Pareto Frontier', zorder=4)
        ax_main.scatter(pareto_x_sorted, pareto_y_sorted, 
                       facecolors='none', edgecolors='gold', s=500, linewidths=4, zorder=7)
    
    # Set labels and formatting for main plot
    ax_main.set_xlabel('Budget Usage Ratio (vs Historical)', fontsize=16)
    ax_main.set_ylabel('Health Improvement vs Do Nothing (%)', fontsize=16)
    ax_main.set_title('Pareto Analysis: Budget Efficiency vs Health Improvement Percentage', fontsize=18)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelsize=12)
    
    # Add quadrant lines and labels
    ax_main.axhline(y=historical_improvement_pct, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax_main.axvline(x=historical_budget_ratio, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    xlim, ylim = ax_main.get_xlim(), ax_main.get_ylim()
    ax_main.text(xlim[0] + (xlim[1] - xlim[0]) * 0.05, ylim[1] - (ylim[1] - ylim[0]) * 0.05, 'Better Health\nLower Cost', fontsize=12, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8, edgecolor='green'))
    ax_main.text(xlim[1] - (xlim[1] - xlim[0]) * 0.05, ylim[0] + (ylim[1] - ylim[0]) * 0.05, 'Worse Health\nHigher Cost', fontsize=12, ha='right', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8, edgecolor='red'))
    
    # Algorithm legend
    ax_legend.axis('off')
    legend_text = "ALGORITHM REFERENCE:\n" + "="*25 + "\n"
    mid_point = (len(algo_names) + 1) // 2
    for i in range(mid_point):
        line = f"{i+1:2d}. {algo_names[i][:20]:<20}"
        if i + mid_point < len(algo_names):
            line += f"  |  {i + mid_point + 1:2d}. {algo_names[i + mid_point][:20]}"
        legend_text += line + "\n"
    legend_text += f"\n H. Historical ({historical_improvement_pct:.2f}%)"
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Pareto frontier information
    ax_pareto_info.axis('off')
    pareto_text = f"PARETO FRONTIER:\n" + "="*20 + f"\n{len(pareto_indices)} algorithms optimal\n\nPareto Algorithms:\n"
    for idx in pareto_indices:
        pareto_text += f"{idx+1:2d}. {algo_names[idx][:15]}...\n" if len(algo_names[idx]) > 15 else f"{idx+1:2d}. {algo_names[idx]}\n"
    ax_pareto_info.text(0.05, 0.95, pareto_text, transform=ax_pareto_info.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor='gold', alpha=0.8))
    
    # Enhanced performance table
    ax_table.axis('off')
    performance_data = []
    for i, algo_name in enumerate(algo_names):
        data = final_data[algo_name]
        performance_data.append([
            f"{i+1:2d}", algo_name[:25], f"{data['budget_usage_ratio']:.3f}", f"{data['improvement_vs_nothing_pct']:.2f}%",
            f"{data['improvement_vs_historical_pct']:.2f}%", f"{data['efficiency_score']:.2f}", "★" if i in pareto_indices else ""
        ])
    performance_data.append(["H", "Historical Baseline", f"{historical_budget_ratio:.3f}", f"{historical_improvement_pct:.2f}%", "0.00%", "N/A", ""])
    performance_data_sorted = sorted(performance_data[:-1], key=lambda x: float(x[5]), reverse=True)
    performance_data_sorted.append(performance_data[-1])
    
    table = ax_table.table(cellText=performance_data_sorted, colLabels=['#', 'Algorithm', 'Budget Ratio', 'Improve vs Nothing', 'Improve vs History', 'Efficiency', 'Pareto'], cellLoc='center', loc='center', colWidths=[0.06, 0.3, 0.12, 0.15, 0.15, 0.1, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i, row in enumerate(performance_data_sorted):
        for j in range(len(row)):
            cell = table[(i+1, j)]
            if row[6] == "★": cell.set_facecolor('#FFD700'); cell.set_text_props(weight='bold')
            elif row[0] == "H": cell.set_facecolor('#FFCCCC'); cell.set_text_props(weight='bold')
            else: cell.set_facecolor('#F0F0F0')
    for j in range(7): table[(0, j)].set_facecolor('#4472C4'); table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Pareto analysis plot saved to: {save_path}")
    
    plt.show()
    
    return {'pareto_indices': pareto_indices, 'pareto_algorithms': [algo_names[i] for i in pareto_indices], 'performance_data': performance_data_sorted, 'algorithm_mapping': {i+1: name for i, name in enumerate(algo_names)}, 'historical_improvement_pct': historical_improvement_pct}

def plot_simplified_comparison(final_data, save_path=None, figsize=(16, 8)):
    """
    Plot simplified comparison with only 2 core subplots
    """
    if not final_data:
        print("No data to plot!")
        return
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Bridge Maintenance Strategy: Core Performance Analysis', 
                fontsize=16, fontweight='bold')
    
    algo_names = list(final_data.keys())
    
    # Enhanced colors (按算法数量动态生成，避免 zip 截断导致少画模型)
    palette = sns.color_palette("tab20", n_colors=max(len(algo_names), 1))
    colors = [tuple(c) for c in palette]
    
    # 1. Budget vs Improvement Percentage Scatter with correct historical baseline
    budget_ratios = [final_data[algo]['budget_usage_ratio'] for algo in algo_names]
    improvement_pcts = [final_data[algo]['improvement_vs_nothing_pct'] for algo in algo_names]
    
    # Calculate correct historical improvement percentage
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    for i, algo in enumerate(algo_names):
        color = colors[i % len(colors)]
        axes[0].scatter(budget_ratios[i], improvement_pcts[i], 
                       color=color, s=200, alpha=0.8, edgecolors='black', linewidth=2)
        
        text_color = get_contrasting_text_color(color)
        axes[0].annotate(f'{i+1}', (budget_ratios[i], improvement_pcts[i]), 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color=text_color,
                        bbox=dict(boxstyle="circle,pad=0.15", facecolor=color, 
                                edgecolor='black', linewidth=1, alpha=0.9))
    
    # Add historical baseline with correct value
    axes[0].scatter(1.0, historical_improvement_pct, color='red', s=300, marker='*', 
                   label=f'Historical ({historical_improvement_pct:.2f}%)', edgecolors='darkred', linewidths=3)
    axes[0].annotate('H', (1.0, historical_improvement_pct), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor='red', 
                            edgecolor='darkred', linewidth=2))
    
    axes[0].set_xlabel('Budget Usage Ratio (vs Historical)', fontsize=14)
    axes[0].set_ylabel('Health Improvement vs Do Nothing (%)', fontsize=14)
    axes[0].set_title('Budget Efficiency vs Health Improvement', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add quadrant lines
    axes[0].axhline(y=historical_improvement_pct, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    axes[0].axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # 2. Cost Effectiveness Analysis (Health Improvement per $1000)
    cost_effectiveness = [final_data[algo]['health_improvement_per_1M_dollars'] for algo in algo_names]
    total_costs = [final_data[algo]['total_cost'] for algo in algo_names]
    
    # Create efficiency plot with cost as bubble size
    max_cost = max(total_costs) if total_costs else 1
    normalized_sizes = [(cost / max_cost) * 500 + 150 for cost in total_costs]
    
    for i, (algo, size) in enumerate(zip(algo_names, normalized_sizes)):
        color = colors[i % len(colors)]
        axes[1].scatter(i, cost_effectiveness[i], 
                       color=color, s=size, alpha=0.7, edgecolors='black', linewidth=2)
        
        text_color = get_contrasting_text_color(color)
        axes[1].annotate(f'{i+1}', (i, cost_effectiveness[i]), 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color=text_color,
                        bbox=dict(boxstyle="circle,pad=0.1", facecolor=color, 
                                edgecolor='black', linewidth=1, alpha=0.9))
    
    axes[1].set_xlabel('Algorithms', fontsize=14)
    axes[1].set_ylabel('Health Improvement per $1000', fontsize=14)
    axes[1].set_title('Cost Effectiveness (Bubble size ∝ Total Cost)', fontsize=16)
    axes[1].set_xticks(range(len(algo_names)))
    axes[1].set_xticklabels([f'{i+1}' for i in range(len(algo_names))], fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add horizontal line at zero for reference
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # Add algorithm mapping text box
    algo_text = "Algorithm Index:\n" + "\n".join([f"{i+1:2d}. {name[:30]}" for i, name in enumerate(algo_names)])
    algo_text += f"\n H. Historical ({historical_improvement_pct:.2f}%)"
    fig.text(0.30, 0.80, algo_text,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Simplified comparison plot saved to: {save_path}")
    
    plt.show()

def print_performance_table(final_data):
    """
    向控制台打印综合性能表，包括多次运行算法的平均值和标准差。
    """
    if not final_data:
        print("No final data to print in table.")
        return

    algo_names = list(final_data.keys())
    
    # from样本中计算历史改进百分比
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    print("\n" + "="*150)
    print("COMPREHENSIVE BRIDGE MAINTENANCE STRATEGY PERFORMANCE TABLE")
    print("="*150)
    
    # 表头
    header = (f"{'#':<3} {'Algorithm':<30} {'Budget Ratio':<18} {'Improve vs None(%)':<20} {'Improve vs Hist(%)':<20} {'Efficiency Score':<20} "
             f"{'Behavioral Sim.':<18} {'Violation Rate':<18} {'Pareto':<6}")
    print(header)
    print("-" * 150)
    
    # 辅助函数，用于格式化单元格文本
    def format_cell(data, key, format_spec, width):
        mean = data.get(key, 0)
        std = data.get(f"{key}_std", 0)
        num_runs = data.get('num_runs', 1)

        if num_runs > 1 and std > 1e-4: # only当标准差有意义时显示
            text = f"{mean:{format_spec}} ± {std:.2f}"
        else:
            text = f"{mean:{format_spec}}"
        return f"{text:<{width}}"

    # compute帕累托前沿
    budget_ratios = [d['budget_usage_ratio'] for d in final_data.values()]
    improvement_pcts = [d['improvement_vs_nothing_pct'] for d in final_data.values()]
    pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_pcts, maximize_x=False, maximize_y=True)
    
    # by效率分数对算法排序
    sorted_data = sorted(enumerate(algo_names), 
                         key=lambda x: final_data[x[1]]['efficiency_score'], reverse=True)
    
    # print算法数据
    for rank, (orig_idx, algo_name) in enumerate(sorted_data, 1):
        data = final_data[algo_name]
        is_pareto = orig_idx in pareto_indices
        
        row = (f"{orig_idx+1:<3} {algo_name[:29]:<30} "
               f"{format_cell(data, 'budget_usage_ratio', '.3f', 18)} "
               f"{format_cell(data, 'improvement_vs_nothing_pct', '.2f', 20)} "
               f"{format_cell(data, 'improvement_vs_historical_pct', '.2f', 20)} "
               f"{format_cell(data, 'efficiency_score', '.2f', 20)} "
               f"{format_cell(data, 'behavioral_similarity', '.3f', 18)} "
               f"{format_cell(data, 'violation_rate', '.3f', 18)} "
               f"{'★' if is_pareto else '':<6}")
        
        print(row)
    
    # add历史基线
    print("-" * 150)
    historical_row = (f"{'H':<3} {'Historical Baseline':<30} "
                     f"{'1.000':<18} "
                     f"{historical_improvement_pct:<20.2f} "
                     f"{'0.00':<20} "
                     f"{'N/A':<20} "
                     f"{'N/A':<18} "
                     f"{'N/A':<18} "
                     f"{'':<6}")
    print(historical_row)
    print("="*150)

def generate_performance_report(final_data, output_dir):
    """
    生成简化的性能报告，包含方差信息。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'bridge_maintenance_performance_report_{timestamp}.txt')
    
    algo_names = list(final_data.keys())
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BRIDGE MAINTENANCE STRATEGY PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total algorithms analyzed: {len(final_data)}\n\n")
        
        historical_improvement_pct = 0.0
        if final_data:
            sample_data = next(iter(final_data.values()))
            historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
            historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
            
            f.write("BASELINE VALUES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Do Nothing Baseline: {sample_data['do_nothing_baseline']:.4f}\n")
            f.write(f"Historical Policy Baseline: {sample_data['historical_baseline']:.4f}\n")
            f.write(f"Historical vs Do Nothing: {historical_improvement_vs_nothing:.4f}\n")
            f.write(f"Historical Improvement %: {historical_improvement_pct:.2f}%\n\n")
        
        f.write("ALGORITHM INDEX MAPPING\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(algo_names):
            f.write(f"{i+1:2d}. {name}\n")
        f.write(f" H. Historical Baseline ({historical_improvement_pct:.2f}%)\n\n")
        
        f.write("COMPREHENSIVE PERFORMANCE TABLE\n")
        f.write("=" * 140 + "\n")
        
        budget_ratios = [d['budget_usage_ratio'] for d in final_data.values()]
        improvement_pcts = [d['improvement_vs_nothing_pct'] for d in final_data.values()]
        pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_pcts, maximize_x=False, maximize_y=True)
        
        sorted_data = sorted(enumerate(algo_names), key=lambda x: final_data[x[1]]['efficiency_score'], reverse=True)
        
        f.write(f"{'#':<3} {'Algorithm':<30} {'Budget Ratio':<18} {'Improve vs None(%)':<20} {'Improve vs Hist(%)':<20} {'Efficiency Score':<20} {'Pareto':<6}\n")
        f.write("-" * 140 + "\n")
        
        def format_cell_file(data, key, format_spec):
            mean = data.get(key, 0)
            std = data.get(f"{key}_std", 0)
            num_runs = data.get('num_runs', 1)
            if num_runs > 1 and std > 1e-4:
                return f"{mean:{format_spec}} ± {std:.2f}"
            return f"{mean:{format_spec}}"

        for rank, (orig_idx, algo_name) in enumerate(sorted_data, 1):
            data = final_data[algo_name]
            is_pareto = orig_idx in pareto_indices
            
            f.write(f"{orig_idx+1:<3} {algo_name[:29]:<30} "
                   f"{format_cell_file(data, 'budget_usage_ratio', '.3f'):<18} "
                   f"{format_cell_file(data, 'improvement_vs_nothing_pct', '.2f'):<20} "
                   f"{format_cell_file(data, 'improvement_vs_historical_pct', '.2f'):<20} "
                   f"{format_cell_file(data, 'efficiency_score', '.2f'):<20} "
                   f"{'★' if is_pareto else '':<6}\n")
        
        f.write("-" * 140 + "\n")
        f.write(f"{'H':<3} {'Historical Baseline':<30} "
               f"{'1.000':<18} "
               f"{historical_improvement_pct:<20.2f} "
               f"{'0.00':<20} "
               f"{'N/A':<20} "
               f"{'':<6}\n")
        f.write("=" * 140 + "\n")
    
    print(f"Performance report saved to: {report_path}")
    return report_path

def generate_bridge_maintenance_report(training_results_dir="training_results", 
                                     output_dir="bridge_maintenance_analysis",
                                     algos_for_variance=None,
                                     load_latest_n_for_all=False,
                                     num_recent_files=3):
    """
    生成简化的桥梁维护分析报告，仅包含核心功能。
    """
    print("Starting simplified bridge maintenance strategy analysis...")
    
    if algos_for_variance is None:
        algos_for_variance = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("1. Loading training history data...")
    training_histories = load_training_histories(
        training_results_dir, 
        multi_file_algos=("ALL" if load_latest_n_for_all else algos_for_variance),
        num_files=num_recent_files,
    )
    
    if not training_histories:
        print("No training history data found!")
        return
    
    print("2. Extracting final evaluation metrics and calculating improvement percentages...")
    final_data = extract_final_metrics(training_histories)
    
    if not final_data:
        print("No final evaluation data found!")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    algo_names = list(final_data.keys())
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    print("\n" + "="*80)
    print("ALGORITHM INDEX MAPPING")
    print("="*80)
    for i, name in enumerate(algo_names):
        runs = final_data[name].get('num_runs', 1)
        print(f"{i+1:2d}. {name} ({runs} run{'s' if runs > 1 else ''})")
    print(f" H. Historical Baseline ({historical_improvement_pct:.2f}%)")
    print("="*80)
    
    print_performance_table(final_data)
    
    print("\n4. Generating Pareto analysis...")
    pareto_path = os.path.join(output_dir, f'bridge_pareto_analysis_core_{timestamp}.png')
    plot_pareto_analysis(final_data, save_path=pareto_path)
    
    print("5. Generating simplified comparison (2 core subplots)...")
    comparison_path = os.path.join(output_dir, f'bridge_core_comparison_{timestamp}.png')
    plot_simplified_comparison(final_data, save_path=comparison_path)
    
    print("6. Generating simplified performance report...")
    report_path = generate_performance_report(final_data, output_dir)
    
    print("7. Saving final data summary...")
    summary_path = os.path.join(output_dir, f'bridge_final_metrics_core_{timestamp}.json')
    
    final_data_with_mapping = {
        'algorithm_mapping': {i+1: name for i, name in enumerate(algo_names)},
        'baseline_values': {
            'do_nothing_baseline': sample_data['do_nothing_baseline'],
            'historical_baseline': sample_data['historical_baseline'],
            'historical_improvement_vs_nothing': historical_improvement_vs_nothing,
            'historical_improvement_pct': historical_improvement_pct
        },
        'metrics': final_data
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_data_with_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Final metrics saved to: {summary_path}")
    
    print("\nSimplified bridge maintenance strategy analysis completed!")
    print(f"All analysis results saved to directory: {output_dir}")
    print(f"Key outputs:")
    print(f"  - Pareto analysis: {pareto_path}")
    print(f"  - Core comparison (2 subplots): {comparison_path}")
    print(f"  - Performance report: {report_path}")
    print(f"  - Final metrics: {summary_path}")

# use示例
if __name__ == "__main__":
    # at这里手动选择需要计算方差的算法
    # 程序将为这些算法查找最近的3个结果文件并计算均值和标准差
    selected_algorithms_for_variance = [
        "discrete_bc_50", 
        "multitask_cpq", 
        "cdt",
        "qmix_cql",
        "random_marl",
        "multitask_bc",
        "iqlcql_marl",
        "iqlcql_marl_without_budget",
    ]

    # 生成简化的桥梁维护分析报告
    generate_bridge_maintenance_report(
        training_results_dir="paper/benchmark/training_results",
        output_dir="paper/benchmark/bridge_maintenance_analysis",
        algos_for_variance=selected_algorithms_for_variance,
        # if你希望像 onestep 这类算法也自动聚合多次运行，打开这个开关
        load_latest_n_for_all=True,
        num_recent_files=3,
    )