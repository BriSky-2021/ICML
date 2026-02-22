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
warnings.filterwarnings('ignore')

def load_training_histories(training_results_dir="training_results"):
    """
    Load training history data for all algorithms
    
    Args:
        training_results_dir: Training results directory
    
    Returns:
        dict: {algorithm_name: training_history}
    """
    if not os.path.exists(training_results_dir):
        print(f"Directory {training_results_dir} does not exist!")
        return {}
    
    training_histories = {}
    
    # Get all training history files
    pattern = os.path.join(training_results_dir, "*_training_history_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No training history files found in directory {training_results_dir}!")
        return {}
    
    # Group by algorithm name, take the latest file
    algo_files = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Parse filename: algo_name_training_history_timestamp.json
        parts = filename.replace('_training_history_', '|').replace('.json', '').split('|')
        if len(parts) == 2:
            algo_name = parts[0]
            timestamp = parts[1]
            
            if algo_name not in algo_files or timestamp > algo_files[algo_name][1]:
                algo_files[algo_name] = (file_path, timestamp)
    
    # Load the latest training history for each algorithm
    for algo_name, (file_path, _) in algo_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                training_histories[algo_name] = json.load(f)
            print(f"Successfully loaded training history for {algo_name}")
        except Exception as e:
            print(f"Failed to load training history for {algo_name}: {e}")
    
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
    Extract final evaluation metrics (last epoch only) from training history
    
    Args:
        training_histories: Training history dictionary
    
    Returns:
        dict: {algorithm_name: final_metrics}
    """
    final_data = {}
    
    for algo_name, history in training_histories.items():
        if 'eval_metrics' not in history or not history['eval_metrics']:
            print(f"Algorithm {algo_name} has no evaluation data")
            continue
        
        # Get the last evaluation record
        last_eval = history['eval_metrics'][-1]
        metrics = last_eval['metrics']
        
        # Extract key metrics for bridge maintenance analysis
        final_metrics = {
            'epoch': last_eval['epoch'],
            
            # 行为相似性 (重命名的accuracy)
            'behavioral_similarity': metrics.get('behavioral_similarity_mean', 0.0),
            
            # 成本相关
            'total_cost': metrics.get('mean_total_cost', 0.0),
            'budget_usage_ratio': metrics.get('budget_usage_ratio', 0.0),  # budget使用占比 vs 历史
            
            # 违规率
            'violation_rate': metrics.get('violation_rate_mean', 0.0),
            
            # 桥梁健康改善指标
            'health_gain_absolute': metrics.get('bridge_avg_health_gain_absolute', 0.0),  # 绝对健康改善
            'health_gain_vs_history': metrics.get('bridge_avg_health_gain_vs_history', 0.0),  # 相对历史的改善
            'health_gain_vs_nothing': metrics.get('bridge_avg_health_gain_vs_nothing', 0.0),  # 相对什么都不做的改善
            'health_gain_normalized': metrics.get('bridge_avg_health_gain_normalized', 0.0),  # normalize改善
            
            # 原有效率指标
            'health_gain_per_1000_dollars': metrics.get('health_gain_per_1000_dollars', 0.0),
            
            # 综合评分
            'comprehensive_score': metrics.get('comprehensive_score', 0.0),
            
            # 总评估数量
            'total_evaluations': metrics.get('total_evaluations', 0),
            
            # hard constraint指标（如果存在）
            'behavioral_similarity_r': metrics.get('behavioral_similarity_mean_r', None),
            'total_cost_r': metrics.get('mean_total_cost_r', None),
            'budget_usage_ratio_r': metrics.get('budget_usage_ratio_r', None),
            'health_gain_absolute_r': metrics.get('bridge_avg_health_gain_absolute_r', None),
            'health_gain_vs_history_r': metrics.get('bridge_avg_health_gain_vs_history_r', None),
            'health_gain_vs_nothing_r': metrics.get('bridge_avg_health_gain_vs_nothing_r', None),
            'comprehensive_score_r': metrics.get('comprehensive_score_r', None),
        }
        
        final_data[algo_name] = final_metrics
        print(f"Algorithm {algo_name}: extracted final metrics from epoch {final_metrics['epoch']}")
    
    # compute历史策略相对于什么都不做的改善值
    historical_improvement_vs_nothing = -0.0180 - (-0.0242)  # = 0.0062
    
    # compute增强的提升比例指标
    enhanced_final_data = calculate_improvement_percentages(
        final_data, 
        historical_improvement_vs_nothing=historical_improvement_vs_nothing
    )
    
    return enhanced_final_data

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
    Plot Pareto analysis using improvement percentages with correct historical baseline
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
    budget_ratios = []
    improvement_percentages = []  # use相对于什么都不做的提升比例
    
    # compute历史基准点的正确值
    sample_data = next(iter(final_data.values()))
    historical_budget_ratio = 1.0
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    for algo_name in algo_names:
        data = final_data[algo_name]
        budget_ratios.append(data['budget_usage_ratio'])
        improvement_percentages.append(data['improvement_vs_nothing_pct'])  # use提升比例
    
    # Calculate Pareto frontier (minimize budget, maximize improvement percentage)
    pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_percentages, 
                                             maximize_x=False, maximize_y=True)
    
    # Enhanced colors with better contrast
    colors = [
        '#FF4444', '#4444FF', '#44FF44', '#FF8800', '#8844FF', '#FF4488',
        '#44FFFF', '#FFFF44', '#FF8844', '#8888FF', '#44FF88', '#FF44FF',
        '#888844', '#448888', '#884444', '#FF8888', '#8888FF', '#88FF88'
    ]
    
    # Different marker styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '8', 'P', 'X', '1']
    
    # Plot all algorithms on main plot
    for i, (algo_name, color) in enumerate(zip(algo_names, colors)):
        marker = markers[i % len(markers)]
        size = 400 if i in pareto_indices else 200
        alpha = 1.0 if i in pareto_indices else 0.8
        
        # Plot the point
        scatter = ax_main.scatter(budget_ratios[i], improvement_percentages[i], 
                                 color=color, marker=marker, s=size, alpha=alpha,
                                 edgecolors='black', linewidths=3 if i in pareto_indices else 2,
                                 label=f'{i+1:2d}. {algo_name}', zorder=5)
        
        # Add algorithm number with better visibility
        text_color = get_contrasting_text_color(color)
        ax_main.annotate(f'{i+1}', (budget_ratios[i], improvement_percentages[i]), 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color=text_color, zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.1", facecolor=color, 
                                edgecolor='black', linewidth=1, alpha=0.9))
    
    # Add historical baseline point with correct improvement percentage
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
        
        # Sort by x coordinate for proper line drawing
        pareto_points = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted, pareto_y_sorted = zip(*pareto_points)
        
        ax_main.plot(pareto_x_sorted, pareto_y_sorted, 'r-', linewidth=3, alpha=0.8,
                    label='Pareto Frontier', zorder=4)
        
        # Highlight Pareto points with enhanced outline
        ax_main.scatter(pareto_x_sorted, pareto_y_sorted, 
                       facecolors='none', edgecolors='gold', s=500, linewidths=4, zorder=7)
    
    # Set labels and formatting for main plot
    ax_main.set_xlabel('Budget Usage Ratio (vs Historical)', fontsize=16)
    ax_main.set_ylabel('Health Improvement vs Do Nothing (%)', fontsize=16)
    ax_main.set_title('Pareto Analysis: Budget Efficiency vs Health Improvement Percentage', fontsize=18)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelsize=12)
    
    # Add quadrant lines
    ax_main.axhline(y=historical_improvement_pct, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax_main.axvline(x=historical_budget_ratio, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # Add quadrant labels
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    
    ax_main.text(xlim[0] + (xlim[1] - xlim[0]) * 0.05, ylim[1] - (ylim[1] - ylim[0]) * 0.05, 
                'Better Health\nLower Cost', fontsize=12, ha='left', va='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8, edgecolor='green'))
    ax_main.text(xlim[1] - (xlim[1] - xlim[0]) * 0.05, ylim[0] + (ylim[1] - ylim[0]) * 0.05, 
                'Worse Health\nHigher Cost', fontsize=12, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8, edgecolor='red'))
    
    # Algorithm legend
    ax_legend.axis('off')
    legend_text = "ALGORITHM REFERENCE:\n" + "="*25 + "\n"
    
    # Split algorithms into two columns for better readability
    mid_point = len(algo_names) // 2
    for i in range(max(mid_point + 1, len(algo_names) - mid_point)):
        line = ""
        if i < len(algo_names):
            line += f"{i+1:2d}. {algo_names[i][:20]}"
        if i + mid_point + 1 < len(algo_names):
            line += f"  |  {i + mid_point + 1:2d}. {algo_names[i + mid_point + 1][:20]}"
        legend_text += line + "\n"
    
    legend_text += f"\n H. Historical ({historical_improvement_pct:.2f}%)"
    
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Pareto frontier information
    ax_pareto_info.axis('off')
    pareto_text = f"PARETO FRONTIER:\n" + "="*20 + f"\n{len(pareto_indices)} algorithms optimal\n\nPareto Algorithms:\n"
    for idx in pareto_indices:
        pareto_text += f"{idx+1:2d}. {algo_names[idx][:15]}...\n" if len(algo_names[idx]) > 15 else f"{idx+1:2d}. {algo_names[idx]}\n"
    
    ax_pareto_info.text(0.05, 0.95, pareto_text, transform=ax_pareto_info.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='gold', alpha=0.8))
    
    # Enhanced performance table
    ax_table.axis('off')
    
    # Create performance ranking table
    performance_data = []
    for i, algo_name in enumerate(algo_names):
        data = final_data[algo_name]
        performance_data.append([
            f"{i+1:2d}",
            algo_name[:25],
            f"{data['budget_usage_ratio']:.3f}",
            f"{data['improvement_vs_nothing_pct']:.2f}%",
            f"{data['improvement_vs_historical_pct']:.2f}%",
            f"{data['efficiency_score']:.2f}",
            "★" if i in pareto_indices else ""
        ])
    
    # Add historical baseline
    performance_data.append([
        "H",
        "Historical Baseline",
        f"{historical_budget_ratio:.3f}",
        f"{historical_improvement_pct:.2f}%",
        "0.00%",
        "N/A",
        ""
    ])
    
    # Sort by efficiency score
    performance_data_sorted = sorted(performance_data[:-1], 
                                   key=lambda x: float(x[5]), reverse=True)
    performance_data_sorted.append(performance_data[-1])  # Add historical baseline at end
    
    # Create table with better formatting
    table = ax_table.table(cellText=performance_data_sorted,
                          colLabels=['#', 'Algorithm', 'Budget Ratio', 'Improve vs Nothing', 'Improve vs History', 'Efficiency', 'Pareto'],
                          cellLoc='center', loc='center',
                          colWidths=[0.06, 0.3, 0.12, 0.15, 0.15, 0.1, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Enhanced table styling
    for i, row in enumerate(performance_data_sorted):
        for j in range(len(row)):
            cell = table[(i+1, j)]
            if row[6] == "★":  # Pareto optimal
                cell.set_facecolor('#FFD700')  # Gold
                cell.set_text_props(weight='bold')
            elif row[0] == "H":  # Historical baseline
                cell.set_facecolor('#FFCCCC')  # Light red
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F0F0F0')  # Light gray
    
    # Style header row
    for j in range(7):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Pareto analysis plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'pareto_indices': pareto_indices,
        'pareto_algorithms': [algo_names[i] for i in pareto_indices],
        'performance_data': performance_data_sorted,
        'algorithm_mapping': {i+1: name for i, name in enumerate(algo_names)},
        'historical_improvement_pct': historical_improvement_pct
    }

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
    
    # Enhanced colors
    colors = [
        '#FF4444', '#4444FF', '#44FF44', '#FF8800', '#8844FF', '#FF4488',
        '#44FFFF', '#FFFF44', '#FF8844', '#8888FF', '#44FF88', '#FF44FF',
        '#888844', '#448888', '#884444', '#FF8888', '#8888FF', '#88FF88'
    ]
    
    # 1. Budget vs Improvement Percentage Scatter with correct historical baseline
    budget_ratios = [final_data[algo]['budget_usage_ratio'] for algo in algo_names]
    improvement_pcts = [final_data[algo]['improvement_vs_nothing_pct'] for algo in algo_names]
    
    # Calculate correct historical improvement percentage
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    for i, (algo, color) in enumerate(zip(algo_names, colors)):
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
    
    for i, (algo, color, size) in enumerate(zip(algo_names, colors, normalized_sizes)):
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
    Print comprehensive performance table to console (without action distributions)
    """
    algo_names = list(final_data.keys())
    
    # Calculate historical improvement percentage
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    print("\n" + "="*140)
    print("COMPREHENSIVE BRIDGE MAINTENANCE STRATEGY PERFORMANCE TABLE")
    print("="*140)
    
    # Table header
    header = (f"{'#':<3} {'Algorithm':<30} {'Budget':<8} {'Improve':<8} {'Improve':<8} {'Efficiency':<10} "
             f"{'Behavioral':<10} {'Violation':<10} {'Cost/1M$':<12} {'Total Cost':<12} {'Pareto':<6}")
    
    subheader = (f"{'':3} {'':30} {'Ratio':<8} {'vs None':<8} {'vs Hist':<8} {'Score':<10} "
                f"{'Similar':<10} {'Rate':<10} {'Health':<12} {'($)':<12} {'Opt':<6}")
    
    print(header)
    print(subheader)
    print("-" * 140)
    
    # Calculate Pareto frontier
    budget_ratios = [final_data[algo]['budget_usage_ratio'] for algo in algo_names]
    improvement_pcts = [final_data[algo]['improvement_vs_nothing_pct'] for algo in algo_names]
    pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_pcts, 
                                             maximize_x=False, maximize_y=True)
    
    # Sort algorithms by efficiency score
    sorted_data = sorted(enumerate(algo_names), 
                        key=lambda x: final_data[x[1]]['efficiency_score'], reverse=True)
    
    # Print algorithm data
    for rank, (orig_idx, algo_name) in enumerate(sorted_data, 1):
        data = final_data[algo_name]
        is_pareto = orig_idx in pareto_indices
        
        row = (f"{orig_idx+1:<3} {algo_name[:29]:<30} "
               f"{data['budget_usage_ratio']:<8.3f} "
               f"{data['improvement_vs_nothing_pct']:<8.2f} "
               f"{data['improvement_vs_historical_pct']:<8.2f} "
               f"{data['efficiency_score']:<10.2f} "
               f"{data['behavioral_similarity']:<10.3f} "
               f"{data['violation_rate']:<10.3f} "
               f"{data['health_improvement_per_1M_dollars']:<12.4f} "
               f"{data['total_cost']:<12.0f} "
               f"{'★' if is_pareto else '':<6}")
        
        print(row)
    
    # Add historical baseline
    print("-" * 140)
    historical_row = (f"{'H':<3} {'Historical Baseline':<30} "
                     f"{'1.000':<8} "
                     f"{historical_improvement_pct:<8.2f} "
                     f"{'0.00':<8} "
                     f"{'N/A':<10} "
                     f"{'N/A':<10} "
                     f"{'N/A':<10} "
                     f"{'N/A':<12} "
                     f"{'N/A':<12} "
                     f"{'':<6}")
    print(historical_row)
    print("="*140)
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print("-" * 50)
    
    # Best performers
    best_efficiency = max(final_data.items(), key=lambda x: x[1]['efficiency_score'])
    best_budget = min(final_data.items(), key=lambda x: x[1]['budget_usage_ratio'])
    best_health = max(final_data.items(), key=lambda x: x[1]['improvement_vs_nothing_pct'])
    best_similarity = max(final_data.items(), key=lambda x: x[1]['behavioral_similarity'])
    lowest_violation = min(final_data.items(), key=lambda x: x[1]['violation_rate'])
    best_cost_effectiveness = max(final_data.items(), key=lambda x: x[1]['health_improvement_per_1M_dollars'])
    
    best_efficiency_idx = algo_names.index(best_efficiency[0]) + 1
    best_budget_idx = algo_names.index(best_budget[0]) + 1
    best_health_idx = algo_names.index(best_health[0]) + 1
    best_similarity_idx = algo_names.index(best_similarity[0]) + 1
    lowest_violation_idx = algo_names.index(lowest_violation[0]) + 1
    best_cost_effectiveness_idx = algo_names.index(best_cost_effectiveness[0]) + 1
    
    print(f"Best Efficiency Score:        [{best_efficiency_idx:2d}] {best_efficiency[0]} ({best_efficiency[1]['efficiency_score']:.2f})")
    print(f"Most Budget Efficient:        [{best_budget_idx:2d}] {best_budget[0]} ({best_budget[1]['budget_usage_ratio']:.3f})")
    print(f"Best Health Improvement:      [{best_health_idx:2d}] {best_health[0]} ({best_health[1]['improvement_vs_nothing_pct']:.2f}%)")
    print(f"Best Behavioral Similarity:   [{best_similarity_idx:2d}] {best_similarity[0]} ({best_similarity[1]['behavioral_similarity']:.3f})")
    print(f"Lowest Violation Rate:        [{lowest_violation_idx:2d}] {lowest_violation[0]} ({lowest_violation[1]['violation_rate']:.3f})")
    print(f"Best Cost Effectiveness:      [{best_cost_effectiveness_idx:2d}] {best_cost_effectiveness[0]} ({best_cost_effectiveness[1]['health_improvement_per_1M_dollars']:.4f})")
    
    # Pareto analysis
    print(f"\nPARETO FRONTIER ANALYSIS:")
    print("-" * 50)
    print(f"Pareto optimal algorithms: {len(pareto_indices)}/{len(algo_names)}")
    print("Pareto optimal algorithms:")
    for idx in pareto_indices:
        data = final_data[algo_names[idx]]
        print(f"  [{idx+1:2d}] {algo_names[idx]}: Budget {data['budget_usage_ratio']:.3f}, Health {data['improvement_vs_nothing_pct']:.2f}%")
    
    # Historical comparison
    better_budget = sum(1 for data in final_data.values() if data['budget_usage_ratio'] < 1.0)
    better_health = sum(1 for data in final_data.values() if data['improvement_vs_nothing_pct'] > historical_improvement_pct)
    better_both = sum(1 for data in final_data.values() 
                     if data['budget_usage_ratio'] < 1.0 and data['improvement_vs_nothing_pct'] > historical_improvement_pct)
    
    print(f"\nCOMPARISON WITH HISTORICAL BASELINE:")
    print("-" * 50)
    print(f"Algorithms with better budget efficiency: {better_budget}/{len(algo_names)}")
    print(f"Algorithms with better health outcomes:   {better_health}/{len(algo_names)}")
    print(f"Algorithms better in both dimensions:     {better_both}/{len(algo_names)}")

def generate_performance_report(final_data, output_dir):
    """
    Generate simplified performance report
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'bridge_maintenance_performance_report_{timestamp}.txt')
    
    algo_names = list(final_data.keys())
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BRIDGE MAINTENANCE STRATEGY PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total algorithms analyzed: {len(final_data)}\n\n")
        
        # Baseline information
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
        
        # Algorithm index mapping
        f.write("ALGORITHM INDEX MAPPING\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(algo_names):
            f.write(f"{i+1:2d}. {name}\n")
        f.write(f" H. Historical Baseline ({historical_improvement_pct:.2f}%)\n\n")
        
        # Performance summary table
        f.write("COMPREHENSIVE PERFORMANCE TABLE\n")
        f.write("=" * 120 + "\n")
        
        # Calculate Pareto frontier
        budget_ratios = [final_data[algo]['budget_usage_ratio'] for algo in algo_names]
        improvement_pcts = [final_data[algo]['improvement_vs_nothing_pct'] for algo in algo_names]
        pareto_indices = calculate_pareto_frontier(budget_ratios, improvement_pcts, 
                                                 maximize_x=False, maximize_y=True)
        
        # Sort algorithms by efficiency score
        sorted_data = sorted(enumerate(algo_names), 
                            key=lambda x: final_data[x[1]]['efficiency_score'], reverse=True)
        
        # Write table header
        f.write(f"{'#':<3} {'Algorithm':<30} {'Budget':<8} {'Improve':<8} {'Improve':<8} {'Efficiency':<10} {'Behavioral':<10} {'Violation':<10} {'Pareto':<6}\n")
        f.write(f"{'':3} {'':30} {'Ratio':<8} {'vs None':<8} {'vs Hist':<8} {'Score':<10} {'Similar':<10} {'Rate':<10} {'Opt':<6}\n")
        f.write("-" * 120 + "\n")
        
        # Write algorithm data
        for rank, (orig_idx, algo_name) in enumerate(sorted_data, 1):
            data = final_data[algo_name]
            is_pareto = orig_idx in pareto_indices
            
            f.write(f"{orig_idx+1:<3} {algo_name[:29]:<30} "
                   f"{data['budget_usage_ratio']:<8.3f} "
                   f"{data['improvement_vs_nothing_pct']:<8.2f} "
                   f"{data['improvement_vs_historical_pct']:<8.2f} "
                   f"{data['efficiency_score']:<10.2f} "
                   f"{data['behavioral_similarity']:<10.3f} "
                   f"{data['violation_rate']:<10.3f} "
                   f"{'★' if is_pareto else '':<6}\n")
        
        # Add historical baseline
        f.write("-" * 120 + "\n")
        f.write(f"{'H':<3} {'Historical Baseline':<30} "
               f"{'1.000':<8} "
               f"{historical_improvement_pct:<8.2f} "
               f"{'0.00':<8} "
               f"{'N/A':<10} "
               f"{'N/A':<10} "
               f"{'N/A':<10} "
               f"{'N/A':<6}\n")
        f.write("=" * 120 + "\n")
    
    print(f"Performance report saved to: {report_path}")
    return report_path

def generate_bridge_maintenance_report(training_results_dir="training_results", 
                                     output_dir="bridge_maintenance_analysis"):
    """
    Generate simplified bridge maintenance analysis report with core functionality only
    """
    print("Starting simplified bridge maintenance strategy analysis...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Load training history data
    print("1. Loading training history data...")
    training_histories = load_training_histories(training_results_dir)
    
    if not training_histories:
        print("No training history data found!")
        return
    
    # 2. Extract final evaluation metrics with corrected improvement percentages
    print("2. Extracting final evaluation metrics and calculating improvement percentages...")
    final_data = extract_final_metrics(training_histories)
    
    if not final_data:
        print("No final evaluation data found!")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Print algorithm mapping
    algo_names = list(final_data.keys())
    sample_data = next(iter(final_data.values()))
    historical_improvement_vs_nothing = sample_data['historical_improvement_vs_nothing']
    historical_improvement_pct = (historical_improvement_vs_nothing / abs(sample_data['do_nothing_baseline'])) * 100
    
    print("\n" + "="*80)
    print("ALGORITHM INDEX MAPPING")
    print("="*80)
    for i, name in enumerate(algo_names):
        print(f"{i+1:2d}. {name}")
    print(f" H. Historical Baseline ({historical_improvement_pct:.2f}%)")
    print("="*80)
    
    # 3. Print comprehensive performance table to console
    print_performance_table(final_data)
    
    # 4. Generate Pareto analysis
    print("\n4. Generating Pareto analysis...")
    pareto_path = os.path.join(output_dir, f'bridge_pareto_analysis_core_{timestamp}.png')
    pareto_results = plot_pareto_analysis(final_data, save_path=pareto_path)
    
    # 5. Generate simplified comparison (2 subplots)
    print("5. Generating simplified comparison (2 core subplots)...")
    comparison_path = os.path.join(output_dir, f'bridge_core_comparison_{timestamp}.png')
    plot_simplified_comparison(final_data, save_path=comparison_path)
    
    # 6. Generate simplified text report
    print("6. Generating simplified performance report...")
    report_path = generate_performance_report(final_data, output_dir)
    
    # 7. Save final data summary
    print("7. Saving final data summary...")
    summary_path = os.path.join(output_dir, f'bridge_final_metrics_core_{timestamp}.json')
    
    # Add algorithm mapping and historical data to the saved data
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

# Usage example
if __name__ == "__main__":
    # Generate simplified bridge maintenance analysis report
    generate_bridge_maintenance_report(
        training_results_dir="paper/benchmark/training_results",
        output_dir="paper/benchmark/bridge_maintenance_analysis"
    )