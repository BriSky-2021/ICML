import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Union
from bridge_maintenance_simulator_v2 import BridgeMaintenanceSimulator


class AdvancedMultiAlgorithmSimulator:
    def __init__(self, test_data_path, env_info_path=None, action_costs=None, 
                 episode_idx=0, initial_health_level=9, output_dir="simulation_results"):
        """
        高级多算法仿真器
        
        Args:
            test_data_path: 测试数据集路径
            env_info_path: 环境信息文件路径
            action_costs: 动作成本字典
            episode_idx: 使用的测试episode索引
            initial_health_level: 初始健康等级（0-9）
            output_dir: 结果输出目录
        """
        self.test_data_path = test_data_path
        self.env_info_path = env_info_path
        self.action_costs = action_costs or {0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03}
        self.episode_idx = episode_idx
        self.initial_health_level = initial_health_level
        self.output_dir = output_dir
        
        # create输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 可用的算法列表
        self.available_algorithms = [
            "multitask_cpq",
            "multitask_bc", 
            "cdt",
            "random_osrl",
            "discrete_bc_50",
            "iqlcql_marl",
            "qmix_cql",
        ]
        
        # 可用的预算分配策略
        self.available_strategies = [
            'original',
            'uniform', 
            'importance',
            'importance_top10',
            'critical_first',
            'threshold_based',
            'importance_health_combined',
            'preventive_maintenance',
            'rotating_focus'
        ]
        
        # 策略说明
        self.strategy_descriptions = {
            'original': 'Original',
            'uniform': 'Uniform',
            'importance': 'Importance-based',
            'importance_top10': 'Uniform Top 10% Importance',
            'critical_first': 'Worst 30%',
            'threshold_based': 'Threshold',
            'importance_health_combined': 'Importance and Health',
            'preventive_maintenance': 'Preventive',
            'rotating_focus': 'Rotating Focus'
        }
        
        # 存储所有仿真结果
        self.simulation_results = {}
        
        print(f"高级多算法仿真器初始化完成")
        print(f"可用算法: {self.available_algorithms}")
        print(f"可用策略: {len(self.available_strategies)} 种")
        print(f"output directory: {self.output_dir}")
    

    def scenario_1_budget_scaling(self, algorithm_names: Union[str, List[str]], 
                                budget_multipliers: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
                                allocation_strategy: str = 'original',
                                n_years: int = 50):
        """
        情况1: 选定算法(们)，不同预算缩放因子实验
        
        Args:
            algorithm_names: 算法名称或算法列表
            budget_multipliers: 预算缩放因子列表
            allocation_strategy: 预算分配策略
            n_years: 仿真年数
        """
        # ensurealgorithm_names是列表
        if isinstance(algorithm_names, str):
            algorithm_names = [algorithm_names]
        
        print(f"\n{'='*80}")
        print(f"情况1: 预算缩放因子实验")
        print(f"算法: {algorithm_names}")
        print(f"分配策略: {allocation_strategy}")
        print(f"缩放因子: {budget_multipliers}")
        print(f"{'='*80}")
        
        all_scenario_results = {}
        
        # for每个算法分别执行原有逻辑
        for algorithm_name in algorithm_names:
            print(f"\n🔹 算法: {algorithm_name}")
            scenario_results = {}
            
            for multiplier in budget_multipliers:
                print(f"\n--- 预算缩放因子: {multiplier} ---")
                
                try:
                    # create仿真器
                    simulator = BridgeMaintenanceSimulator(
                        model_path=self.find_latest_model(algorithm_name),
                        algorithm_name=algorithm_name,
                        test_data_path=self.test_data_path,
                        env_info_path=self.env_info_path,
                        action_costs=self.action_costs,
                        episode_idx=self.episode_idx,
                        initial_health_level=self.initial_health_level,
                        budget_multiplier=multiplier,
                        budget_allocation_strategy=allocation_strategy
                    )
                    
                    # 运行仿真
                    results = simulator.run_simulation(n_years=n_years)
                    
                    # store result
                    scenario_results[f"x{multiplier}"] = {
                        'simulator': simulator,
                        'results': results,
                        'multiplier': multiplier
                    }
                    
                    print(f"预算缩放因子 {multiplier} 仿真完成")
                    
                except Exception as e:
                    print(f"预算缩放因子 {multiplier} 仿真失败: {e}")
                    continue
            
            # 存储单个算法的结果
            all_scenario_results[algorithm_name] = scenario_results
            
            # 为单个算法绘制对比图表
            if scenario_results:
                self.plot_budget_scaling_comparison(scenario_results, algorithm_name, allocation_strategy)
                scenario_key = f"scenario1_{algorithm_name}_{allocation_strategy}"
                self.save_scenario_results(scenario_results, scenario_key)
        
        # if有多个算法，绘制算法间对比图
        if len(algorithm_names) > 1:
            self.plot_multi_algorithm_budget_scaling(all_scenario_results, budget_multipliers, allocation_strategy)
        
        # 存储到主结果中
        scenario_key = f"scenario1_{'_'.join(algorithm_names)}_{allocation_strategy}"
        self.simulation_results[scenario_key] = all_scenario_results
        
        return all_scenario_results

    def scenario_2_strategy_comparison(self, algorithm_names: Union[str, List[str]],
                                    budget_multiplier: float = 1.0,
                                    strategies: Optional[List[str]] = None,
                                    n_years: int = 50):
        """
        情况2: 选定算法(们)，所有预算分配策略对比
        
        Args:
            algorithm_names: 算法名称或算法列表
            budget_multiplier: 预算缩放因子
            strategies: 要对比的策略列表，None表示使用所有策略
            n_years: 仿真年数
        """
        # ensurealgorithm_names是列表
        if isinstance(algorithm_names, str):
            algorithm_names = [algorithm_names]
        
        if strategies is None:
            strategies = self.available_strategies
            
        print(f"\n{'='*80}")
        print(f"情况2: 预算分配策略对比实验")
        print(f"算法: {algorithm_names}")
        print(f"预算缩放因子: {budget_multiplier}")
        print(f"对比策略: {strategies}")
        print(f"{'='*80}")
        
        all_scenario_results = {}
        
        # for每个算法分别执行原有逻辑
        for algorithm_name in algorithm_names:
            print(f"\n🔹 算法: {algorithm_name}")
            scenario_results = {}
            
            for strategy in strategies:
                print(f"\n--- 预算分配策略: {strategy} ({self.strategy_descriptions.get(strategy, '')}) ---")
                
                try:
                    # create仿真器
                    simulator = BridgeMaintenanceSimulator(
                        model_path=self.find_latest_model(algorithm_name),
                        algorithm_name=algorithm_name,
                        test_data_path=self.test_data_path,
                        env_info_path=self.env_info_path,
                        action_costs=self.action_costs,
                        episode_idx=self.episode_idx,
                        initial_health_level=self.initial_health_level,
                        budget_multiplier=budget_multiplier,
                        budget_allocation_strategy=strategy
                    )
                    
                    # 运行仿真
                    results = simulator.run_simulation(n_years=n_years)
                    
                    # store result
                    scenario_results[strategy] = {
                        'simulator': simulator,
                        'results': results,
                        'strategy': strategy
                    }
                    
                    print(f"策略 {strategy} 仿真完成")
                    
                except Exception as e:
                    print(f"策略 {strategy} 仿真失败: {e}")
                    continue
            
            # 存储单个算法的结果
            all_scenario_results[algorithm_name] = scenario_results
            
            # 为单个算法绘制对比图表
            if scenario_results:
                self.plot_strategy_comparison(scenario_results, algorithm_name, budget_multiplier)
                scenario_key = f"scenario2_{algorithm_name}_x{budget_multiplier}"
                self.save_scenario_results(scenario_results, scenario_key)
        
        # if有多个算法，绘制算法间对比图
        if len(algorithm_names) > 1:
            self.plot_multi_algorithm_strategy_comparison(all_scenario_results, strategies, budget_multiplier)
        
        # 存储到主结果中
        scenario_key = f"scenario2_{'_'.join(algorithm_names)}_x{budget_multiplier}"
        self.simulation_results[scenario_key] = all_scenario_results
        
        return all_scenario_results


    def scenario_3_algorithm_comparison(self, algorithms: Optional[List[str]] = None,
                                      budget_multiplier: float = 1.0,
                                      allocation_strategy: str = 'original',
                                      n_years: int = 50):
        """
        情况3: 给定预算因子和分配策略下的算法对比
        
        Args:
            algorithms: 要对比的算法列表，None表示使用所有算法
            budget_multiplier: 预算缩放因子
            allocation_strategy: 预算分配策略
            n_years: 仿真年数
        """
        if algorithms is None:
            algorithms = self.available_algorithms
            
        print(f"\n{'='*80}")
        print(f"情况3: 算法对比实验")
        print(f"算法列表: {algorithms}")
        print(f"预算缩放因子: {budget_multiplier}")
        print(f"分配策略: {allocation_strategy}")
        print(f"{'='*80}")
        
        scenario_results = {}
        
        for algorithm in algorithms:
            print(f"\n--- 算法: {algorithm} ---")
            
            try:
                # 查找模型
                model_path = self.find_latest_model(algorithm)
                if model_path is None:
                    print(f"未找到算法 {algorithm} 的模型，跳过")
                    continue
                
                # create仿真器
                simulator = BridgeMaintenanceSimulator(
                    model_path=model_path,
                    algorithm_name=algorithm,
                    test_data_path=self.test_data_path,
                    env_info_path=self.env_info_path,
                    action_costs=self.action_costs,
                    episode_idx=self.episode_idx,
                    initial_health_level=self.initial_health_level,
                    budget_multiplier=budget_multiplier,
                    budget_allocation_strategy=allocation_strategy
                )
                
                # 运行仿真
                results = simulator.run_simulation(n_years=n_years)
                
                # store result
                scenario_results[algorithm] = {
                    'simulator': simulator,
                    'results': results,
                    'algorithm': algorithm
                }
                
                print(f"算法 {algorithm} 仿真完成")
                
            except Exception as e:
                print(f"算法 {algorithm} 仿真失败: {e}")
                continue
        
        # 存储到主结果中
        scenario_key = f"scenario3_{allocation_strategy}_x{budget_multiplier}"
        self.simulation_results[scenario_key] = scenario_results
        
        # 绘制对比图表
        if scenario_results:
            self.plot_algorithm_comparison(scenario_results, budget_multiplier, allocation_strategy)
            self.save_scenario_results(scenario_results, scenario_key)
        
        return scenario_results
    
    def find_latest_model(self, algorithm_name: str) -> Optional[str]:
        """查找指定算法的最新模型"""
        try:
            from bridge_maintenance_simulator_v1 import find_latest_model
            print(f"找到算法 {algorithm_name} 的模型")
            return find_latest_model(algorithm_name=algorithm_name)
        except:
            print(f"无法找到算法 {algorithm_name} 的模型")
            exit(0)
            return None
    

    def plot_multi_algorithm_budget_scaling(self, all_results: Dict, budget_multipliers: List[float], strategy: str):
        """绘制多算法预算缩放对比图"""
        # 设置全英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # create2×2布局图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Algorithm Budget Scaling Comparison ({strategy})', 
                    fontsize=16, fontweight='bold')
        
        algorithms = list(all_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        for algo_idx, (algorithm, results) in enumerate(all_results.items()):
            # 提取数据
            multipliers = []
            avg_healths = []
            total_costs = []
            cost_efficiencies = []
            
            for key, result_data in results.items():
                multiplier = result_data['multiplier']
                results_dict = result_data['results']
                summary_stats = results_dict.get('summary_statistics', {})
                
                multipliers.append(multiplier)
                avg_healths.append(summary_stats.get('avg_health_over_time', 0))
                total_costs.append(summary_stats.get('total_cost', 0))
                
                # 效率指标
                efficiency_metrics = summary_stats.get('efficiency_metrics', {})
                cost_per_health = efficiency_metrics.get('cost_per_health_point', 0)
                if cost_per_health > 0:
                    cost_efficiencies.append(1 / cost_per_health * 10000)
                else:
                    cost_efficiencies.append(0)
            
            # 排序数据
            sorted_indices = np.argsort(multipliers)
            multipliers = [multipliers[i] for i in sorted_indices]
            avg_healths = [avg_healths[i] for i in sorted_indices]
            total_costs = [total_costs[i] for i in sorted_indices]
            cost_efficiencies = [cost_efficiencies[i] for i in sorted_indices]
            
            # 绘制四个子图
            axes[0, 0].plot(multipliers, avg_healths, 'o-', label=algorithm, 
                        color=colors[algo_idx], linewidth=2, markersize=6)
            axes[0, 1].plot(multipliers, total_costs, 'o-', label=algorithm, 
                        color=colors[algo_idx], linewidth=2, markersize=6)
            axes[1, 0].plot(multipliers, cost_efficiencies, 'o-', label=algorithm, 
                        color=colors[algo_idx], linewidth=2, markersize=6)
        
        # 设置图表标题和标签
        axes[0, 0].set_title('Average Health vs Budget', fontweight='bold')
        axes[0, 0].set_xlabel('Budget Multiplier')
        axes[0, 0].set_ylabel('Average Health Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Total Cost vs Budget', fontweight='bold')
        axes[0, 1].set_xlabel('Budget Multiplier')
        axes[0, 1].set_ylabel('Total Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Cost Efficiency vs Budget', fontweight='bold')
        axes[1, 0].set_xlabel('Budget Multiplier')
        axes[1, 0].set_ylabel('Efficiency Score (×10000)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # fourth个子图：算法排名对比
        axes[1, 1].set_title('Algorithm Performance Ranking', fontweight='bold')
        # here可以根据平均健康度或效率进行排名显示
        # 简化处理：显示最后一个预算下的性能对比
        final_healths = []
        for algorithm in algorithms:
            results = all_results[algorithm]
            max_mult_key = max(results.keys(), key=lambda x: results[x]['multiplier'])
            final_health = results[max_mult_key]['results']['summary_statistics'].get('avg_health_over_time', 0)
            final_healths.append(final_health)
        
        bars = axes[1, 1].bar(algorithms, final_healths, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Final Average Health')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_algorithm_budget_scaling_{strategy}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Multi-algorithm budget scaling chart saved: {filepath}")
        plt.show()

    def plot_multi_algorithm_strategy_comparison(self, all_results: Dict, strategies: List[str], budget_multiplier: float):
        """绘制多算法策略对比图"""
        # 设置全英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # create2×2布局图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Algorithm Strategy Comparison (Budget×{budget_multiplier})', 
                    fontsize=16, fontweight='bold')
        
        algorithms = list(all_results.keys())
        n_strategies = len(strategies)
        n_algorithms = len(algorithms)
        
        # 准备数据矩阵
        health_matrix = np.zeros((n_algorithms, n_strategies))
        cost_matrix = np.zeros((n_algorithms, n_strategies))
        efficiency_matrix = np.zeros((n_algorithms, n_strategies))
        
        for algo_idx, (algorithm, results) in enumerate(all_results.items()):
            for strat_idx, strategy in enumerate(strategies):
                if strategy in results:
                    summary_stats = results[strategy]['results']['summary_statistics']
                    health_matrix[algo_idx, strat_idx] = summary_stats.get('avg_health_over_time', 0)
                    cost_matrix[algo_idx, strat_idx] = summary_stats.get('total_cost', 0)
                    
                    efficiency_metrics = summary_stats.get('efficiency_metrics', {})
                    cost_per_health = efficiency_metrics.get('cost_per_health_point', 0)
                    if cost_per_health > 0:
                        efficiency_matrix[algo_idx, strat_idx] = 1 / cost_per_health * 10000
        
        # 绘制热力图
        strategy_names = [self.strategy_descriptions.get(s, s)[:10] for s in strategies]
        
        # 健康度热力图
        im1 = axes[0, 0].imshow(health_matrix, cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_title('Average Health Heatmap', fontweight='bold')
        axes[0, 0].set_xticks(range(n_strategies))
        axes[0, 0].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(n_algorithms))
        axes[0, 0].set_yticklabels(algorithms)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 成本热力图
        im2 = axes[0, 1].imshow(cost_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0, 1].set_title('Total Cost Heatmap', fontweight='bold')
        axes[0, 1].set_xticks(range(n_strategies))
        axes[0, 1].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[0, 1].set_yticks(range(n_algorithms))
        axes[0, 1].set_yticklabels(algorithms)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 效率热力图
        im3 = axes[1, 0].imshow(efficiency_matrix, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Efficiency Heatmap', fontweight='bold')
        axes[1, 0].set_xticks(range(n_strategies))
        axes[1, 0].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(n_algorithms))
        axes[1, 0].set_yticklabels(algorithms)
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 综合排名
        # according to健康度和效率的综合评分进行排名
        axes[1, 1].set_title('Overall Performance Ranking', fontweight='bold')
        combined_scores = []
        for algo_idx, algorithm in enumerate(algorithms):
            avg_health = np.mean(health_matrix[algo_idx, :])
            avg_efficiency = np.mean(efficiency_matrix[algo_idx, :])
            # simple的综合评分
            combined_score = avg_health * 0.6 + avg_efficiency * 0.4 / 1000  # normalize效率分数
            combined_scores.append(combined_score)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        bars = axes[1, 1].bar(algorithms, combined_scores, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_algorithm_strategy_comparison_x{budget_multiplier}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Multi-algorithm strategy comparison chart saved: {filepath}")
        plt.show()

    def plot_budget_scaling_comparison(self, results: Dict, algorithm_name: str, strategy: str):
        """绘制预算缩放因子对比图表（简化专业版）"""
        if not results:
            return
            
        # 设置全英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # create4个子图（2×2布局）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Budget Scaling Analysis - {algorithm_name} ({strategy})', 
                    fontsize=16, fontweight='bold')
        
        # 提取和排序数据
        multipliers = []
        metrics_data = {
            'avg_healths': [], 'total_costs': [], 'health_histories': [],
            'cost_histories': [], 'budget_utilizations': [], 'cost_efficiencies': [],
            'action_distributions': []
        }
        
        for key, result_data in results.items():
            multiplier = result_data['multiplier']
            results_dict = result_data['results']
            summary_stats = results_dict.get('summary_statistics', {})
            
            multipliers.append(multiplier)
            
            # 基础指标
            metrics_data['avg_healths'].append(summary_stats.get('avg_health_over_time', 0))
            metrics_data['total_costs'].append(summary_stats.get('total_cost', 0))
            
            # 时间序列数据
            health_histories = results_dict['health_histories']
            annual_avg_health = [np.mean(health) for health in health_histories]
            metrics_data['health_histories'].append(annual_avg_health)
            
            cost_histories = results_dict['total_costs']
            metrics_data['cost_histories'].append(cost_histories)
            
            # 效率指标
            efficiency_metrics = summary_stats.get('efficiency_metrics', {})
            cost_per_health = efficiency_metrics.get('cost_per_health_point', 0)
            if cost_per_health > 0:
                metrics_data['cost_efficiencies'].append(1 / cost_per_health * 10000)  # 转换为效率指标
            else:
                metrics_data['cost_efficiencies'].append(0)
            
            # 经费使用率
            budget_stats = summary_stats.get('budget_statistics', {})
            total_allocated = budget_stats.get('total_budget_allocated', 1)
            actual_used = summary_stats.get('total_cost', 0)
            utilization = min(actual_used / total_allocated, 1.0) if total_allocated > 0 else 0
            metrics_data['budget_utilizations'].append(utilization)
            
            # action分布
            action_stats = summary_stats.get('action_statistics', {})
            metrics_data['action_distributions'].append({
                'no_action': action_stats.get('no_action_ratio', 0),
                'minor_repair': action_stats.get('minor_repair_ratio', 0),
                'medium_repair': action_stats.get('medium_repair_ratio', 0),
                'major_repair': action_stats.get('major_repair_ratio', 0)
            })
        
        # 排序所有数据
        sorted_indices = np.argsort(multipliers)
        multipliers = [multipliers[i] for i in sorted_indices]
        for key in ['avg_healths', 'total_costs', 'cost_efficiencies', 'budget_utilizations']:
            metrics_data[key] = [metrics_data[key][i] for i in sorted_indices]
        for key in ['health_histories', 'cost_histories', 'action_distributions']:
            metrics_data[key] = [metrics_data[key][i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(multipliers)))
        
        # 1. Health Evolution Over Time
        axes[0, 0].set_title('Health Evolution Over Time', fontweight='bold')
        for i, mult in enumerate(multipliers):
            years = list(range(1, len(metrics_data['health_histories'][i]) + 1))
            axes[0, 0].plot(years, metrics_data['health_histories'][i], 
                        label=f'{mult}x Budget', color=colors[i], linewidth=2)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Health Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Costs Over Time
        axes[0, 1].set_title('Costs Over Time', fontweight='bold')
        for i, mult in enumerate(multipliers):
            years = list(range(1, len(metrics_data['cost_histories'][i]) + 1))
            axes[0, 1].plot(years, metrics_data['cost_histories'][i], 
                        label=f'{mult}x Budget', color=colors[i], linewidth=2)
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Annual Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total Cost vs Budget
        axes[1, 0].set_title('Total Cost vs Budget', fontweight='bold')
        axes[1, 0].plot(multipliers, metrics_data['total_costs'], 'bo-', linewidth=3, markersize=8)
        axes[1, 0].set_xlabel('Budget Multiplier')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].grid(True, alpha=0.3)
        # add数值标签
        for i, (mult, cost) in enumerate(zip(multipliers, metrics_data['total_costs'])):
            axes[1, 0].annotate(f'{cost:.0f}', (mult, cost), 
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        # 4. Cost Efficiency
        axes[1, 1].set_title('Cost Efficiency', fontweight='bold')
        axes[1, 1].plot(multipliers, metrics_data['cost_efficiencies'], 'ro-', linewidth=3, markersize=8)
        axes[1, 1].set_xlabel('Budget Multiplier')
        axes[1, 1].set_ylabel('Efficiency Score (×10000)')
        axes[1, 1].grid(True, alpha=0.3)
        # add数值标签
        for i, (mult, eff) in enumerate(zip(multipliers, metrics_data['cost_efficiencies'])):
            axes[1, 1].annotate(f'{eff:.3f}', (mult, eff), 
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"budget_scaling_analysis_{algorithm_name}_{strategy}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Budget scaling analysis chart saved: {filepath}")
        plt.show()
        
        # output对比表格
        self._print_budget_scaling_table(multipliers, metrics_data, algorithm_name, strategy)

    def plot_strategy_comparison(self, results: Dict, algorithm_name: str, budget_multiplier: float):
        """绘制策略对比图表（简化专业版）"""
        if not results:
            return
            
        # 设置全英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # create4个子图（2×2布局）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategy Comparison - {algorithm_name} (Budget×{budget_multiplier})', 
                    fontsize=16, fontweight='bold')
        
        # 提取数据
        strategies = list(results.keys())
        strategy_names = [self.strategy_descriptions.get(s, s) for s in strategies]
        
        metrics_data = {
            'avg_healths': [], 'total_costs': [], 'health_histories': [],
            'cost_histories': [], 'budget_utilizations': [], 'cost_efficiencies': [],
            'action_distributions': []
        }
        
        for strategy in strategies:
            result_data = results[strategy]
            results_dict = result_data['results']
            summary_stats = results_dict.get('summary_statistics', {})
            
            # 基础指标
            metrics_data['avg_healths'].append(summary_stats.get('avg_health_over_time', 0))
            metrics_data['total_costs'].append(summary_stats.get('total_cost', 0))
            
            # 时间序列数据
            health_histories = results_dict['health_histories']
            annual_avg_health = [np.mean(health) for health in health_histories]
            metrics_data['health_histories'].append(annual_avg_health)
            
            cost_histories = results_dict['total_costs']
            metrics_data['cost_histories'].append(cost_histories)
            
            # 效率指标
            efficiency_metrics = summary_stats.get('efficiency_metrics', {})
            cost_per_health = efficiency_metrics.get('cost_per_health_point', 0)
            if cost_per_health > 0:
                metrics_data['cost_efficiencies'].append(1 / cost_per_health * 10000)
            else:
                metrics_data['cost_efficiencies'].append(0)
            
            # 经费使用率
            budget_stats = summary_stats.get('budget_statistics', {})
            total_allocated = budget_stats.get('total_budget_allocated', 1)
            actual_used = summary_stats.get('total_cost', 0)
            utilization = min(actual_used / total_allocated, 1.0) if total_allocated > 0 else 0
            metrics_data['budget_utilizations'].append(utilization)
            
            # action分布
            action_stats = summary_stats.get('action_statistics', {})
            metrics_data['action_distributions'].append({
                'no_action': action_stats.get('no_action_ratio', 0),
                'minor_repair': action_stats.get('minor_repair_ratio', 0),
                'medium_repair': action_stats.get('medium_repair_ratio', 0),
                'major_repair': action_stats.get('major_repair_ratio', 0)
            })
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        # 1. Health Evolution Over Time
        axes[0, 0].set_title('Health Evolution Over Time', fontweight='bold')
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            years = list(range(1, len(metrics_data['health_histories'][i]) + 1))
            axes[0, 0].plot(years, metrics_data['health_histories'][i], 
                        label=name[:15], color=colors[i], linewidth=2)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Health Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Costs Over Time
        axes[0, 1].set_title('Costs Over Time', fontweight='bold')
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            years = list(range(1, len(metrics_data['cost_histories'][i]) + 1))
            axes[0, 1].plot(years, metrics_data['cost_histories'][i], 
                        label=name[:15], color=colors[i], linewidth=2)
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Annual Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total Cost by Strategy
        axes[1, 0].set_title('Total Cost by Strategy', fontweight='bold')
        bars = axes[1, 0].bar(range(len(strategies)), metrics_data['total_costs'], 
                            color=colors, alpha=0.8)
        axes[1, 0].set_xticks(range(len(strategies)))
        axes[1, 0].set_xticklabels([name[:10] for name in strategy_names], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].grid(True, alpha=0.3)
        # add数值标签
        for bar, cost in zip(bars, metrics_data['total_costs']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['total_costs'])*0.01,
                        f'{cost:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Cost Efficiency by Strategy
        axes[1, 1].set_title('Cost Efficiency by Strategy', fontweight='bold')
        bars = axes[1, 1].bar(range(len(strategies)), metrics_data['cost_efficiencies'], 
                            color=colors, alpha=0.8)
        axes[1, 1].set_xticks(range(len(strategies)))
        axes[1, 1].set_xticklabels([name[:10] for name in strategy_names], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Efficiency Score (×10000)')
        axes[1, 1].grid(True, alpha=0.3)
        # add数值标签
        for bar, eff in zip(bars, metrics_data['cost_efficiencies']):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['cost_efficiencies'])*0.01,
                        f'{eff:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_comparison_{algorithm_name}_x{budget_multiplier}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Strategy comparison chart saved: {filepath}")
        plt.show()
        
        # output对比表格
        self._print_strategy_comparison_table(strategies, strategy_names, metrics_data, algorithm_name, budget_multiplier)

    def plot_algorithm_comparison(self, results: Dict, budget_multiplier: float, strategy: str):
        """绘制算法对比图表（简化专业版）"""
        if not results:
            return
            
        # 设置全英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # create4个子图（2×2布局）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Algorithm Comparison - {strategy} (Budget×{budget_multiplier})', 
                    fontsize=16, fontweight='bold')
        
        # 提取数据
        algorithms = list(results.keys())
        
        metrics_data = {
            'avg_healths': [], 'total_costs': [], 'health_histories': [],
            'cost_histories': [], 'budget_utilizations': [], 'cost_efficiencies': [],
            'action_distributions': []
        }
        
        for algorithm in algorithms:
            result_data = results[algorithm]
            results_dict = result_data['results']
            summary_stats = results_dict.get('summary_statistics', {})
            
            # 基础指标
            metrics_data['avg_healths'].append(summary_stats.get('avg_health_over_time', 0))
            metrics_data['total_costs'].append(summary_stats.get('total_cost', 0))
            
            # 时间序列数据
            health_histories = results_dict['health_histories']
            annual_avg_health = [np.mean(health) for health in health_histories]
            metrics_data['health_histories'].append(annual_avg_health)
            
            cost_histories = results_dict['total_costs']
            metrics_data['cost_histories'].append(cost_histories)
            
            # 效率指标
            efficiency_metrics = summary_stats.get('efficiency_metrics', {})
            cost_per_health = efficiency_metrics.get('cost_per_health_point', 0)
            if cost_per_health > 0:
                metrics_data['cost_efficiencies'].append(1 / cost_per_health * 10000)
            else:
                metrics_data['cost_efficiencies'].append(0)
            
            # 经费使用率
            budget_stats = summary_stats.get('budget_statistics', {})
            total_allocated = budget_stats.get('total_budget_allocated', 1)
            actual_used = summary_stats.get('total_cost', 0)
            utilization = min(actual_used / total_allocated, 1.0) if total_allocated > 0 else 0
            metrics_data['budget_utilizations'].append(utilization)
            
            # action分布
            action_stats = summary_stats.get('action_statistics', {})
            metrics_data['action_distributions'].append({
                'no_action': action_stats.get('no_action_ratio', 0),
                'minor_repair': action_stats.get('minor_repair_ratio', 0),
                'medium_repair': action_stats.get('medium_repair_ratio', 0),
                'major_repair': action_stats.get('major_repair_ratio', 0)
            })
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        
        # 1. Health Evolution Over Time
        axes[0, 0].set_title('Health Evolution Over Time', fontweight='bold')
        for i, algorithm in enumerate(algorithms):
            years = list(range(1, len(metrics_data['health_histories'][i]) + 1))
            axes[0, 0].plot(years, metrics_data['health_histories'][i], 
                        label=algorithm, color=colors[i], linewidth=2)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Health Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Costs Over Time
        axes[0, 1].set_title('Costs Over Time', fontweight='bold')
        for i, algorithm in enumerate(algorithms):
            years = list(range(1, len(metrics_data['cost_histories'][i]) + 1))
            axes[0, 1].plot(years, metrics_data['cost_histories'][i], 
                        label=algorithm, color=colors[i], linewidth=2)
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Annual Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total Cost by Algorithm
        axes[1, 0].set_title('Total Cost by Algorithm', fontweight='bold')
        bars = axes[1, 0].bar(range(len(algorithms)), metrics_data['total_costs'], 
                            color=colors, alpha=0.8)
        axes[1, 0].set_xticks(range(len(algorithms)))
        axes[1, 0].set_xticklabels(algorithms, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].grid(True, alpha=0.3)
        # add数值标签
        for bar, cost in zip(bars, metrics_data['total_costs']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['total_costs'])*0.01,
                        f'{cost:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Cost Efficiency by Algorithm
        axes[1, 1].set_title('Cost Efficiency by Algorithm', fontweight='bold')
        bars = axes[1, 1].bar(range(len(algorithms)), metrics_data['cost_efficiencies'], 
                            color=colors, alpha=0.8)
        axes[1, 1].set_xticks(range(len(algorithms)))
        axes[1, 1].set_xticklabels(algorithms, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Efficiency Score (×10000)')
        axes[1, 1].grid(True, alpha=0.3)
        # add数值标签
        for bar, eff in zip(bars, metrics_data['cost_efficiencies']):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['cost_efficiencies'])*0.01,
                        f'{eff:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"algorithm_comparison_{strategy}_x{budget_multiplier}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Algorithm comparison chart saved: {filepath}")
        plt.show()
        
        # output对比表格
        self._print_algorithm_comparison_table(algorithms, metrics_data, budget_multiplier, strategy)

    # 同时修改表格输出函数，删除经费使用率列
    def _print_budget_scaling_table(self, multipliers, metrics_data, algorithm_name, strategy):
        """打印预算缩放对比表格"""
        
        print(f"\n{'='*90}")
        print(f"BUDGET SCALING ANALYSIS - {algorithm_name.upper()} ({strategy.upper()})")
        print(f"{'='*90}")
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'-'*90}")
        print(f"{'Budget':>8} {'Avg Health':>12} {'Total Cost':>12} {'Efficiency':>12}")
        print(f"{'-'*90}")
        
        for i, mult in enumerate(multipliers):
            print(f"{mult:>6.1f}x {metrics_data['avg_healths'][i]:>11.3f} "
                f"{metrics_data['total_costs'][i]:>11.0f} {metrics_data['cost_efficiencies'][i]:>11.1f}")
        
        print(f"\n🔧 ACTION DISTRIBUTION")
        print(f"{'-'*90}")
        print(f"{'Budget':>8} {'No Action':>12} {'Minor':>12} {'Medium':>12} {'Major':>12}")
        print(f"{'-'*90}")
        
        for i, mult in enumerate(multipliers):
            actions = metrics_data['action_distributions'][i]
            print(f"{mult:>6.1f}x {actions['no_action']:>11.1%} "
                f"{actions['minor_repair']:>11.1%} {actions['medium_repair']:>11.1%} "
                f"{actions['major_repair']:>11.1%}")

    def _print_strategy_comparison_table(self, strategies, strategy_names, metrics_data, algorithm_name, budget_multiplier):
        """打印策略对比表格"""
        
        print(f"\n{'='*110}")
        print(f"STRATEGY COMPARISON - {algorithm_name.upper()} (BUDGET×{budget_multiplier})")
        print(f"{'='*110}")
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'-'*110}")
        print(f"{'Strategy':<20} {'Avg Health':>12} {'Total Cost':>12} {'Efficiency':>12}")
        print(f"{'-'*110}")
        
        for i, name in enumerate(strategy_names):
            print(f"{name:<20} {metrics_data['avg_healths'][i]:>11.3f} "
                f"{metrics_data['total_costs'][i]:>11.0f} {metrics_data['cost_efficiencies'][i]:>11.1f}")
        
        print(f"\n🔧 ACTION DISTRIBUTION")
        print(f"{'-'*110}")
        print(f"{'Strategy':<20} {'No Action':>12} {'Minor':>12} {'Medium':>12} {'Major':>12}")
        print(f"{'-'*110}")
        
        for i, name in enumerate(strategy_names):
            actions = metrics_data['action_distributions'][i]
            print(f"{name:<20} {actions['no_action']:>11.1%} "
                f"{actions['minor_repair']:>11.1%} {actions['medium_repair']:>11.1%} "
                f"{actions['major_repair']:>11.1%}")

    def _print_algorithm_comparison_table(self, algorithms, metrics_data, budget_multiplier, strategy):
        """打印算法对比表格"""
        
        print(f"\n{'='*90}")
        print(f"ALGORITHM COMPARISON - {strategy.upper()} (BUDGET×{budget_multiplier})")
        print(f"{'='*90}")
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'-'*90}")
        print(f"{'Algorithm':<20} {'Avg Health':>12} {'Total Cost':>12} {'Efficiency':>12}")
        print(f"{'-'*90}")
        
        for i, algorithm in enumerate(algorithms):
            print(f"{algorithm:<20} {metrics_data['avg_healths'][i]:>11.3f} "
                f"{metrics_data['total_costs'][i]:>11.0f} {metrics_data['cost_efficiencies'][i]:>11.1f}")
        
        print(f"\n🔧 ACTION DISTRIBUTION")
        print(f"{'-'*90}")
        print(f"{'Algorithm':<20} {'No Action':>12} {'Minor':>12} {'Medium':>12} {'Major':>12}")
        print(f"{'-'*90}")
        
        for i, algorithm in enumerate(algorithms):
            actions = metrics_data['action_distributions'][i]
            print(f"{algorithm:<20} {actions['no_action']:>11.1%} "
                f"{actions['minor_repair']:>11.1%} {actions['medium_repair']:>11.1%} "
                f"{actions['major_repair']:>11.1%}")
            """打印算法对比表格"""
            
            print(f"\n{'='*100}")
            print(f"ALGORITHM COMPARISON - {strategy.upper()} (BUDGET×{budget_multiplier})")
            print(f"{'='*100}")
            
            print(f"\n📊 PERFORMANCE SUMMARY")
            print(f"{'-'*100}")
            print(f"{'Algorithm':<20} {'Avg Health':>12} {'Total Cost':>12} {'Efficiency':>12} {'Utilization':>12}")
            print(f"{'-'*100}")
            
            for i, algorithm in enumerate(algorithms):
                print(f"{algorithm:<20} {metrics_data['avg_healths'][i]:>11.3f} "
                    f"{metrics_data['total_costs'][i]:>11.0f} {metrics_data['cost_efficiencies'][i]:>11.1f} "
                    f"{metrics_data['budget_utilizations'][i]:>11.1%}")
            
            print(f"\n🔧 ACTION DISTRIBUTION")
            print(f"{'-'*100}")
            print(f"{'Algorithm':<20} {'No Action':>12} {'Minor':>12} {'Medium':>12} {'Major':>12}")
            print(f"{'-'*100}")
            
            for i, algorithm in enumerate(algorithms):
                actions = metrics_data['action_distributions'][i]
                print(f"{algorithm:<20} {actions['no_action']:>11.1%} "
                    f"{actions['minor_repair']:>11.1%} {actions['medium_repair']:>11.1%} "
                    f"{actions['major_repair']:>11.1%}")

    def save_scenario_results(self, results: Dict, scenario_key: str):
        """保存仿真场景结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_key}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备保存的数据（移除不能序列化的对象）
        save_data = {}
        for key, result_data in results.items():
            save_data[key] = {
                'results': result_data['results'],
                'parameters': {
                    'budget_multiplier': result_data.get('multiplier'),
                    'strategy': result_data.get('strategy'),
                    'algorithm': result_data.get('algorithm')
                }
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"仿真结果已保存: {filepath}")
    
    def run_interactive_mode(self):
        """交互式运行模式"""
        print("\n" + "="*80)
        print("🚧 高级桥梁维修仿真系统 🚧")
        print("="*80)
        
        while True:
            print("\n请选择仿真模式:")
            print("1. 预算缩放因子实验 (选定算法，不同预算)")
            print("2. 预算分配策略对比 (选定算法，不同策略)")
            print("3. 算法对比实验 (选定策略和预算，不同算法)")
            print("4. 查看可用选项")
            print("5. 退出")
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                self._interactive_scenario_1()
            elif choice == '2':
                self._interactive_scenario_2()
            elif choice == '3':
                self._interactive_scenario_3()
            elif choice == '4':
                self._show_available_options()
            elif choice == '5':
                print("感谢使用桥梁维修仿真系统！")
                break
            else:
                print("无效选择，请重新输入。")
    
    def _interactive_scenario_1(self):
        """交互式场景1"""
        print("\n--- 预算缩放因子实验 ---")
        
        # select算法（支持多个）
        print(f"可用算法: {self.available_algorithms}")
        print("单个算法直接输入名称，多个算法用逗号分隔")
        algorithm_input = input("请选择算法: ").strip()
        algorithms = [a.strip() for a in algorithm_input.split(',') if a.strip() in self.available_algorithms]
        
        if not algorithms:
            print("未选择有效算法")
            return
        
        # select策略
        print(f"可用策略: {self.available_strategies}")
        strategy = input("请选择预算分配策略 (默认: original): ").strip() or 'original'
        if strategy not in self.available_strategies:
            print(f"无效策略，使用默认策略 original")
            strategy = 'original'
        
        # budget因子
        multipliers_input = input("请输入预算缩放因子 (用逗号分隔，默认: 0.25,0.5,1.0,2.0,4.0): ").strip()
        if multipliers_input:
            try:
                multipliers = [float(x.strip()) for x in multipliers_input.split(',')]
            except:
                print("无效输入，使用默认值")
                multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        else:
            multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        # 仿真年数
        years_input = input("请输入仿真年数 (默认: 50): ").strip()
        n_years = int(years_input) if years_input.isdigit() else 50
        
        print(f"\n开始仿真: {algorithms}, {strategy}, {multipliers}, {n_years}年")
        self.scenario_1_budget_scaling(algorithms, multipliers, strategy, n_years)

    def _interactive_scenario_2(self):
        """交互式场景2"""
        print("\n--- 预算分配策略对比 ---")
        
        # select算法（支持多个）
        print(f"可用算法: {self.available_algorithms}")
        print("单个算法直接输入名称，多个算法用逗号分隔")
        algorithm_input = input("请选择算法: ").strip()
        algorithms = [a.strip() for a in algorithm_input.split(',') if a.strip() in self.available_algorithms]
        
        if not algorithms:
            print("未选择有效算法")
            return
        
        # budget因子
        multiplier_input = input("请输入预算缩放因子 (默认: 1.0): ").strip()
        multiplier = float(multiplier_input) if multiplier_input else 1.0
        
        # select策略
        print(f"可用策略: {self.available_strategies}")
        print("直接回车使用所有策略，或用逗号分隔输入特定策略")
        strategies_input = input("请选择策略: ").strip()
        if strategies_input:
            strategies = [s.strip() for s in strategies_input.split(',')]
            strategies = [s for s in strategies if s in self.available_strategies]
        else:
            strategies = None
        
        # 仿真年数
        years_input = input("请输入仿真年数 (默认: 50): ").strip()
        n_years = int(years_input) if years_input.isdigit() else 50
        
        print(f"\n开始仿真: {algorithms}, x{multiplier}, {strategies or '所有策略'}, {n_years}年")
        self.scenario_2_strategy_comparison(algorithms, multiplier, strategies, n_years)
    
    def _interactive_scenario_3(self):
        """交互式场景3"""
        print("\n--- 算法对比实验 ---")
        
        # select算法
        print(f"可用算法: {self.available_algorithms}")
        print("直接回车使用所有算法，或用逗号分隔输入特定算法")
        algorithms_input = input("请选择算法: ").strip()
        if algorithms_input:
            algorithms = [a.strip() for a in algorithms_input.split(',')]
            algorithms = [a for a in algorithms if a in self.available_algorithms]
        else:
            algorithms = None
        
        # budget因子
        multiplier_input = input("请输入预算缩放因子 (默认: 1.0): ").strip()
        multiplier = float(multiplier_input) if multiplier_input else 1.0
        
        # select策略
        print(f"可用策略: {self.available_strategies}")
        strategy = input("请选择预算分配策略 (默认: original): ").strip() or 'original'
        if strategy not in self.available_strategies:
            print(f"无效策略，使用默认策略 original")
            strategy = 'original'
        
        # 仿真年数
        years_input = input("请输入仿真年数 (默认: 50): ").strip()
        n_years = int(years_input) if years_input.isdigit() else 50
        
        print(f"\n开始仿真: {algorithms or '所有算法'}, x{multiplier}, {strategy}, {n_years}年")
        self.scenario_3_algorithm_comparison(algorithms, multiplier, strategy, n_years)
    
    def _show_available_options(self):
        """显示可用选项"""
        print("\n--- 可用选项 ---")
        print(f"算法: {self.available_algorithms}")
        print(f"策略数量: {len(self.available_strategies)}")
        print("\n策略详情:")
        for strategy, desc in self.strategy_descriptions.items():
            print(f"  {strategy}: {desc}")


# use示例
def main():
    """主函数 - 演示各种使用方式"""
    
    # create高级仿真器
    simulator = AdvancedMultiAlgorithmSimulator(
        test_data_path="marl/data_benchmark/episodes/test_buffer.pt",
        env_info_path="marl/data_benchmark/episodes/train_env_info.json",
        action_costs={0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03},
        episode_idx=0,
        initial_health_level=9,
        output_dir="advanced_simulation_results"
    )
    
    # 运行交互式模式
    simulator.run_interactive_mode()
    
    # or者直接运行特定场景（示例）
    # scenario_1_example(simulator)
    # scenario_2_example(simulator)
    # scenario_3_example(simulator)

def scenario_1_example(simulator):
    """场景1示例：预算缩放实验"""
    print("=== 场景1示例：multitask_cpq算法在不同预算下的表现 ===")
    simulator.scenario_1_budget_scaling(
        algorithm_name="multitask_cpq",
        budget_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        allocation_strategy='critical_first',
        n_years=30
    )

def scenario_2_example(simulator):
    """场景2示例：策略对比实验"""
    print("=== 场景2示例：multitask_cpq算法在不同策略下的表现 ===")
    simulator.scenario_2_strategy_comparison(
        algorithm_name="multitask_cpq",
        budget_multiplier=1.0,
        strategies=['original', 'critical_first', 'importance_top10', 'preventive_maintenance'],
        n_years=30
    )

def scenario_3_example(simulator):
    """场景3示例：算法对比实验"""
    print("=== 场景3示例：不同算法在critical_first策略下的表现 ===")
    simulator.scenario_3_algorithm_comparison(
        algorithms=["multitask_cpq", "random_osrl"],
        budget_multiplier=1.0,
        allocation_strategy='critical_first',
        n_years=30
    )

if __name__ == "__main__":
    main()