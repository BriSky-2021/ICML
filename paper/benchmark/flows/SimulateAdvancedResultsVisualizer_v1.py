import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import re
from collections import defaultdict


class AdvancedResultsVisualizer:
    def __init__(self, results_dir: str = "advanced_simulation_results", output_dir: str = "enhanced_visualization_results"):
        """
        高级仿真结果可视化器
        
        Args:
            results_dir: 仿真结果目录
            output_dir: 可视化输出目录
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # create输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置matplotlib参数
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        
        # ✅ 平滑参数配置 - 在文件开头预设不同图表的平滑参数
        self.smoothing_config = {
            'health_evolution': {
                'method': 'none',
                'params': {
                    'window_length': 7,
                    'polyorder': 2,
                    'sigma': 1.5
                },
                'enabled': False
            },
            'cost_evolution': {
                'method': 'gaussian',
                'params': {
                    'window_length': 5,
                    'polyorder': 2,
                    'sigma': 1.5
                },
                'enabled': True
            },
            'efficiency_bars': {
                'method': 'none',
                'params': {},
                'enabled': False  # 柱状图不需要平滑
            }
        }
        
        # ✅ 显示配置
        self.display_config = {
            'efficiency_multiplier': 100,  # 效率指标乘以100显示
            'show_data_points': True,      #whether显示原始数据点
            'point_sampling_rate': 0.3     # data点采样率（0-1）
        }

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
        
        # already知的算法列表（用于更好的解析）
        self.known_algorithms = [
            'multitask_cpq', 'multitask_bc', 'cdt', 'random_osrl', 
            'discrete_bc_50', 'iqlcql_marl', 'qmix_cql'
        ]
        
        # already知的策略列表
        self.known_strategies = list(self.strategy_descriptions.keys())
        
        # 分析可用的结果文件
        self.available_files = self._analyze_available_files()
        
        print(f"可视化器初始化完成")
        print(f"结果目录: {results_dir}")
        print(f"输出目录: {output_dir}")
    
    def _safe_float(self, value, default=0.0):
        """安全转换为浮点数"""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        except (ValueError, TypeError):
            print(f"警告: 无法转换 '{value}' 为浮点数，使用默认值 {default}")
            return default

    def debug_file_analysis(self):
        """调试文件分析，显示实际文件名和解析结果"""
        print(f"\n{'='*80}")
        print("🔍 文件名调试分析")
        print(f"{'='*80}")
        
        # get所有JSON文件
        json_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        if not json_files:
            print("❌ 未找到任何JSON文件")
            return
        
        print(f"📁 找到 {len(json_files)} 个JSON文件:")
        print(f"{'-'*80}")
        
        parsed_count = 0
        unparsed_files = []
        
        for i, filename in enumerate(json_files, 1):
            print(f"\n{i}. 文件名: {filename}")
            
            # 尝试解析
            file_info = self._parse_filename(filename)
            
            if file_info:
                parsed_count += 1
                print(f"   ✅ 解析成功:")
                print(f"      场景: {file_info['scenario']}")
                print(f"      算法: {file_info.get('algorithm', 'N/A')}")
                print(f"      策略: {file_info.get('strategy', 'N/A')}")
                print(f"      预算倍数: {file_info.get('budget_multiplier', 'N/A')}")
                print(f"      时间戳: {file_info.get('timestamp', 'N/A')}")
            else:
                unparsed_files.append(filename)
                print(f"   ❌ 解析失败")
                
                # 尝试手动分析文件名模式
                print(f"      文件名模式分析:")
                parts = filename.replace('.json', '').split('_')
                print(f"      分割部分: {parts}")
                
                # check是否匹配已知模式
                self._analyze_filename_pattern(filename)
        
        print(f"\n{'='*80}")
        print(f"📊 解析统计:")
        print(f"   总文件数: {len(json_files)}")
        print(f"   成功解析: {parsed_count}")
        print(f"   解析失败: {len(unparsed_files)}")
        
        if unparsed_files:
            print(f"\n❌ 未能解析的文件:")
            for file in unparsed_files:
                print(f"   - {file}")
    
    def _analyze_filename_pattern(self, filename: str):
        """分析单个文件名的模式"""
        name = filename.replace('.json', '')
        parts = name.split('_')
        
        print(f"      详细分析:")
        print(f"        总部分数: {len(parts)}")
        print(f"        各部分: {parts}")
        
        # check时间戳（最后两部分应该是日期和时间）
        if len(parts) >= 2:
            date_part = parts[-2]
            time_part = parts[-1]
            timestamp = f"{date_part}_{time_part}"
            
            if re.match(r'\d{8}_\d{6}', timestamp):
                print(f"        时间戳: ✅ {timestamp}")
                
                # 分析其余部分
                remaining_parts = parts[:-2]
                print(f"        剩余部分: {remaining_parts}")
                
                if len(remaining_parts) >= 3:
                    scenario = remaining_parts[0]
                    print(f"        场景: {scenario}")
                    
                    # 尝试找到已知策略
                    strategy_found = None
                    for strategy in self.known_strategies:
                        if strategy in remaining_parts:
                            strategy_found = strategy
                            strategy_idx = remaining_parts.index(strategy)
                            break
                    
                    if strategy_found:
                        print(f"        找到策略: {strategy_found} (位置: {strategy_idx})")
                        algorithm_parts = remaining_parts[1:strategy_idx]
                        algorithm = '_'.join(algorithm_parts)
                        print(f"        推测算法: {algorithm}")
                    else:
                        print(f"        未找到已知策略")
            else:
                print(f"        时间戳: ❌ {timestamp}")
    
    def _analyze_available_files(self) -> Dict:
        """分析可用的结果文件，按场景和算法分组"""
        files_info = {
            'scenario1': defaultdict(list),  # budget缩放实验
            'scenario2': defaultdict(list),  # 策略对比实验
            'scenario3': defaultdict(list),  # algorithm对比实验
        }
        
        # get所有JSON文件
        json_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        print(f"正在分析 {len(json_files)} 个JSON文件...")
        
        for filename in json_files:
            # 解析文件名
            file_info = self._parse_filename(filename)
            if file_info:
                scenario = file_info['scenario']
                
                # according to场景类型确定分组键
                if scenario == 'scenario1' or scenario == 'scenario2':
                    algorithm = file_info['algorithm']
                    if algorithm:  # ensure算法名不为空
                        files_info[scenario][algorithm].append({
                            'filename': filename,
                            'timestamp': file_info['timestamp'],
                            'full_info': file_info
                        })
                elif scenario == 'scenario3':
                    # scenario3是多算法对比，以策略为键
                    strategy = file_info['strategy']
                    if strategy:
                        files_info[scenario][strategy].append({
                            'filename': filename,
                            'timestamp': file_info['timestamp'],
                            'full_info': file_info
                        })
            else:
                print(f"⚠️  无法解析文件: {filename}")
        
        # for每个算法的文件按时间戳排序
        for scenario in files_info:
            for key in files_info[scenario]:
                files_info[scenario][key].sort(
                    key=lambda x: x['timestamp'], reverse=True
                )
        
        return files_info
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """
        解析文件名，提取场景、算法等信息
        支持复杂的算法名，如: qmix_cql, iqlcql_marl, discrete_bc_50
        
        文件名格式:
        scenario1_algorithm_strategy_timestamp.json
        scenario2_algorithm_xbudget_timestamp.json  
        scenario3_strategy_xbudget_timestamp.json
        """
        # 移除.json后缀
        name = filename.replace('.json', '')
        parts = name.split('_')
        
        if len(parts) < 4:
            return None
        
        # 提取时间戳（最后两部分：日期_时间）
        if len(parts) >= 2:
            date_part = parts[-2]
            time_part = parts[-1]
            timestamp = f"{date_part}_{time_part}"
            
            # 验证时间戳格式
            if not re.match(r'\d{8}_\d{6}', timestamp):
                return None
        else:
            return None
        
        # get场景
        scenario = parts[0]
        if scenario not in ['scenario1', 'scenario2', 'scenario3']:
            return None
        
        # 剩余部分（去除scenario和时间戳）
        middle_parts = parts[1:-2]
        
        if scenario == 'scenario1':
            # scenario1_algorithm_strategy_date_time
            #requires找到策略，策略之前的都是算法名
            
            strategy_found = None
            strategy_idx = -1
            
            # from后往前查找已知策略
            for i in range(len(middle_parts) - 1, -1, -1):
                if middle_parts[i] in self.known_strategies:
                    strategy_found = middle_parts[i]
                    strategy_idx = i
                    break
            
            if strategy_found and strategy_idx > 0:
                algorithm_parts = middle_parts[:strategy_idx]
                algorithm = '_'.join(algorithm_parts)
                
                return {
                    'scenario': scenario,
                    'algorithm': algorithm,
                    'strategy': strategy_found,
                    'timestamp': timestamp,
                    'budget_multiplier': None
                }
        
        elif scenario == 'scenario2':
            # scenario2_algorithm_xbudget_date_time
            requires找到以x开头的预算部分
            
            budget_idx = -1
            budget_multiplier = None
            
            # 查找预算部分
            for i, part in enumerate(middle_parts):
                if part.startswith('x') and len(part) > 1:
                    try:
                        budget_multiplier = float(part[1:])
                        budget_idx = i
                        break
                    except:
                        continue
            
            if budget_idx > 0 and budget_multiplier is not None:
                algorithm_parts = middle_parts[:budget_idx]
                algorithm = '_'.join(algorithm_parts)
                
                return {
                    'scenario': scenario,
                    'algorithm': algorithm,
                    'strategy': None,
                    'timestamp': timestamp,
                    'budget_multiplier': budget_multiplier
                }
        
        elif scenario == 'scenario3':
            # scenario3_strategy_xbudget_date_time
            requires找到以x开头的预算部分
            
            budget_idx = -1
            budget_multiplier = None
            
            # 查找预算部分
            for i, part in enumerate(middle_parts):
                if part.startswith('x') and len(part) > 1:
                    try:
                        budget_multiplier = float(part[1:])
                        budget_idx = i
                        break
                    except:
                        continue
            
            if budget_idx > 0 and budget_multiplier is not None:
                strategy_parts = middle_parts[:budget_idx]
                strategy = '_'.join(strategy_parts)
                
                return {
                    'scenario': scenario,
                    'algorithm': None,
                    'strategy': strategy,
                    'timestamp': timestamp,
                    'budget_multiplier': budget_multiplier
                }
        
        return None
    
    def list_available_scenarios_and_algorithms(self):
        """列出可用的场景和算法"""
        print(f"\n{'='*60}")
        print("📋 可用的仿真结果分析")
        print(f"{'='*60}")
        
        for scenario in ['scenario1', 'scenario2', 'scenario3']:
            scenario_names = {
                'scenario1': '预算缩放实验',
                'scenario2': '策略对比实验', 
                'scenario3': '算法对比实验'
            }
            
            print(f"\n🔹 {scenario_names[scenario]} ({scenario}):")
            
            if scenario in self.available_files and self.available_files[scenario]:
                for key, files in self.available_files[scenario].items():
                    latest_file = files[0] if files else None
                    if latest_file:
                        file_info = latest_file['full_info']
                        if scenario == 'scenario3':
                            print(f"  📁 策略: {key}")
                        else:
                            print(f"  📁 算法: {key}")
                        print(f"     最新文件: {latest_file['filename']}")
                        print(f"     时间戳: {latest_file['timestamp']}")
                        if file_info.get('strategy'):
                            print(f"     策略: {file_info['strategy']}")
                        if file_info.get('budget_multiplier'):
                            print(f"     预算倍数: {file_info['budget_multiplier']}")
                        print(f"     共有文件: {len(files)} 个")
            else:
                print("  ❌ 未找到相关文件")
    

    def _ensure_numeric_data(self, data, data_name=""):
        """确保数据为数值类型"""
        if isinstance(data, (list, tuple)):
            try:
                return [float(x) for x in data]
            except (ValueError, TypeError) as e:
                print(f"警告: {data_name} 数据转换失败: {e}")
                return [0.0] * len(data)
        elif isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, str):
            try:
                return float(data)
            except:
                print(f"警告: 无法转换字符串 '{data}' 为数值")
                return 0.0
        else:
            print(f"警告: 未知数据类型 {type(data)}")
            return 0.0

    def _safe_extract_metrics(self, results_data, key_path):
        """安全提取指标数据"""
        try:
            # by路径提取数据
            current = results_data
            for key in key_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            
            # ensure为数值类型
            if isinstance(current, (list, tuple)):
                return self._ensure_numeric_data(current, '.'.join(key_path))
            else:
                return self._ensure_numeric_data(current, '.'.join(key_path))
        
        except Exception as e:
            print(f"警告: 提取 {'.'.join(key_path)} 失败: {e}")
            return None

    def apply_smoothing(self, data: List[float], chart_type: str = 'health_evolution') -> List[float]:
        """
        对数据应用平滑处理（基于预设配置）
        
        Args:
            data: 原始数据
            chart_type: 图表类型 ('health_evolution', 'cost_evolution', 'efficiency_bars')
        
        Returns:
            平滑后的数据
        """
        if len(data) < 3:
            return data
        
        # get该图表类型的平滑配置
        config = self.smoothing_config.get(chart_type, self.smoothing_config['health_evolution'])
        
        if not config['enabled'] or config['method'] == 'none':
            return data
        
        method = config['method']
        params = config['params']
        
        try:
            # will数据转换为浮点数数组
            data_array = np.array([float(x) for x in data], dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"警告: 数据转换失败，使用原始数据: {e}")
            return data
        
        try:
            if method == 'savgol':
                window_length = params.get('window_length', 5)
                polyorder = params.get('polyorder', 2)
                
                # ensure窗口长度合适
                if len(data) < window_length:
                    window_length = max(3, len(data) // 2)
                    if window_length % 2 == 0:  # ensure是奇数
                        window_length -= 1
                
                # ensure多项式阶数合适
                if window_length <= polyorder:
                    polyorder = max(1, window_length - 1)
                
                return savgol_filter(data_array, window_length, polyorder).tolist()
                
            elif method == 'gaussian':
                sigma = params.get('sigma', 1.0)
                return gaussian_filter1d(data_array, sigma=sigma).tolist()
                
            elif method == 'moving_average':
                window_length = params.get('window_length', 5)
                if window_length >= len(data):
                    window_length = max(3, len(data) // 3)
                
                smoothed = []
                for i in range(len(data)):
                    start_idx = max(0, i - window_length // 2)
                    end_idx = min(len(data), i + window_length // 2 + 1)
                    smoothed.append(np.mean(data_array[start_idx:end_idx]))
                return smoothed
                
            else:
                print(f"警告: 未知的平滑方法 '{method}'，使用原始数据")
                return data
                
        except Exception as e:
            print(f"警告: {method}平滑失败: {e}")
            return data

    def _parse_health_histories_string(self, health_str):
        """
        解析健康历史字符串，转换为二维数组
        
        Args:
            health_str: 字符串格式的健康历史数据
            
        Returns:
            numpy.ndarray: [n_years, n_bridges] 的健康状态数组
        """
        try:
            # 移除首尾的方括号
            if health_str.startswith('[') and health_str.endswith(']'):
                health_str = health_str[1:-1]
            
            # replace换行符为空格，并分割数值
            health_str = health_str.replace('\n', ' ')
            values = health_str.split()
            
            # 转换为浮点数数组
            health_values = np.array([float(val) for val in values if val.strip()])
            
            return health_values
            
        except Exception as e:
            print(f"警告: 解析健康历史字符串失败: {e}")
            return np.array([])

    def _parse_health_histories(self, health_histories, actual_n_agents=None):
        """
        解析健康历史数据，计算每年的平均健康状态
        
        Args:
            health_histories: 健康历史数据（可能是字符串列表或数值列表）
            actual_n_agents: 实际桥梁数量
            
        Returns:
            list: 每年平均健康状态列表
        """
        annual_avg_health = []
        
        try:
            if not health_histories:
                return [0.0]
            
            for i, health_data in enumerate(health_histories):
                if isinstance(health_data, str):
                    # 字符串格式，需要解析
                    health_array = self._parse_health_histories_string(health_data)
                    if len(health_array) > 0:
                        if actual_n_agents and len(health_array) >= actual_n_agents:
                            # if知道实际桥梁数量，只计算前 actual_n_agents 个桥梁的平均值
                            avg_health = np.mean(health_array[:actual_n_agents])
                        else:
                            # else计算所有桥梁的平均值
                            avg_health = np.mean(health_array)
                        annual_avg_health.append(float(avg_health))
                    else:
                        annual_avg_health.append(0.0)
                        
                elif isinstance(health_data, (list, tuple, np.ndarray)):
                    # 数值格式
                    health_numeric = self._ensure_numeric_data(health_data, f"health_history_{i}")
                    if health_numeric:
                        if actual_n_agents and len(health_numeric) >= actual_n_agents:
                            avg_health = np.mean(health_numeric[:actual_n_agents])
                        else:
                            avg_health = np.mean(health_numeric)
                        annual_avg_health.append(float(avg_health))
                    else:
                        annual_avg_health.append(0.0)
                else:
                    print(f"警告: 未知的健康历史数据格式: {type(health_data)}")
                    annual_avg_health.append(0.0)
                    
        except Exception as e:
            print(f"警告: 解析健康历史数据失败: {e}")
            return [0.0]
        
        return annual_avg_health if annual_avg_health else [0.0]

    def _parse_cost_histories(self, cost_histories):
        """
        解析成本历史数据
        
        Args:
            cost_histories: 成本历史数据
            
        Returns:
            list: 每年成本列表
        """
        try:
            if not cost_histories:
                return [0.0]
            
            if isinstance(cost_histories, str):
                # if是字符串格式，尝试解析
                cost_array = self._parse_health_histories_string(cost_histories)
                return cost_array.tolist() if len(cost_array) > 0 else [0.0]
            elif isinstance(cost_histories, (list, tuple)):
                # ensure为数值列表
                return self._ensure_numeric_data(cost_histories, "cost_histories")
            else:
                print(f"警告: 未知的成本历史数据格式: {type(cost_histories)}")
                return [0.0]
                
        except Exception as e:
            print(f"警告: 解析成本历史数据失败: {e}")
            return [0.0]

    def load_scenario_results(self, scenario_file: str) -> Dict:
        """加载场景结果文件"""
        filepath = os.path.join(self.results_dir, scenario_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"已加载结果文件: {scenario_file}")
        return data
    
    def plot_budget_scaling_enhanced(self, results_data: Dict, algorithm_name: str, 
                               allocation_strategy: str):
        """绘制增强版预算缩放对比图（1行3列布局）- 使用预设平滑配置"""
        
        print(f"\n开始绘制预算缩放分析: {algorithm_name} - {allocation_strategy}")
        
        # create1行3列布局
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Budget Scaling Analysis - {algorithm_name} ({allocation_strategy})', 
                    fontsize=16, fontweight='bold')
        
        # 提取和排序数据
        multipliers = []
        metrics_data = {
            'avg_healths': [], 'total_costs': [], 'health_histories': [],
            'cost_histories': [], 'cost_efficiencies': []
        }
        
        for key, result_data in results_data.items():
            try:
                print(f"\n处理数据键: {key}")
                
                # 安全提取预算倍数
                multiplier = result_data.get('parameters', {}).get('budget_multiplier')
                if multiplier is None:
                    multiplier = result_data.get('results', {}).get('summary_statistics', {}).get('budget_multiplier', 1.0)
                
                multiplier = self._safe_float(multiplier, 1.0)
                results_dict = result_data.get('results', {})
                summary_stats = results_dict.get('summary_statistics', {})
                
                multipliers.append(multiplier)
                
                # 安全提取基础指标
                avg_health = self._safe_extract_metrics(summary_stats, ['avg_health_over_time']) or 0
                total_cost = self._safe_extract_metrics(summary_stats, ['total_cost']) or 0
                
                metrics_data['avg_healths'].append(avg_health)
                metrics_data['total_costs'].append(total_cost)
                
                # get实际桥梁数量
                actual_n_agents = summary_stats.get('active_bridges', summary_stats.get('total_bridges', None))
                actual_n_agents = int(actual_n_agents) if actual_n_agents is not None else None
                
                # 解析健康历史数据
                health_histories = results_dict.get('health_histories', [])
                annual_avg_health = self._parse_health_histories(health_histories, actual_n_agents)
                metrics_data['health_histories'].append(annual_avg_health)
                
                # 解析成本历史数据
                cost_histories = results_dict.get('total_costs', [])
                cost_histories_parsed = self._parse_cost_histories(cost_histories)
                metrics_data['cost_histories'].append(cost_histories_parsed)
                
                # ✅ 效率指标计算 - 应用显示倍数
                efficiency_metrics = summary_stats.get('efficiency_metrics', {})
                cost_per_health = self._safe_extract_metrics(efficiency_metrics, ['cost_per_health_point']) or 0
                if cost_per_health > 0:
                    # 效率 = (1 / 单位健康成本) * 显示倍数
                    efficiency = (1 / cost_per_health) * self.display_config['efficiency_multiplier']
                else:
                    efficiency = 0
                metrics_data['cost_efficiencies'].append(efficiency)
                    
            except Exception as e:
                print(f"警告: 处理数据键 {key} 时出错: {e}")
                continue
        
        if not multipliers:
            print("错误: 没有有效的数据可以绘制")
            return
        
        # 排序所有数据
        try:
            sorted_indices = np.argsort(multipliers)
            multipliers = [multipliers[i] for i in sorted_indices]
            for key in ['avg_healths', 'total_costs', 'cost_efficiencies']:
                metrics_data[key] = [metrics_data[key][i] for i in sorted_indices]
            for key in ['health_histories', 'cost_histories']:
                metrics_data[key] = [metrics_data[key][i] for i in sorted_indices]
        except Exception as e:
            print(f"警告: 数据排序失败: {e}")
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(multipliers)))
        
        # ✅ 1. Health Evolution Over Time (左图) - 使用健康演化平滑配置
        axes[0].set_title('Health Evolution Over Time', fontweight='bold', fontsize=14)
        for i, mult in enumerate(multipliers):
            try:
                health_data = metrics_data['health_histories'][i]
                if health_data and len(health_data) > 0:
                    years = list(range(1, len(health_data) + 1))
                    original_data = health_data
                    
                    # ✅ 使用预设的健康演化平滑配置
                    smoothed_data = self.apply_smoothing(original_data, 'health_evolution')
                    
                    axes[0].plot(years, smoothed_data, label=f'{mult}x Budget', 
                                color=colors[i], linewidth=2.5, alpha=0.8)
                    
                    # ✅ 可选显示原始数据点
                    if self.display_config['show_data_points'] and len(years) <= 30:
                        sample_rate = self.display_config['point_sampling_rate']
                        step = max(1, int(1 / sample_rate))
                        axes[0].scatter(years[::step], original_data[::step], 
                                    color=colors[i], alpha=0.4, s=15)
            except Exception as e:
                print(f"警告: 绘制健康演化图时出错 (multiplier={mult}): {e}")
        
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Average Health Level', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=3.50)
        
        # ✅ 2. Costs Over Time (中图) - 使用成本演化平滑配置
        axes[1].set_title('Costs Over Time', fontweight='bold', fontsize=14)
        for i, mult in enumerate(multipliers):
            try:
                cost_data = metrics_data['cost_histories'][i]
                if cost_data and len(cost_data) > 0:
                    years = list(range(1, len(cost_data) + 1))
                    original_data = cost_data
                    
                    # ✅ 使用预设的成本演化平滑配置
                    smoothed_data = self.apply_smoothing(original_data, 'cost_evolution')
                    
                    axes[1].plot(years, smoothed_data, label=f'{mult}x Budget', 
                                color=colors[i], linewidth=2.5, alpha=0.8)
                    
                    # optional显示原始数据点
                    if self.display_config['show_data_points'] and len(years) <= 30:
                        sample_rate = self.display_config['point_sampling_rate']
                        step = max(1, int(1 / sample_rate))
                        axes[1].scatter(years[::step], original_data[::step], 
                                    color=colors[i], alpha=0.4, s=15)
            except Exception as e:
                print(f"警告: 绘制成本演化图时出错 (multiplier={mult}): {e}")
        
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Annual Cost', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)
        
        # ✅ 3. Combined Cost & Efficiency (右图) - 效率乘以配置倍数显示
        axes[2].set_title('Total Cost vs Efficiency', fontweight='bold', fontsize=14)
        
        # create双y轴
        ax2_twin = axes[2].twinx()
        
        # 柱状图显示总成本
        x_pos = np.arange(len(multipliers))
        bars = axes[2].bar(x_pos, metrics_data['total_costs'], 
                        color=colors, alpha=0.7, width=0.6, label='Total Cost')
        
        # at柱状图上添加数值标签
        for bar, cost in zip(bars, metrics_data['total_costs']):
            axes[2].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + max(metrics_data['total_costs'])*0.02,
                        f'{cost:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 折线图显示效率（已经乘以 display_config['efficiency_multiplier']）
        line = ax2_twin.plot(x_pos, metrics_data['cost_efficiencies'], 
                        'ro-', linewidth=3, markersize=8, color='red', 
                        label=f'Efficiency (×{self.display_config["efficiency_multiplier"]})', alpha=0.8)
        
        # at折线图上添加数值标签
        for x, eff in zip(x_pos, metrics_data['cost_efficiencies']):
            eff100=100000*eff
            ax2_twin.annotate(f'{eff100:.2f}', (x, eff), 
                            textcoords="offset points", xytext=(0,15), 
                            ha='center', fontsize=9, fontweight='bold', color='red')
        
        # 设置坐标轴
        axes[2].set_xlabel('Budget Multiplier', fontsize=12)
        axes[2].set_ylabel('Total Cost', fontsize=12, color='black')
        ax2_twin.set_ylabel(f'Efficiency Score (×{self.display_config["efficiency_multiplier"]})', 
                        fontsize=12, color='red')
        
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([f'{m}x' for m in multipliers])
        axes[2].grid(True, alpha=0.3)
        
        # 图例
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_budget_scaling_{algorithm_name}_{allocation_strategy}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Enhanced budget scaling chart saved: {filepath}")
        plt.show()

    def plot_strategy_comparison_enhanced(self, results_data: Dict, algorithm_name: str, 
                                        budget_multiplier: float):
        """
        绘制增强版策略对比图（1行3列布局）- 使用预设平滑配置
        *** THIS FUNCTION HAS BEEN MODIFIED TO BE CONSISTENT WITH plot_budget_scaling_enhanced ***
        """
        print(f"\n开始绘制策略对比分析: {algorithm_name} - 预算倍数 x{budget_multiplier}")

        # create1行3列布局
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Strategy Comparison - {algorithm_name} (Budget ×{budget_multiplier})', 
                    fontsize=16, fontweight='bold')
        
        # 提取数据
        strategies = list(results_data.keys())
        strategy_names = [self.strategy_descriptions.get(s, s) for s in strategies]
        
        metrics_data = {
            'avg_healths': [], 'total_costs': [], 'health_histories': [],
            'cost_histories': [], 'cost_efficiencies': []
        }
        
        for strategy in strategies:
            try:
                result_data = results_data[strategy]
                results_dict = result_data.get('results', {})
                summary_stats = results_dict.get('summary_statistics', {})
                
                # 安全提取基础指标
                metrics_data['avg_healths'].append(self._safe_extract_metrics(summary_stats, ['avg_health_over_time']) or 0)
                metrics_data['total_costs'].append(self._safe_extract_metrics(summary_stats, ['total_cost']) or 0)
                
                # get实际桥梁数量
                actual_n_agents = summary_stats.get('active_bridges', summary_stats.get('total_bridges', None))
                actual_n_agents = int(actual_n_agents) if actual_n_agents is not None else None
                
                # use健壮的解析器解析时间序列数据
                health_histories = results_dict.get('health_histories', [])
                annual_avg_health = self._parse_health_histories(health_histories, actual_n_agents)
                metrics_data['health_histories'].append(annual_avg_health)
                
                cost_histories = results_dict.get('total_costs', [])
                cost_histories_parsed = self._parse_cost_histories(cost_histories)
                metrics_data['cost_histories'].append(cost_histories_parsed)
                
                # ✅ 效率指标计算 - 应用显示倍数
                efficiency_metrics = summary_stats.get('efficiency_metrics', {})
                cost_per_health = self._safe_extract_metrics(efficiency_metrics, ['cost_per_health_point']) or 0
                if cost_per_health > 0:
                    efficiency = (1 / cost_per_health) * self.display_config['efficiency_multiplier']
                else:
                    efficiency = 0
                metrics_data['cost_efficiencies'].append(efficiency)

            except Exception as e:
                print(f"警告: 处理策略 {strategy} 时出错: {e}")
                # 附加空/默认值以保持列表对齐
                metrics_data['avg_healths'].append(0)
                metrics_data['total_costs'].append(0)
                metrics_data['health_histories'].append([])
                metrics_data['cost_histories'].append([])
                metrics_data['cost_efficiencies'].append(0)
                continue

        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        # ✅ 1. Health Evolution Over Time (左图)
        axes[0].set_title('Health Evolution Over Time', fontweight='bold', fontsize=14)
        for i, name in enumerate(strategy_names):
            try:
                health_data = metrics_data['health_histories'][i]
                if health_data and len(health_data) > 0:
                    years = list(range(1, len(health_data) + 1))
                    original_data = health_data
                    
                    # ✅ 使用预设的健康演化平滑配置
                    smoothed_data = self.apply_smoothing(original_data, 'health_evolution')
                    
                    axes[0].plot(years, smoothed_data, label=name[:15], 
                                color=colors[i], linewidth=2.5, alpha=0.8)
                    
                    # ✅ 可选显示原始数据点
                    if self.display_config['show_data_points'] and len(years) <= 30:
                        sample_rate = self.display_config['point_sampling_rate']
                        step = max(1, int(1 / sample_rate))
                        axes[0].scatter(years[::step], original_data[::step], 
                                      color=colors[i], alpha=0.4, s=15)
            except Exception as e:
                print(f"警告: 绘制策略 '{name}' 的健康演化图时出错: {e}")

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Average Health Level', fontsize=12)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        
        # ✅ 2. Costs Over Time (中图)
        axes[1].set_title('Costs Over Time', fontweight='bold', fontsize=14)
        for i, name in enumerate(strategy_names):
            try:
                cost_data = metrics_data['cost_histories'][i]
                if cost_data and len(cost_data) > 0:
                    years = list(range(1, len(cost_data) + 1))
                    original_data = cost_data
                    
                    # ✅ 使用预设的成本演化平滑配置
                    smoothed_data = self.apply_smoothing(original_data, 'cost_evolution')
                    
                    axes[1].plot(years, smoothed_data, label=name[:15], 
                                color=colors[i], linewidth=2.5, alpha=0.8)
                    
                    # ✅ 可选显示原始数据点
                    if self.display_config['show_data_points'] and len(years) <= 30:
                        sample_rate = self.display_config['point_sampling_rate']
                        step = max(1, int(1 / sample_rate))
                        axes[1].scatter(years[::step], original_data[::step], 
                                      color=colors[i], alpha=0.4, s=15)
            except Exception as e:
                print(f"警告: 绘制策略 '{name}' 的成本演化图时出错: {e}")

        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Annual Cost', fontsize=12)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)
        
        # ✅ 3. Combined Cost & Efficiency (右图)
        axes[2].set_title('Total Cost vs Efficiency', fontweight='bold', fontsize=14)
        
        # create双y轴
        ax2_twin = axes[2].twinx()
        
        # 柱状图显示总成本
        x_pos = np.arange(len(strategies))
        bars = axes[2].bar(x_pos, metrics_data['total_costs'], 
                          color=colors, alpha=0.7, width=0.6, label='Total Cost')
        
        # at柱状图上添加数值标签
        for bar, cost in zip(bars, metrics_data['total_costs']):
            axes[2].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + max(metrics_data['total_costs'])*0.02 if metrics_data['total_costs'] else 0,
                        f'{cost:.0f}', ha='center', va='bottom', fontsize=9, 
                        fontweight='bold')
        
        # 折线图显示效率
        line = ax2_twin.plot(x_pos, metrics_data['cost_efficiencies'], 
                           'ro-', linewidth=3, markersize=8, color='red', 
                           label=f'Efficiency (×{self.display_config["efficiency_multiplier"]})', alpha=0.8)
        
        # at折线图上添加数值标签
        for x, eff in zip(x_pos, metrics_data['cost_efficiencies']):
            eff100=100000*eff
            ax2_twin.annotate(f'{eff100:.2f}', (x, eff), 
                            textcoords="offset points", xytext=(0,15), 
                            ha='center', fontsize=9, fontweight='bold', color='red')
        
        # 设置坐标轴
        axes[2].set_xlabel('Strategy', fontsize=12)
        axes[2].set_ylabel('Total Cost', fontsize=12, color='black')
        ax2_twin.set_ylabel(f'Efficiency Score (×{self.display_config["efficiency_multiplier"]})', fontsize=12, color='red')
        
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([name[:10] for name in strategy_names], rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        # 图例
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # save图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_strategy_comparison_{algorithm_name}_x{budget_multiplier}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Enhanced strategy comparison chart saved: {filepath}")
        plt.show()
    
    def visualize_scenario_with_algorithm_selection(self, scenario: str, 
                                                   smoothing_method: str = 'savgol',
                                                   smoothing_params: Dict = None):
        """可视化指定场景，支持手动选择算法"""
        if scenario not in self.available_files:
            print(f"场景 {scenario} 没有可用文件")
            return
        
        available_keys = list(self.available_files[scenario].keys())
        if not available_keys:
            print(f"场景 {scenario} 没有可用的条目 (算法/策略)")
            return
        
        key_name = '策略' if scenario == 'scenario3' else '算法'
        print(f"\n场景 {scenario} 可用{key_name}:")
        for i, key in enumerate(available_keys, 1):
            files_count = len(self.available_files[scenario][key])
            latest_timestamp = self.available_files[scenario][key][0]['timestamp']
            print(f"{i}. {key} (共{files_count}个文件，最新: {latest_timestamp})")
        
        # select
        print(f"\n选择要可视化的{key_name}:")
        print(f"输入编号选择单个{key_name}，或用逗号分隔多个编号，或输入'all'选择全部")
        
        choice = input("请输入选择: ").strip().lower()
        
        selected_keys = []
        if choice == 'all':
            selected_keys = available_keys
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected_keys = [available_keys[i] for i in indices 
                                     if 0 <= i < len(available_keys)]
            except:
                print(f"无效输入，选择第一个{key_name}")
                selected_keys = [available_keys[0]]
        
        print(f"选择的{key_name}: {selected_keys}")
        
        # get每个算法/策略的最新文件并可视化
        for key in selected_keys:
            latest_file = self.available_files[scenario][key][0]['filename']
            print(f"\n处理 {key_name} {key}，文件: {latest_file}")
            
            try:
                # load数据
                data = self.load_scenario_results(latest_file)
                
                # according to场景类型调用相应的可视化函数
                if scenario == 'scenario1':
                    file_info = self.available_files[scenario][key][0]['full_info']
                    strategy = file_info['strategy']
                    self.plot_budget_scaling_enhanced(data, key, strategy)
                elif scenario == 'scenario2':
                    file_info = self.available_files[scenario][key][0]['full_info']
                    budget_multiplier = file_info['budget_multiplier']
                    # ✅ 移除平滑参数传递，函数已更新
                    self.plot_strategy_comparison_enhanced(data, key, budget_multiplier)
                else:
                    print(f"场景 {scenario} 暂不支持此方式的可视化")
                    
            except Exception as e:
                print(f"处理 {key_name} {key} 时出错: {e}")
                continue
    
    def interactive_visualize(self):
        """交互式可视化主界面"""
        print(f"\n{'='*60}")
        print("🎨 高级仿真结果可视化系统 🎨")
        print(f"{'='*60}")
        
        while True:
            print(f"\n主菜单:")
            print("1. 查看可用结果概览")
            print("2. 调试文件名解析")
            print("3. 可视化预算缩放实验 (Scenario 1)")
            print("4. 可视化策略对比实验 (Scenario 2)")
            print("5. 可视化算法对比实验 (Scenario 3)")
            print("6. 平滑参数设置说明")
            print("7. 退出")
            
            choice = input("\n请输入选择 (1-7): ").strip()
            
            if choice == '1':
                self.list_available_scenarios_and_algorithms()
            elif choice == '2':
                self.debug_file_analysis()
            elif choice == '3':
                self._interactive_scenario_visualization('scenario1')
            elif choice == '4':
                self._interactive_scenario_visualization('scenario2')
            elif choice == '5':
                print("Scenario 3 (算法对比) 功能开发中...")
            elif choice == '6':
                self._show_smoothing_help()
            elif choice == '7':
                print("退出可视化系统")
                break
            else:
                print("无效选择，请重新输入")
    
    def _interactive_scenario_visualization(self, scenario: str):
        """交互式场景可视化"""
        scenario_names = {
            'scenario1': '预算缩放实验',
            'scenario2': '策略对比实验'
        }
        
        print(f"\n--- {scenario_names[scenario]} 可视化 ---")
        print(f"✅ 使用预设平滑配置:")
        print(f"   健康演化: {self.smoothing_config['health_evolution']['method']} (启用: {self.smoothing_config['health_evolution']['enabled']})")
        print(f"   成本演化: {self.smoothing_config['cost_evolution']['method']} (启用: {self.smoothing_config['cost_evolution']['enabled']})")
        print(f"   效率显示倍数: {self.display_config['efficiency_multiplier']}x")
        
        # direct执行可视化，无需用户选择平滑参数
        self.visualize_scenario_with_algorithm_selection(scenario)
    
    def _get_smoothing_params(self, method: str) -> Dict:
        """获取平滑参数"""
        params = {}
        
        if method == 'savgol':
            window = input("窗口长度 (默认: 7): ").strip()
            params['window_length'] = int(window) if window.isdigit() else 7
            
            poly = input("多项式阶数 (默认: 2): ").strip()
            params['polyorder'] = int(poly) if poly.isdigit() else 2
            
        elif method == 'gaussian':
            sigma = input("高斯标准差 (默认: 1.5): ").strip()
            params['sigma'] = float(sigma) if sigma else 1.5
            
        elif method == 'moving_average':
            window = input("窗口长度 (默认: 5): ").strip()
            params['window_length'] = int(window) if window.isdigit() else 5
        
        return params
    
    def _show_smoothing_help(self):
        """显示平滑参数说明"""
        print(f"\n{'='*50}")
        print("📊 当前平滑配置")
        print(f"{'='*50}")
        
        for chart_type, config in self.smoothing_config.items():
            status = "启用" if config['enabled'] else "禁用"
            print(f"\n{chart_type.upper()}:")
            print(f"  状态: {status}")
            print(f"  方法: {config['method']}")
            if config['params']:
                print(f"  参数: {config['params']}")
        
        print(f"\n显示配置:")
        print(f"  效率显示倍数: {self.display_config['efficiency_multiplier']}x")
        print(f"  显示数据点: {self.display_config['show_data_points']}")
        print(f"  数据点采样率: {self.display_config['point_sampling_rate']}")
        
        print(f"\n💡 要修改配置，请编辑类初始化中的 smoothing_config 和 display_config")


def main():
    """主函数"""
    # create可视化器
    visualizer = AdvancedResultsVisualizer(
        results_dir="advanced_simulation_results",
        output_dir="enhanced_visualization_results"
    )
    
    # 启动交互式模式
    visualizer.interactive_visualize()

if __name__ == "__main__":
    main()