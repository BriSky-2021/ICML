"""
桥梁维修100年仿真测试
加载训练好的模型，进行100年的桥梁维修决策，并生成健康状态分布堆积图
使用真实测试数据集中的episode数据，只替换健康状态
"""

import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import defaultdict
import glob

from utils.transition_util import build_transition_matrices


def convert_to_serializable(obj):
    """
    将包含numpy类型的对象转换为JSON可序列化的格式
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    else:
        return obj

def extract_data_from_buffer(buffer):
    all_obs, all_actions, episode_budgets, all_health, all_raw_cost, all_rewards, all_log_budgets = [], [], [], [], [], [], []
    all_importance = []  # ✅ 新增：存储重要性信息
    actual_n_agents_list = []  # 新增：记录每个episode的实际智能体数量
    
    num_eps = buffer.episodes_in_buffer
    for i in range(num_eps):
        ep = buffer[i:i+1]
        obs = ep['obs'].cpu().numpy()           # [1, T, n_agents, state_dim]
        actions = ep['actions'].cpu().numpy()   # [1, T, n_agents] or [1, T, n_agents, 1]
        btotal = np.expm1(ep['btotal'].cpu().numpy()) if 'btotal' in ep.data.episode_data else 100000
        log_budgets = ep['log_budget'].cpu().numpy()   # [1, T, n_agents]
        
        # 记录实际智能体数量，并squeeze掉多余的维度
        actual_n_agents = ep['n_bridges_actual'].cpu().numpy().squeeze()  # 从 [1, 1] 变成标量
        actual_n_agents_list.append(actual_n_agents)
        
        all_log_budgets.append(log_budgets[0])
        all_obs.append(obs[0])
        all_actions.append(actions[0])
        episode_budgets.append(btotal)
        
        if 'health_state' in ep.data.transition_data:
            health = ep['health_state'].cpu().numpy()  # [1, T+1, n_agents]
            all_health.append(health[0])
        # ---- Raw cost修正：无论是否存在都append ----
        if 'raw_cost' in ep.data.transition_data:
            raw_cost = ep['raw_cost'].cpu().numpy()  # [1, T, n_agents] 或 [1, T, n_agents, 1]
            all_raw_cost.append(raw_cost[0])
        else:
            print(ep['raw_cost'].cpu().numpy().shape)
            # 注意 actions[0] shape 可能是 (T, n_agents) 或 (T, n_agents, 1)
            all_raw_cost.append(np.zeros_like(actions[0]))
        if 'reward' in ep.data.transition_data:
            rewards = ep['reward'].cpu().numpy()  # [1, T, n_agents]
            all_rewards.append(rewards[0])
        else:
            all_rewards.append(np.zeros_like(actions[0]))
        
        if 'importance' in ep.data.transition_data:
            importance = ep['importance'].cpu().numpy()  # [1, T, n_agents, 1]
            all_importance.append(importance[0])
        else:
            print("false to get importance")
            exit(0)
            all_importance.append(np.zeros_like(actions[0]))
    
    data = np.stack(all_obs)        # [num_eps, T, n_agents, state_dim]
    actions = np.stack(all_actions) # [num_eps, T, n_agents] 或 [num_eps, T, n_agents, 1]
    episode_budgets = np.array(episode_budgets).squeeze()
    health = np.stack(all_health) if all_health else None
    raw_cost = np.stack(all_raw_cost) if all_raw_cost else None
    rewards = np.stack(all_rewards) if all_rewards else None
    log_budgets = np.stack(all_log_budgets) if all_log_budgets else None
    actual_n_agents = np.array(actual_n_agents_list)  # [num_eps] - 现在应该是正确的形状
    importance = np.stack(all_importance) if all_importance else None  # [num_eps, n_agents]

    print(f'产生的ep的长度为{data.shape}')

    return data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents, importance

class BridgeMaintenanceSimulator:
    def __init__(self, model_path, algorithm_name, test_data_path, env_info_path=None, action_costs=None, 
                 episode_idx=0, initial_health_level=9, budget_multiplier=1.0, budget_allocation_strategy='original'):
        """
        初始化桥梁维修仿真器
        
        Args:
            model_path: 训练好的模型路径
            test_data_path: 测试数据集路径
            env_info_path: 环境信息文件路径（包含归一化参数）
            action_costs: 动作成本字典
            episode_idx: 使用的测试episode索引
            initial_health_level: 初始健康等级（0-9，对应原始健康值）
        """
        self.model_path = model_path
        self.algorithm_name = algorithm_name
        self.test_data_path = test_data_path
        self.episode_idx = episode_idx
        self.budget_multiplier = budget_multiplier  # ✅ 新增预算乘子
        self.budget_allocation_strategy = budget_allocation_strategy  # ✅ 新增预算分配策略
        self.model = self.load_model()
        
        # 加载测试数据
        self.test_data = self.load_test_data()

        
        # 加载环境信息和归一化参数
        self.env_info = self.load_env_info(env_info_path)
        self.norm_params = self.env_info.get('normalization_params', {})

        # ✅ 应用预算乘子处理
        if budget_multiplier != 1.0:
            self.apply_budget_multiplier()
        
        # ✅ 应用预算分配策略
        if budget_allocation_strategy != 'original':
            self.apply_budget_allocation_strategy()

        # 动作成本
        if action_costs is None:
            self.action_costs = {0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03}
        else:
            self.action_costs = action_costs
        
        # 健康转移矩阵
        self.transition_matrices = build_transition_matrices()
        
        # 初始健康等级（原始值，0-9）
        self.initial_health_level = initial_health_level
        
        # 健康状态映射
        self.health_categories = {
            'poor': (0, 3),      # 0-3
            'fair': (4, 5),      # 4-5
            'good': (6, 7),      # 6-7
            'excellent': (8, 9)  # 8-9
        }
        
        # ✅ 检测模型类型和数据需求
        self.model_type = self.detect_model_type()
        self.needs_budget_in_obs = self.check_budget_in_obs()
        
        print(f"检测到模型类型: {self.model_type}")
        print(f"模型是否需要预算在观察中: {self.needs_budget_in_obs}")


        # 仿真结果存储
        self.simulation_results = {
            'years': [],
            'health_distributions': [],
            'health_category_distributions': [],
            'total_costs': [],
            'action_distributions': [],
            'health_histories': [],
            'normalized_health_histories': []
        }
        
        # 添加序列历史存储
        self.sequence_history = {
            'observations': [],
            'actions': [],
            'returns_to_go': [],
            'costs_to_go': [],
            'time_steps': []
        }
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            model = th.load(self.model_path, map_location='cuda')
            print(f"成功加载模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    

    def load_test_data(self):
        """加载测试数据集"""
        try:
            # 使用与Multi_Task_Run_v3.py相同的方式加载测试数据
            from Multi_Task_Run_v3 import load_buffer
            
            device = th.device('cpu')
            test_buffer = load_buffer(self.test_data_path, device)
            test_data, test_actions, test_budgets, test_health, test_raw_cost, test_reward, test_log_budgets, test_actual_n_agents, test_importance = extract_data_from_buffer(test_buffer)
            
            # 选择指定的episode
            if self.episode_idx >= len(test_data):
                print(f"警告: episode索引 {self.episode_idx} 超出范围，使用第0个episode")
                self.episode_idx = 0
            
            # 提取指定episode的数据
            episode_data = {
                'data': test_data[self.episode_idx],  # [T, n_agents, state_dim]
                'actions': test_actions[self.episode_idx],  # [T, n_agents]
                'budgets': test_budgets[self.episode_idx],  # scalar
                'health': test_health[self.episode_idx],  # [T+1, n_agents]
                'raw_cost': test_raw_cost[self.episode_idx],  # [T, n_agents]
                'reward': test_reward[self.episode_idx],  # [T, n_agents]
                'log_budgets': test_log_budgets[self.episode_idx],  # [T, n_agents]
                'actual_n_agents': test_actual_n_agents[self.episode_idx],  # scalar
                'importance': test_importance[self.episode_idx]  # [T, n_agents]
            }
            
            print(f"成功加载测试数据，使用episode {self.episode_idx}")
            print(f"  - 数据形状: {episode_data['data'].shape}")
            print(f"  - 动作形状: {episode_data['actions'].shape}")
            print(f"  - 健康形状: {episode_data['health'].shape}")
            print(f"  - 实际智能体数量: {episode_data['actual_n_agents']}")
            print(f"  - 重要性形状: {episode_data['importance'].shape}")

            return episode_data
            
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            raise
    
    def apply_budget_multiplier(self):
        """
        应用预算乘子，修改测试数据中的预算值
        """
        print(f"应用预算乘子 {self.budget_multiplier}...")
        
        # 获取原始数据
        original_log_budgets = self.test_data['log_budgets'].copy()  # [T, n_agents, 1] or [T, n_agents]
        original_episode_budget = self.test_data['budgets']
        
        # 获取归一化参数
        if 'normalization_params' in self.env_info and 'log_budgets' in self.env_info['normalization_params']:
            log_budget_norm_params = self.env_info['normalization_params']['log_budgets']
            mean = log_budget_norm_params['mean']
            std = log_budget_norm_params['std']
        else:
            print("警告: 未找到log_budgets归一化参数，使用默认值")
            mean = 0.0
            std = 1.0
        
        # 处理数据维度
        if original_log_budgets.ndim == 3 and original_log_budgets.shape[-1] == 1:
            original_log_budgets = original_log_budgets.squeeze(-1)  # [T, n_agents]
        
        print(f"原始log_budgets形状: {original_log_budgets.shape}")
        print(f"原始episode预算: {original_episode_budget}")
        
        # 步骤1: 反归一化得到原始log预算值
        raw_log_budgets = original_log_budgets * std + mean
        
        # 步骤2: 转换为原始预算值（非log）
        raw_budgets = np.expm1(raw_log_budgets) 
        
        # 步骤3: 应用预算乘子
        scaled_budgets = raw_budgets * self.budget_multiplier
        
        # 步骤4: 重新log变换
        scaled_log_budgets = np.log1p(scaled_budgets)
        
        # 步骤5: 重新归一化
        new_log_budgets = (scaled_log_budgets - mean) / (std + 1e-8)
        
        # 更新测试数据
        if self.test_data['log_budgets'].ndim == 3:
            self.test_data['log_budgets'] = new_log_budgets[..., np.newaxis]  # 恢复 [T, n_agents, 1] 格式
        else:
            self.test_data['log_budgets'] = new_log_budgets
        
        # 更新episode预算
        new_episode_budget = original_episode_budget * self.budget_multiplier
        self.test_data['budgets'] = new_episode_budget
        
        # 打印统计信息
        print(f"预算乘子应用完成:")
        print(f"  原始预算范围: {np.min(raw_budgets):.2f} - {np.max(raw_budgets):.2f}")
        print(f"  调整后预算范围: {np.min(scaled_budgets):.2f} - {np.max(scaled_budgets):.2f}")
        print(f"  原始episode预算: {original_episode_budget:.2f}")
        print(f"  调整后episode预算: {new_episode_budget:.2f}")
        print(f"  原始log预算范围: {np.min(original_log_budgets):.4f} - {np.max(original_log_budgets):.4f}")
        print(f"  调整后log预算范围: {np.min(new_log_budgets):.4f} - {np.max(new_log_budgets):.4f}")
    
        #exit(0)

    def apply_budget_allocation_strategy(self):
        """
        应用预算分配策略
        """
        if self.budget_allocation_strategy == 'original':
            return
        
        print(f"应用预算分配策略: {self.budget_allocation_strategy}")
        
        # 获取原始数据
        original_log_budgets = self.test_data['log_budgets'].copy()
        importance_data = self.test_data.get('importance', None)
        actual_n_agents = self.test_data['actual_n_agents']
        
        # 获取健康状态数据（从观察数据中提取）
        obs_data = self.test_data['data']  # [T, n_agents, state_dim]
        health_states = obs_data[:, :, 0]  # 假设健康状态是第一个特征 [T, n_agents]
        
        # 获取归一化参数
        if hasattr(self, 'env_info') and 'normalization_params' in self.env_info and 'log_budgets' in self.env_info['normalization_params']:
            log_budget_norm_params = self.env_info['normalization_params']['log_budgets']
            mean = log_budget_norm_params['mean']
            std = log_budget_norm_params['std']
        else:
            mean, std = 0.0, 1.0
        
        # 处理数据维度
        if original_log_budgets.ndim == 3 and original_log_budgets.shape[-1] == 1:
            original_log_budgets = original_log_budgets.squeeze(-1)
        
        T, n_agents = original_log_budgets.shape
        
        # 反归一化和转换
        raw_log_budgets = original_log_budgets * std + mean
        raw_budgets = np.expm1(raw_log_budgets)
        total_budgets_per_timestep = np.sum(raw_budgets[:, :actual_n_agents], axis=1)
        
        # 根据策略重新分配预算
        new_raw_budgets = np.zeros_like(raw_budgets)
        
        if self.budget_allocation_strategy == 'uniform':
            # 原有的均分策略
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                uniform_budget = total_budget_t / actual_n_agents
                new_raw_budgets[t, :actual_n_agents] = uniform_budget
        
        elif self.budget_allocation_strategy == 'importance_top10':
            # 原有的前10%策略
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到均分策略")
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    uniform_budget = total_budget_t / actual_n_agents
                    new_raw_budgets[t, :actual_n_agents] = uniform_budget
            else:
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)
                
                top_k = max(1, int(np.ceil(actual_n_agents * 0.1)))
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    importance_t = importance_data[t, :actual_n_agents]
                    top_indices = np.argsort(importance_t)[-top_k:]
                    budget_per_top_bridge = total_budget_t / top_k
                    new_raw_budgets[t, top_indices] = budget_per_top_bridge
        
        # ✅ 新增：基于健康状态的工程实践策略
        elif self.budget_allocation_strategy == 'critical_first':
            """策略：危急优先 - 优先维修健康状态最差的桥梁"""
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                health_t = health_states[t, :actual_n_agents]
                
                # 按健康状态排序（从差到好）
                health_order = np.argsort(health_t)
                
                # 计算需要维修的桥梁数量（预算能覆盖的数量）
                avg_maintenance_cost = total_budget_t / actual_n_agents  # 平均维修成本
                target_bridges = min(max(1, int(actual_n_agents * 0.3)), actual_n_agents)  # 最多30%的桥梁
                
                # 将预算分配给最差的桥梁
                budget_per_bridge = total_budget_t / target_bridges
                critical_indices = health_order[:target_bridges]
                new_raw_budgets[t, critical_indices] = budget_per_bridge
        
        elif self.budget_allocation_strategy == 'threshold_based':
            """策略：阈值维修 - 只维修健康状态低于阈值的桥梁"""
            health_threshold = 0.3  # 健康状态阈值（归一化后）
            
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                health_t = health_states[t, :actual_n_agents]
                
                # 找到需要维修的桥梁（健康状态低于阈值）
                need_repair = health_t < health_threshold
                repair_indices = np.where(need_repair)[0]
                
                if len(repair_indices) > 0:
                    # 按健康状态排序，优先最差的
                    repair_health = health_t[repair_indices]
                    repair_order = np.argsort(repair_health)
                    sorted_repair_indices = repair_indices[repair_order]
                    
                    # 分配预算
                    budget_per_bridge = total_budget_t / len(repair_indices)
                    new_raw_budgets[t, sorted_repair_indices] = budget_per_bridge
                else:
                    # 如果没有桥梁需要紧急维修，选择健康状态最差的几个
                    worst_count = max(1, int(actual_n_agents * 0.1))
                    worst_indices = np.argsort(health_t)[:worst_count]
                    budget_per_bridge = total_budget_t / worst_count
                    new_raw_budgets[t, worst_indices] = budget_per_bridge
        
        elif self.budget_allocation_strategy == 'importance_health_combined':
            """策略：重要性+健康状态综合 - 考虑重要性和健康状态的加权组合"""
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到critical_first策略")
                # 回退到critical_first策略
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    health_t = health_states[t, :actual_n_agents]
                    health_order = np.argsort(health_t)
                    target_bridges = min(max(1, int(actual_n_agents * 0.3)), actual_n_agents)
                    budget_per_bridge = total_budget_t / target_bridges
                    critical_indices = health_order[:target_bridges]
                    new_raw_budgets[t, critical_indices] = budget_per_bridge
            else:
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)
                
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    health_t = health_states[t, :actual_n_agents]
                    importance_t = importance_data[t, :actual_n_agents]
                    
                    # 计算综合优先级：重要性权重0.6，健康紧急性权重0.4
                    health_urgency = 1.0 - health_t  # 健康状态越差，紧急性越高
                    
                    # 归一化重要性和紧急性
                    norm_importance = (importance_t - np.min(importance_t)) / (np.max(importance_t) - np.min(importance_t) + 1e-8)
                    norm_urgency = (health_urgency - np.min(health_urgency)) / (np.max(health_urgency) - np.min(health_urgency) + 1e-8)
                    
                    # 综合评分
                    combined_score = 0.6 * norm_importance + 0.4 * norm_urgency
                    
                    # 选择评分最高的桥梁进行维修
                    target_count = max(1, int(actual_n_agents * 0.25))  # 25%的桥梁
                    top_indices = np.argsort(combined_score)[-target_count:]
                    
                    budget_per_bridge = total_budget_t / target_count
                    new_raw_budgets[t, top_indices] = budget_per_bridge
        
        elif self.budget_allocation_strategy == 'preventive_maintenance':
            """策略：预防性维护 - 优先维护健康状态中等的桥梁，防止恶化"""
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                health_t = health_states[t, :actual_n_agents]
                
                # 定义预防性维护的健康状态范围（0.3-0.7）
                preventive_mask = (health_t >= 0.3) & (health_t <= 0.7)
                preventive_indices = np.where(preventive_mask)[0]
                
                if len(preventive_indices) > 0:
                    # 在预防性维护范围内，优先健康状态较差的
                    preventive_health = health_t[preventive_indices]
                    preventive_order = np.argsort(preventive_health)  # 从差到好
                    
                    # 选择其中一部分进行维护
                    target_count = min(len(preventive_indices), max(1, int(actual_n_agents * 0.2)))
                    selected_preventive = preventive_indices[preventive_order[:target_count]]
                    
                    budget_per_bridge = total_budget_t / target_count
                    new_raw_budgets[t, selected_preventive] = budget_per_bridge
                else:
                    # 如果没有处于预防性维护范围的桥梁，回退到critical_first
                    health_order = np.argsort(health_t)
                    target_bridges = max(1, int(actual_n_agents * 0.2))
                    critical_indices = health_order[:target_bridges]
                    budget_per_bridge = total_budget_t / target_bridges
                    new_raw_budgets[t, critical_indices] = budget_per_bridge
        
        elif self.budget_allocation_strategy == 'rotating_focus':
            """策略：轮换重点 - 每年重点关注不同的桥梁组，避免长期忽视"""
            # 将桥梁按重要性分组
            if importance_data is not None:
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)
                
                # 使用第一个时间步的重要性进行分组
                base_importance = importance_data[0, :actual_n_agents]
                importance_groups = np.array_split(np.argsort(base_importance), 3)  # 分成3组
            else:
                # 如果没有重要性数据，随机分组
                bridge_indices = np.arange(actual_n_agents)
                np.random.shuffle(bridge_indices)
                importance_groups = np.array_split(bridge_indices, 3)
            
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                health_t = health_states[t, :actual_n_agents]
                
                # 根据年份确定重点组（循环）
                focus_group_idx = t % 3
                focus_group = importance_groups[focus_group_idx]
                
                # 在重点组内选择最需要维修的桥梁
                if len(focus_group) > 0:
                    focus_health = health_t[focus_group]
                    focus_order = np.argsort(focus_health)  # 从差到好
                    
                    # 选择重点组内最差的一半桥梁
                    target_count = max(1, len(focus_group) // 2)
                    selected_bridges = focus_group[focus_order[:target_count]]
                    
                    budget_per_bridge = total_budget_t / target_count
                    new_raw_budgets[t, selected_bridges] = budget_per_bridge
        
        else:
            raise ValueError(f"未知的预算分配策略: {self.budget_allocation_strategy}")
        
        # 重新log变换和归一化
        new_log_budgets = np.log1p(new_raw_budgets)
        final_log_budgets = (new_log_budgets - mean) / (std + 1e-8)
        
        # 更新测试数据
        if self.test_data['log_budgets'].ndim == 3:
            self.test_data['log_budgets'] = final_log_budgets[..., np.newaxis]
        else:
            self.test_data['log_budgets'] = final_log_budgets
        
        # 打印统计信息
        print(f"预算分配策略应用完成:")
        print(f"  策略: {self.budget_allocation_strategy}")
        print(f"  原始预算范围: {np.min(raw_budgets[:, :actual_n_agents]):.2f} - {np.max(raw_budgets[:, :actual_n_agents]):.2f}")
        print(f"  重分配后预算范围: {np.min(new_raw_budgets[:, :actual_n_agents]):.2f} - {np.max(new_raw_budgets[:, :actual_n_agents]):.2f}")
        
        # 验证总预算守恒
        original_total = np.sum(raw_budgets[:, :actual_n_agents], axis=1)
        new_total = np.sum(new_raw_budgets[:, :actual_n_agents], axis=1)
        max_diff = np.max(np.abs(original_total - new_total))
        print(f"  预算守恒检查: 最大差异={max_diff:.6f}")
        
        # ✅ 分析预算分配的稀疏性（更符合你的系统特点）
        total_positions = T * actual_n_agents
        zero_budget_positions = np.sum(new_raw_budgets[:, :actual_n_agents] == 0)
        nonzero_budget_positions = total_positions - zero_budget_positions
        sparsity_ratio = zero_budget_positions / total_positions
        
        print(f"  预算稀疏性分析:")
        print(f"    零预算位置: {zero_budget_positions}/{total_positions} ({sparsity_ratio:.1%})")
        print(f"    非零预算位置: {nonzero_budget_positions}/{total_positions} ({1-sparsity_ratio:.1%})")
        
        if nonzero_budget_positions > 0:
            nonzero_budgets = new_raw_budgets[:, :actual_n_agents][new_raw_budgets[:, :actual_n_agents] > 0]
            print(f"    非零预算统计: 均值={np.mean(nonzero_budgets):.2f}, 中位数={np.median(nonzero_budgets):.2f}")
            print(f"    每年平均维修桥梁数: {nonzero_budget_positions/T:.1f}/{actual_n_agents}")

    def __apply_budget_allocation_strategy(self):
        """
        应用预算分配策略
        """
        if self.budget_allocation_strategy == 'original':
            return  # 不做任何更改
        
        print(f"应用预算分配策略: {self.budget_allocation_strategy}")
        
        # 获取原始数据
        original_log_budgets = self.test_data['log_budgets'].copy()  # [T, n_agents] or [T, n_agents, 1]
        original_episode_budget = self.test_data['budgets']
        importance_data = self.test_data.get('importance', None)  # [T, n_agents] or None
        actual_n_agents = self.test_data['actual_n_agents']
        
        # ✅ 获取并保持原始归一化参数不变（这是关键！）
        if hasattr(self, 'env_info') and 'normalization_params' in self.env_info and 'log_budgets' in self.env_info['normalization_params']:
            log_budget_norm_params = self.env_info['normalization_params']['log_budgets']
            mean = log_budget_norm_params['mean']  # 保持训练时的归一化参数
            std = log_budget_norm_params['std']    # 保持训练时的归一化参数
        else:
            print("警告: 未找到log_budgets归一化参数，使用默认值")
            mean = 0.0
            std = 1.0
        
        # 处理数据维度
        if original_log_budgets.ndim == 3 and original_log_budgets.shape[-1] == 1:
            original_log_budgets = original_log_budgets.squeeze(-1)  # [T, n_agents]
        
        T, n_agents = original_log_budgets.shape
        
        # 步骤1: 反归一化得到原始log预算值
        raw_log_budgets = original_log_budgets * std + mean
        
        # 步骤2: 转换为原始预算值（非log）
        raw_budgets = np.expm1(raw_log_budgets)
        
        # 步骤3: 计算每个时间步的总预算
        total_budgets_per_timestep = np.sum(raw_budgets[:, :actual_n_agents], axis=1)  # [T]
        
        # 步骤4: 根据策略重新分配预算
        if self.budget_allocation_strategy == 'uniform':
            # 策略2：均分
            new_raw_budgets = np.zeros_like(raw_budgets)
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                uniform_budget = total_budget_t / actual_n_agents
                new_raw_budgets[t, :actual_n_agents] = uniform_budget
                
        elif self.budget_allocation_strategy == 'importance':
            # 策略3：根据重要性分配
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到均分策略")
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    uniform_budget = total_budget_t / actual_n_agents
                    new_raw_budgets[t, :actual_n_agents] = uniform_budget
            else:
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)
                
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    importance_t = importance_data[t, :actual_n_agents]
                    
                    importance_sum = np.sum(importance_t)
                    if importance_sum > 0:
                        importance_weights = importance_t / importance_sum
                    else:
                        importance_weights = np.ones(actual_n_agents) / actual_n_agents
                    
                    new_raw_budgets[t, :actual_n_agents] = total_budget_t * importance_weights
        
        elif self.budget_allocation_strategy == 'importance_top10':
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到均分策略")
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    uniform_budget = total_budget_t / actual_n_agents
                    new_raw_budgets[t, :actual_n_agents] = uniform_budget
            else:
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)
                
                new_raw_budgets = np.zeros_like(raw_budgets)
                top_k = max(1, int(np.ceil(actual_n_agents * 0.1)))
                
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    importance_t = importance_data[t, :actual_n_agents]
                    top_indices = np.argsort(importance_t)[-top_k:]
                    budget_per_top_bridge = total_budget_t / top_k
                    new_raw_budgets[t, top_indices] = budget_per_top_bridge
        else:
            raise ValueError(f"未知的预算分配策略: {self.budget_allocation_strategy}")
        
        # ✅ 步骤5: 重新log变换
        new_log_budgets = np.log1p(new_raw_budgets)
        
        # ✅ 步骤6: 使用原始归一化参数进行归一化（保持模型能理解的数值范围）
        final_log_budgets = (new_log_budgets - mean) / (std + 1e-8)
        
        # 更新测试数据
        if self.test_data['log_budgets'].ndim == 3:
            self.test_data['log_budgets'] = final_log_budgets[..., np.newaxis]
        else:
            self.test_data['log_budgets'] = final_log_budgets
        
        # ✅ 不要修改env_info中的归一化参数，保持模型兼容性
        
        # 打印统计信息
        print(f"预算分配策略应用完成:")
        print(f"  策略: {self.budget_allocation_strategy}")
        print(f"  使用训练时归一化参数: mean={mean:.4f}, std={std:.4f}")
        print(f"  原始预算范围: {np.min(raw_budgets[:, :actual_n_agents]):.2f} - {np.max(raw_budgets[:, :actual_n_agents]):.2f}")
        print(f"  重分配后预算范围: {np.min(new_raw_budgets[:, :actual_n_agents]):.2f} - {np.max(new_raw_budgets[:, :actual_n_agents]):.2f}")
        print(f"  原始log预算范围: {np.min(raw_log_budgets[:, :actual_n_agents]):.4f} - {np.max(raw_log_budgets[:, :actual_n_agents]):.4f}")
        print(f"  重分配后log预算范围: {np.min(new_log_budgets[:, :actual_n_agents]):.4f} - {np.max(new_log_budgets[:, :actual_n_agents]):.4f}")
        print(f"  最终归一化log预算范围: {np.min(final_log_budgets[:, :actual_n_agents]):.4f} - {np.max(final_log_budgets[:, :actual_n_agents]):.4f}")
        
        # 验证总预算守恒
        original_total = np.sum(raw_budgets[:, :actual_n_agents], axis=1)
        new_total = np.sum(new_raw_budgets[:, :actual_n_agents], axis=1)
        max_diff = np.max(np.abs(original_total - new_total))
        print(f"  预算守恒检查: 最大差异={max_diff:.6f}")
        
        # ✅ 验证模型输入的合理性：检查是否超出训练时见过的范围
        train_log_budget_range = f"约 [{mean-3*std:.2f}, {mean+3*std:.2f}]"  # 训练时大约99.7%的数据范围
        current_range = [np.min(final_log_budgets[:, :actual_n_agents]), np.max(final_log_budgets[:, :actual_n_agents])]
        print(f"  训练时log预算范围(±3σ): {train_log_budget_range}")
        print(f"  当前log预算范围: [{current_range[0]:.2f}, {current_range[1]:.2f}]")
        
        if current_range[0] < mean - 3*std or current_range[1] > mean + 3*std:
            print(f"  ⚠️  警告: 当前预算值超出训练时范围，模型可能无法准确处理")
        else:
            print(f"  ✅ 当前预算值在训练时范围内，模型应能正确理解")
        
        # ✅ 详细的预算分配分析
        if self.budget_allocation_strategy == 'importance_top10' and None:
            print(f"  importance_top10策略详细分析:")
            print(f"    前10%桥梁数量: {top_k}/{actual_n_agents}")
            
            # 分析预算变化
            for t in range(min(3, T)):
                importance_t = importance_data[t, :actual_n_agents] if importance_data is not None else None
                if importance_t is not None:
                    top_indices = np.argsort(importance_t)[-top_k:]
                    bottom_indices = np.argsort(importance_t)[:-top_k] if top_k < actual_n_agents else []
                    
                    orig_top_budgets = raw_budgets[t, top_indices]
                    new_top_budgets = new_raw_budgets[t, top_indices]
                    
                    print(f"    时间步{t}:")
                    print(f"      前10%桥梁原预算: {orig_top_budgets}")
                    print(f"      前10%桥梁新预算: {new_top_budgets}")
                    print(f"      前10%桥梁重要性: {importance_t[top_indices]}")
                    if len(bottom_indices) > 0:
                        orig_bottom_sum = np.sum(raw_budgets[t, bottom_indices])
                        print(f"      其他桥梁原预算总和: {orig_bottom_sum:.2f} -> 0.00")

    def _apply_budget_allocation_strategy(self):
        """
        应用预算分配策略
        """
        if self.budget_allocation_strategy == 'original':
            return  # 不做任何更改
        
        print(f"应用预算分配策略: {self.budget_allocation_strategy}")
        
        # 获取原始数据
        original_log_budgets = self.test_data['log_budgets'].copy()  # [T, n_agents] or [T, n_agents, 1]
        original_episode_budget = self.test_data['budgets']
        importance_data = self.test_data.get('importance', None)  # [T, n_agents] or None
        actual_n_agents = self.test_data['actual_n_agents']
        
        # 获取归一化参数
        if hasattr(self, 'env_info') and 'normalization_params' in self.env_info and 'log_budgets' in self.env_info['normalization_params']:
            log_budget_norm_params = self.env_info['normalization_params']['log_budgets']
            mean = log_budget_norm_params['mean']
            std = log_budget_norm_params['std']
        else:
            print("警告: 未找到log_budgets归一化参数，使用默认值")
            mean = 0.0
            std = 1.0
        
        # 处理数据维度
        if original_log_budgets.ndim == 3 and original_log_budgets.shape[-1] == 1:
            original_log_budgets = original_log_budgets.squeeze(-1)  # [T, n_agents]
        
        T, n_agents = original_log_budgets.shape
        
        # 步骤1: 反归一化得到原始log预算值
        raw_log_budgets = original_log_budgets * std + mean
        
        # 步骤2: 转换为原始预算值（非log）
        raw_budgets = np.expm1(raw_log_budgets)
        
        # 步骤3: 计算每个时间步的总预算
        total_budgets_per_timestep = np.sum(raw_budgets[:, :actual_n_agents], axis=1)  # [T]
        
        # 步骤4: 根据策略重新分配预算
        if self.budget_allocation_strategy == 'uniform':
            # 策略2：均分
            new_raw_budgets = np.zeros_like(raw_budgets)
            for t in range(T):
                total_budget_t = total_budgets_per_timestep[t]
                uniform_budget = total_budget_t / actual_n_agents
                new_raw_budgets[t, :actual_n_agents] = uniform_budget
                
        elif self.budget_allocation_strategy == 'importance':
            # 策略3：根据重要性分配
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到均分策略")
                # 回退到均分
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    uniform_budget = total_budget_t / actual_n_agents
                    new_raw_budgets[t, :actual_n_agents] = uniform_budget
            else:
                # 处理重要性数据维度
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)  # [T, n_agents]
                
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    # 获取活跃桥梁的重要性
                    importance_t = importance_data[t, :actual_n_agents]  # [actual_n_agents]
                    
                    # 计算重要性权重（归一化）
                    importance_sum = np.sum(importance_t)
                    if importance_sum > 0:
                        importance_weights = importance_t / importance_sum
                    else:
                        # 如果重要性全为0，回退到均分
                        importance_weights = np.ones(actual_n_agents) / actual_n_agents
                    
                    # 根据重要性权重分配预算
                    new_raw_budgets[t, :actual_n_agents] = total_budget_t * importance_weights
        
        elif self.budget_allocation_strategy == 'importance_top10':
            # ✅ 策略4：重要性前10%桥梁均分预算
            if importance_data is None:
                print("警告: 未找到重要性数据，回退到均分策略")
                # 回退到均分
                new_raw_budgets = np.zeros_like(raw_budgets)
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    uniform_budget = total_budget_t / actual_n_agents
                    new_raw_budgets[t, :actual_n_agents] = uniform_budget
            else:
                # 处理重要性数据维度
                if importance_data.ndim == 3 and importance_data.shape[-1] == 1:
                    importance_data = importance_data.squeeze(-1)  # [T, n_agents]
                
                new_raw_budgets = np.zeros_like(raw_budgets)
                
                # 计算前10%的桥梁数量
                top_k = max(1, int(np.ceil(actual_n_agents * 0.1)))  # 至少1个桥梁
                
                for t in range(T):
                    total_budget_t = total_budgets_per_timestep[t]
                    # 获取活跃桥梁的重要性
                    importance_t = importance_data[t, :actual_n_agents]  # [actual_n_agents]
                    
                    # 根据重要性排序，获取前10%桥梁的索引
                    top_indices = np.argsort(importance_t)[-top_k:]  # 重要性最高的top_k个桥梁
                    
                    # 所有预算均分给前10%的桥梁
                    budget_per_top_bridge = total_budget_t / top_k
                    new_raw_budgets[t, top_indices] = budget_per_top_bridge
                    
                    # 其他桥梁预算为0（已经在初始化时设为0）

        else:
            raise ValueError(f"未知的预算分配策略: {self.budget_allocation_strategy}")
        
        # 步骤5: 重新log变换
        new_log_budgets = np.log1p(new_raw_budgets)
        
        # 步骤6: 重新归一化
        final_log_budgets = (new_log_budgets - mean) / (std + 1e-8)
        
        # 更新测试数据
        if self.test_data['log_budgets'].ndim == 3:
            self.test_data['log_budgets'] = final_log_budgets[..., np.newaxis]  # 恢复 [T, n_agents, 1] 格式
        else:
            self.test_data['log_budgets'] = final_log_budgets
        
        # 打印统计信息
        print(f"预算分配策略应用完成:")
        print(f"  策略: {self.budget_allocation_strategy}")
        print(f"  原始预算范围: {np.min(raw_budgets[:, :actual_n_agents]):.2f} - {np.max(raw_budgets[:, :actual_n_agents]):.2f}")
        print(f"  重分配后预算范围: {np.min(new_raw_budgets[:, :actual_n_agents]):.2f} - {np.max(new_raw_budgets[:, :actual_n_agents]):.2f}")
        print(f"  原始log预算范围: {np.min(original_log_budgets[:, :actual_n_agents]):.4f} - {np.max(original_log_budgets[:, :actual_n_agents]):.4f}")
        print(f"  重分配后log预算范围: {np.min(new_log_budgets[:, :actual_n_agents]):.4f} - {np.max(new_log_budgets[:, :actual_n_agents]):.4f}")    


        if self.budget_allocation_strategy == 'importance' and importance_data is not None:
            print(f"  重要性范围: {np.min(importance_data[:, :actual_n_agents]):.3f} - {np.max(importance_data[:, :actual_n_agents]):.3f}")
            
            # 打印前几个时间步的详细分配信息
            for t in range(min(3, T)):
                print(f"  时间步{t}: 总预算={total_budgets_per_timestep[t]:.2f}")
                for a in range(min(3, actual_n_agents)):
                    orig_budget = raw_budgets[t, a]
                    new_budget = new_raw_budgets[t, a]
                    importance_val = importance_data[t, a]
                    print(f"    桥梁{a}: 重要性={importance_val:.3f}, 原预算={orig_budget:.2f}, 新预算={new_budget:.2f}")
        
        # 验证总预算守恒
        original_total = np.sum(raw_budgets[:, :actual_n_agents], axis=1)
        new_total = np.sum(new_raw_budgets[:, :actual_n_agents], axis=1)
        max_diff = np.max(np.abs(original_total - new_total))
        print(f"  预算守恒检查: 最大差异={max_diff:.6f}")

        # exit(0)

    def load_env_info(self, env_info_path):
        """加载环境信息"""
        if env_info_path is None:
            # 默认路径
            env_info_path = "marl/data_benchmark/episodes/train_env_info.json"
        
        try:
            with open(env_info_path, 'r') as f:
                env_info = json.load(f)
            print(f"成功加载环境信息: {env_info_path}")
            return env_info
        except Exception as e:
            print(f"加载环境信息失败: {e}")
            return {}
    
    def normalize_health(self, health_values):
        """
        将原始健康值归一化
        
        Args:
            health_values: 原始健康值（0-9）
        
        Returns:
            normalized_health: 归一化后的健康值
        """
        if 'policy_obs' not in self.norm_params:
            print("警告: 未找到policy_obs归一化参数，使用原始值")
            return health_values
        
        # policy_obs的第一个特征（索引0）是健康值
        mean = self.norm_params['policy_obs']['mean'][0]
        std = self.norm_params['policy_obs']['std'][0]
        
        # 归一化
        normalized = (health_values - mean) / (std + 1e-8)
        return normalized
    
    def denormalize_health(self, normalized_health):
        """
        将归一化健康值反归一化
        
        Args:
            normalized_health: 归一化后的健康值
        
        Returns:
            health_values: 原始健康值（0-9）
        """
        if 'policy_obs' not in self.norm_params:
            return normalized_health
        
        mean = self.norm_params['policy_obs']['mean'][0]
        std = self.norm_params['policy_obs']['std'][0]
        
        # 反归一化
        health_values = normalized_health * (std + 1e-8) + mean
        return health_values
    
    def get_health_category(self, health_value):
        """
        根据健康值获取健康类别
        
        Args:
            health_value: 健康值（0-9）
        
        Returns:
            category: 健康类别
        """
        for category, (min_val, max_val) in self.health_categories.items():
            if min_val <= health_value <= max_val:
                return category
        return 'unknown'
    
    def detect_model_type(self):
        """检测模型类型"""
        model_class_name = self.model.__class__.__name__
        

        #print(f"模型类型: {model_class_name}")
        #exit(0)

        if 'CDT' in model_class_name:
            return 'cdt'
        elif 'cpq' in model_class_name.lower():
            return 'osrl'
        elif any(name in model_class_name.lower() for name in ['marl', 'qmix', 'iql', 'discrete']):
            return 'marl'
        elif any(name in model_class_name.lower() for name in [ 'bc', 'osrl']):
            return 'osrl'
        elif 'random' in model_class_name.lower():
            if 'marl' in model_class_name.lower():
                return 'random_marl'
            else:
                return 'random_osrl'
        else:
            return 'unknown'
    
    def check_budget_in_obs(self):
        """检查模型是否需要预算在观察中"""

        if 'without_budget' in self.algorithm_name.lower():
            return False

        # 检查模型是否有相关属性标识
        if hasattr(self.model, 'has_budget_in_obs'):
            return self.model.has_budget_in_obs
        elif hasattr(self.model, 'needs_budget_input'):
            return not self.model.needs_budget_input  # 如果不需要单独输入预算，说明预算在观察中
        
        # 根据模型类型推断
        if self.model_type == 'marl':
            return True  # 多智能体算法通常将预算拼接到观察中
        elif self.model_type in ['osrl', 'random_osrl']:
            return False  # OSRL算法通常单独传入预算
        elif self.model_type == 'cdt':
            return False  # CDT算法有特殊接口
        else:
            return False  # 默认不需要
    
    def augment_data_with_budgets(self, data, log_budgets, budget_mode='provided'):
        """
        将预算信息拼接到观察数据中（从Multi_Task_Run_v4.py复制）
        
        Args:
            data: [num_eps, T, n_agents, obs_dim] 原始观察数据
            log_budgets: 预算信息，格式根据budget_mode而定
            budget_mode: 'provided' 或 'uniform'
        
        Returns:
            augmented_data: [num_eps, T, n_agents, obs_dim+1] 增强后的观察数据
        """
        data = np.asarray(data)
        num_eps, T, n_agents, obs_dim = data.shape
        
        if budget_mode == 'provided':
            # 使用提供的budgets（已经是log格式）
            if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
                budget_arr = log_budgets  # [num_eps, T, n_agents, 1]
            elif log_budgets.ndim == 3:
                budget_arr = log_budgets[..., np.newaxis]  # 添加最后一维
            else:
                raise ValueError(f"不支持的budgets形状: {log_budgets.shape}")
        else:
            raise ValueError("budget_mode 必须是 'provided'")
        
        # 检查形状兼容性
        if budget_arr.shape[:3] != data.shape[:3]:
            raise ValueError(f"预算形状 {budget_arr.shape[:3]} 与数据形状 {data.shape[:3]} 不匹配")
        
        # 拼接
        augmented_data = np.concatenate([data, budget_arr], axis=-1)
        
        return augmented_data

    def create_observation(self, health_state, budget, bridge_age=50, adt=1000, structure_length=100, importance=0.5):
        """
        创建观察状态（根据模型类型决定是否包含预算）
        
        Args:
            health_state: 当前健康状态（原始值0-9）
            budget: 当前预算
            bridge_age: 桥梁年龄
            adt: 平均日交通量
            structure_length: 结构长度
            importance: 重要性
        
        Returns:
            obs: 观察状态
        """
        # 归一化健康值
        normalized_health = self.normalize_health(health_state)
        
        # 创建基础观察向量 [健康值, 桥梁年龄, ADT, 结构长度, 重要性]
        base_obs = np.array([
            normalized_health,
            bridge_age,
            adt,
            structure_length,
            importance
        ])
        
        # 应用归一化（除了健康值已经归一化）
        if 'policy_obs' in self.norm_params:
            mean = np.array(self.norm_params['policy_obs']['mean'])
            std = np.array(self.norm_params['policy_obs']['std'])
            
            # 健康值已经归一化，其他特征需要归一化
            base_obs[1:] = (base_obs[1:] - mean[1:]) / (std[1:] + 1e-8)
        
        # ✅ 根据模型需求决定是否添加预算
        if self.needs_budget_in_obs:
            # 对于需要预算在观察中的模型（如MARL），添加预算维度
            budget_normalized = budget  # 假设budget已经是log格式
            obs = np.concatenate([base_obs, [budget_normalized]])
        else:
            # 对于不需要预算在观察中的模型（如OSRL），只返回基础观察
            obs = base_obs
        
        return obs
    
    def get_model_action(self, health_state, budget_log, bridge_age=50, adt=1000, structure_length=100, importance=0.5, bridge_idx=0):
        """
        使用模型获取维修动作
        
        Args:
            health_state: 当前健康状态（原始值0-9）
            budget_log: 当前预算（log格式，直接从数据集获取）
            bridge_age: 桥梁年龄
            adt: 平均日交通量
            structure_length: 结构长度
            importance: 重要性
            bridge_idx: 当前桥梁索引
        
        Returns:
            action: 选择的动作
        """
        try:
            # ✅ 根据模型类型创建正确的观察状态
            obs = self.create_observation(health_state, budget_log, bridge_age, adt, structure_length, importance)
            
            # 获取最大桥梁数量
            max_n_agents = self.env_info.get('max_bridges', 500)
            
            # 检测模型设备
            if hasattr(self.model, 'device'):
                model_device = self.model.device
            elif hasattr(self.model, 'parameters'):
                model_device = next(self.model.parameters()).device
            else:
                model_device = th.device('cpu')
            
            # ✅ 根据模型类型选择合适的接口
            if self.model_type == 'marl' or self.model_type == 'random_marl':
                # MARL算法接口
                obs_in = np.zeros((1, max_n_agents, obs.shape[0]), dtype=np.float32)
                obs_in[0, bridge_idx, :] = obs
                
                mask_in = np.zeros((1, max_n_agents), dtype=np.float32)
                mask_in[0, bridge_idx] = 1.0
                
                obs_in = th.tensor(obs_in, dtype=th.float32, device=model_device)
                
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                with th.no_grad():
                    pred = self.model.act(obs_in, mask_in, legal_actions=None)
                
                action = int(pred[0, bridge_idx])
                return action
                
            elif self.model_type == 'osrl' or self.model_type == 'random_osrl':
                # OSRL算法接口：传入log格式的预算
                obs_in = obs.reshape(1, -1)
                
                obs_tensor = th.tensor(obs_in, dtype=th.float32, device=model_device)
                budget_tensor = th.tensor([budget_log], dtype=th.float32, device=model_device)  # 使用log格式预算
                
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                with th.no_grad():
                    pred = self.model.act(obs_tensor, budget_tensor)
            
                # 处理输出
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                
                if hasattr(pred, 'cpu'):
                    pred = pred.cpu().numpy()
                elif hasattr(pred, 'numpy'):
                    pred = pred.numpy()
                elif not isinstance(pred, np.ndarray):
                    pred = np.asarray(pred)
                
                if isinstance(pred, (int, np.integer)):
                    action = int(pred)
                elif hasattr(pred, 'ndim'):
                    if pred.ndim >= 2:
                        action = int(pred[0, 0])
                    elif pred.ndim == 1:
                        action = int(pred[0])
                    else:
                        action = int(pred)
                else:
                    action = int(pred)
                
                return action
                
            # [CDT和其他算法的处理逻辑保持不变]
            # ...
                
        except Exception as e:
            print(f"模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def apply_health_transition(self, current_health, action):
        """
        应用健康状态转移
        
        Args:
            current_health: 当前健康状态（原始值0-9）
            action: 执行的动作
        
        Returns:
            new_health: 新的健康状态（原始值0-9）
        """
        if action not in self.transition_matrices:
            action = 0  # 默认无动作
        
        mat = self.transition_matrices[action]
        
        # 投影健康状态到转移矩阵的类别范围（0-3）
        if current_health <= 2:  # 0-2: critical
            state_idx = 0
        elif current_health <= 4:  # 3-4: poor
            state_idx = 1
        elif current_health <= 6:  # 5-6: fair
            state_idx = 2
        else:  # 7-9: good
            state_idx = 3
        
        probs = mat[state_idx]
        
        # 根据概率分布选择新的健康类别（0-3）
        new_health_category = np.random.choice(len(probs), p=probs)
        
        # 将健康类别映射回0-9的健康值范围
        if new_health_category == 0:  # critical
            new_health = np.random.randint(0, 3)  # 0-2
        elif new_health_category == 1:  # poor
            new_health = np.random.randint(3, 5)  # 3-4
        elif new_health_category == 2:  # fair
            new_health = np.random.randint(5, 7)  # 5-6
        else:  # good (new_health_category == 3)
            new_health = np.random.randint(7, 10)  # 7-9
        
        return new_health
    
    def run_simulation(self, n_years=100, base_budget_per_bridge=10000):
        """
        运行100年桥梁维修仿真
        """
        print(f"开始{n_years}年桥梁维修仿真...")
        
        # 获取测试数据
        test_data = self.test_data['data']  # [T, n_agents, state_dim]
        test_actions = self.test_data['actions']  # [T, n_agents]
        test_health = self.test_data['health']  # [T+1, n_agents]
        test_log_budgets = self.test_data['log_budgets']  # [T, n_agents, 1]
        test_episode_budget = self.test_data['budgets']  # episode总预算
        actual_n_agents = self.test_data['actual_n_agents']
        
        T, n_agents, state_dim = test_data.shape
        
        # ✅ 处理预算数据格式
        if test_log_budgets.ndim == 3 and test_log_budgets.shape[-1] == 1:
            test_log_budgets = test_log_budgets.squeeze(-1)  # [T, n_agents]
        
        print(f"测试数据维度: {test_data.shape}")
        print(f"预算数据维度: {test_log_budgets.shape}")
        print(f"Episode总预算: {test_episode_budget}")
        
        # 初始化桥梁健康状态（使用指定的初始健康等级）
        bridge_health = np.full(n_agents, self.initial_health_level)
        
        # 存储每座桥梁的健康历史
        health_histories = np.zeros((n_years + 1, n_agents))
        normalized_health_histories = np.zeros((n_years + 1, n_agents))
        health_histories[0] = bridge_health.copy()
        normalized_health_histories[0] = self.normalize_health(bridge_health)
        
        for year in range(1, n_years + 1):
            print(f"仿真第{year}年...")
            
            # ✅ 确定当前年份使用的数据索引
            if year <= T:
                data_idx = year - 1  # 使用对应年份的数据
            else:
                data_idx = T - 1     # 超出范围时使用最后一年的数据
            
            # 记录年度数据
            year_data = {
                'year': year,
                'episode_budget': float(test_episode_budget),  # 使用数据集中的episode预算
                'actions': [],
                'costs': [],
                'budgets_used': [],  # 记录实际使用的预算
                'health_before': bridge_health.copy()
            }
            
            total_cost = 0
            
            # 对每座桥梁进行维修决策
            for bridge_idx in range(n_agents):
                # ✅ 使用当前年份的健康状态
                current_health = bridge_health[bridge_idx]
                
                # ✅ 直接从测试数据中获取预算（log格式）
                budget_log = float(test_log_budgets[data_idx, bridge_idx])
                
                # 获取其他特征（从测试数据中）
                test_obs = test_data[data_idx, bridge_idx]  # [state_dim]
                bridge_age = float(test_obs[1]) if len(test_obs) > 1 else 50.0
                adt = float(test_obs[2]) if len(test_obs) > 2 else 1000.0
                structure_length = float(test_obs[3]) if len(test_obs) > 3 else 100.0
                importance = float(test_obs[4]) if len(test_obs) > 4 else 0.5
                
                # 对于超出测试数据范围的年份，适当调整桥梁年龄
                if year > T:
                    bridge_age += (year - T)
                
                # ✅ 关键修复：使用当前更新后的健康状态和正确的预算调用模型
                action = self.get_model_action(
                    current_health,      # 使用当前年份的健康状态
                    budget_log,          # 使用数据集中的log格式预算
                    bridge_age, 
                    adt, 
                    structure_length, 
                    importance, 
                    bridge_idx
                )
                
                # 应用健康状态转移
                new_health = self.apply_health_transition(current_health, action)
                bridge_health[bridge_idx] = new_health  # ✅ 更新健康状态
                
                # 计算成本
                cost = self.action_costs.get(action, 0)
                total_cost += cost
                
                # 记录数据
                year_data['actions'].append(action)
                year_data['costs'].append(cost)
                year_data['budgets_used'].append(budget_log)
                
                # ✅ 调试信息：打印健康状态变化和预算使用
                if bridge_idx < 3 and year <= 3:  # 只打印前3个桥梁前3年的信息
                    print(f"  桥梁{bridge_idx}: 健康{current_health:.1f}->{new_health:.1f}, "
                        f"动作{action}, 成本{cost:.1f}, 预算{budget_log:.4f}")
            
            # 记录健康历史
            health_histories[year] = bridge_health.copy()
            normalized_health_histories[year] = self.normalize_health(bridge_health)
            
            # ✅ 调试信息：检查健康状态和成本变化
            avg_health_before = np.mean(year_data['health_before'])
            avg_health_after = np.mean(bridge_health)
            avg_budget_used = np.mean(year_data['budgets_used'])
            
            print(f"  第{year}年: 总成本={total_cost:.0f}, 平均健康变化={avg_health_before:.2f}->{avg_health_after:.2f}, "
                f"平均预算={avg_budget_used:.4f}")
            
            # 计算健康状态分布
            health_distribution = self.calculate_health_distribution(bridge_health)
            health_category_distribution = self.calculate_health_category_distribution(bridge_health)
            
            # 计算动作分布
            action_distribution = self.calculate_action_distribution(year_data['actions'])
            
            # ✅ 调试信息：打印动作分布和预算统计
            if year <= 3:
                print(f"    动作分布: {action_distribution}")
                print(f"    预算范围: {np.min(year_data['budgets_used']):.4f} - {np.max(year_data['budgets_used']):.4f}")
            
            # 存储年度结果
            self.simulation_results['years'].append(year)
            self.simulation_results['health_distributions'].append(health_distribution)
            self.simulation_results['health_category_distributions'].append(health_category_distribution)
            self.simulation_results['total_costs'].append(total_cost)
            self.simulation_results['action_distributions'].append(action_distribution)
            self.simulation_results['health_histories'].append(health_histories[year].copy())
            self.simulation_results['normalized_health_histories'].append(normalized_health_histories[year].copy())
            
            # ✅ 检查成本变化模式
            if year > 1:
                prev_cost = self.simulation_results['total_costs'][-2]
                cost_change = total_cost - prev_cost
                print(f"    成本变化: {prev_cost:.0f} -> {total_cost:.0f} (Δ{cost_change:+.0f})")
        
        print("仿真完成!")
        return self.simulation_results
    
    def calculate_health_distribution(self, health_states):
        """计算健康状态分布"""
        distribution = {}
        for health in range(10):  # 健康状态0-9
            count = np.sum(health_states == health)
            proportion = count / len(health_states)
            distribution[health] = proportion
        return distribution
    
    def calculate_health_category_distribution(self, health_states):
        """计算健康类别分布"""
        distribution = {}
        for category in self.health_categories.keys():
            min_val, max_val = self.health_categories[category]
            count = np.sum((health_states >= min_val) & (health_states <= max_val))
            proportion = count / len(health_states)
            distribution[category] = proportion
        return distribution
    
    def calculate_action_distribution(self, actions):
        """计算动作分布"""
        distribution = {}
        for action in range(4):  # 动作0-3
            count = actions.count(action)
            proportion = count / len(actions)
            distribution[action] = proportion
        return distribution
    
    def plot_health_distribution_stacked(self, save_path=None):
        """
        绘制100年健康状态分布堆积图（0-9）
        
        Args:
            save_path: 保存路径
        """
        years = self.simulation_results['years']
        health_distributions = self.simulation_results['health_distributions']
        
        # 准备数据
        health_levels = list(range(10))  # 0-9
        health_labels = [str(i) for i in range(10)]
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        # 创建堆积图数据
        stacked_data = np.zeros((len(years), len(health_levels)))
        for i, year_dist in enumerate(health_distributions):
            for j, health in enumerate(health_levels):
                stacked_data[i, j] = year_dist.get(health, 0)
        
        # 绘制堆积图
        plt.figure(figsize=(15, 8))
        
        # 创建堆积条形图 - 去掉间隔
        bottom = np.zeros(len(years))
        for i, (health, label, color) in enumerate(zip(health_levels, health_labels, colors)):
            plt.bar(years, stacked_data[:, i], bottom=bottom, 
                   label=f'Health {label}', color=color, alpha=0.8, width=1.0)
            bottom += stacked_data[:, i]
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Proportion of Bridges', fontsize=12)
        plt.title('100-Year Bridge Health Distribution Evolution (0-9 Scale)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 确保y轴从0到1
        plt.ylim(0, 1)
        
        # 添加比例标签
        plt.text(0.02, 0.98, 'Proportions sum to 100%', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"健康分布堆积图已保存至: {save_path}")
        
        plt.show()
    
    def plot_health_category_stacked(self, save_path=None):
        """
        绘制100年健康类别分布堆积图（poor, fair, good, excellent）
        
        Args:
            save_path: 保存路径
        """
        years = self.simulation_results['years']
        health_category_distributions = self.simulation_results['health_category_distributions']
        
        # 准备数据
        categories = ['poor', 'fair', 'good', 'excellent']
        colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1']
        
        # 创建堆积图数据
        stacked_data = np.zeros((len(years), len(categories)))
        for i, year_dist in enumerate(health_category_distributions):
            for j, category in enumerate(categories):
                stacked_data[i, j] = year_dist.get(category, 0)
        
        # 绘制堆积图
        plt.figure(figsize=(15, 8))
        
        # 创建堆积条形图 - 去掉间隔
        bottom = np.zeros(len(years))
        for i, (category, color) in enumerate(zip(categories, colors)):
            plt.bar(years, stacked_data[:, i], bottom=bottom, 
                   label=f'{category.title()}', color=color, alpha=0.8, width=1.0)
            bottom += stacked_data[:, i]
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Proportion of Bridges', fontsize=12)
        plt.title('100-Year Bridge Health Category Distribution Evolution', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 确保y轴从0到1
        plt.ylim(0, 1)
        
        # 添加比例标签
        plt.text(0.02, 0.98, 'Proportions sum to 100%', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"健康类别分布堆积图已保存至: {save_path}")
        
        plt.show()
    
    def plot_cost_evolution(self, save_path=None):
        """绘制成本演化图"""
        years = self.simulation_results['years']
        costs = self.simulation_results['total_costs']
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, costs, linewidth=2, color='red', alpha=0.8)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Annual Cost', fontsize=12)
        plt.title('100-Year Bridge Maintenance Cost Evolution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成本演化图已保存至: {save_path}")
        
        plt.show()
    
    def plot_cumulative_cost_evolution(self, save_path=None):
        """绘制累计成本演化图"""
        years = self.simulation_results['years']
        costs = self.simulation_results['total_costs']
        
        # 计算累计成本
        cumulative_costs = np.cumsum(costs)
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, cumulative_costs, linewidth=2, color='darkred', alpha=0.8)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Cumulative Total Cost', fontsize=12)
        plt.title('100-Year Bridge Maintenance Cumulative Cost Evolution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加总成本标注
        total_cost = cumulative_costs[-1]
        plt.text(0.02, 0.98, f'Total Cost: {total_cost:,.0f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"累计成本演化图已保存至: {save_path}")
        
        plt.show()
    
    def plot_cost_comparison(self, save_path=None):
        """绘制年度成本和累计成本对比图"""
        years = self.simulation_results['years']
        costs = self.simulation_results['total_costs']
        cumulative_costs = np.cumsum(costs)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 年度成本图
        ax1.plot(years, costs, linewidth=2, color='red', alpha=0.8)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Annual Cost', fontsize=12)
        ax1.set_title('Annual Bridge Maintenance Cost', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 累计成本图
        ax2.plot(years, cumulative_costs, linewidth=2, color='darkred', alpha=0.8)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Cumulative Cost', fontsize=12)
        ax2.set_title('Cumulative Bridge Maintenance Cost', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加总成本标注
        total_cost = cumulative_costs[-1]
        ax2.text(0.02, 0.98, f'Total Cost: {total_cost:,.0f}', 
                transform=ax2.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成本对比图已保存至: {save_path}")
        
        plt.show()
    
    def plot_average_health_evolution(self, save_path=None):
        """绘制平均健康状态演化图"""
        years = self.simulation_results['years']
        health_histories = self.simulation_results['health_histories']
        
        # 计算每年平均健康状态
        avg_health = [np.mean(health) for health in health_histories]
        
        plt.figure(figsize=(12, 6))
        plt.plot(years, avg_health, linewidth=2, color='blue', alpha=0.8)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Health Level', fontsize=12)
        plt.title('100-Year Average Bridge Health Evolution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"平均健康演化图已保存至: {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir="marl/new_module/simulation_results"):
        """保存仿真结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_path = os.path.join(save_dir, f'bridge_simulation_{timestamp}.json')
        
        # 转换为可序列化的格式
        serializable_results = convert_to_serializable(self.simulation_results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存图表
        health_plot_path = os.path.join(save_dir, f'health_distribution_{timestamp}.png')
        health_category_plot_path = os.path.join(save_dir, f'health_category_distribution_{timestamp}.png')
        cost_plot_path = os.path.join(save_dir, f'cost_evolution_{timestamp}.png')
        cumulative_cost_plot_path = os.path.join(save_dir, f'cumulative_cost_evolution_{timestamp}.png')
        cost_comparison_plot_path = os.path.join(save_dir, f'cost_comparison_{timestamp}.png')
        avg_health_plot_path = os.path.join(save_dir, f'avg_health_evolution_{timestamp}.png')
        
        self.plot_health_distribution_stacked(health_plot_path)
        self.plot_health_category_stacked(health_category_plot_path)
        self.plot_cost_evolution(cost_plot_path)
        self.plot_cumulative_cost_evolution(cumulative_cost_plot_path)
        self.plot_cost_comparison(cost_comparison_plot_path)
        self.plot_average_health_evolution(avg_health_plot_path)
        
        print(f"仿真结果已保存至: {save_dir}")
        return results_path


def find_latest_model(model_dir="marl/new_module/saved_models", algorithm_name=None):
    """
    查找最新的模型文件
    
    Args:
        model_dir: 模型目录
        algorithm_name: 算法名称（如果为None则查找所有模型）
    
    Returns:
        model_path: 模型路径
    """
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return None
    
    if algorithm_name:
        pattern = os.path.join(model_dir, f"{algorithm_name}_model_*.pth")
    else:
        pattern = os.path.join(model_dir, "*_model_*.pth")
    
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"未找到模型文件: {pattern}")
        return None
    
    # 按修改时间排序，返回最新的
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"找到最新模型: {latest_model}")
    return latest_model


def main():
    """主函数"""
    print("桥梁维修100年仿真测试")
    print("=" * 50)
    
    algorithm_name="discrete_bc_20"
    # 查找最新的模型
    model_path = find_latest_model(algorithm_name=algorithm_name)  # 可以修改算法名称
    
    if model_path is None:
        print("未找到可用的模型文件!")
        return
    
    # 测试数据集路径
    test_data_path = "marl/data_benchmark/episodes/test_buffer.pt"
    
    # 创建仿真器
    simulator = BridgeMaintenanceSimulator(
        model_path=model_path,
        
        algorithm_name=algorithm_name,
        test_data_path=test_data_path,
        env_info_path="marl/data_benchmark/episodes/train_env_info.json",
        action_costs = {0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03},
        episode_idx=0,  # 使用第0个测试episode
        initial_health_level=9,  # 初始健康等级5（高）
        budget_multiplier=1.0,
        budget_allocation_strategy='rotating_focus'
    )
    
    # 运行仿真
    results = simulator.run_simulation(
        n_years=100,    # 100年
        base_budget_per_bridge=10000  # 每座桥梁基础预算10000
    )
    
    # 生成图表
    simulator.plot_health_distribution_stacked()
    simulator.plot_health_category_stacked()
    simulator.plot_cost_evolution()
    simulator.plot_cumulative_cost_evolution()
    simulator.plot_cost_comparison()
    simulator.plot_average_health_evolution()
    
    # 保存结果
    simulator.save_results()
    
    print("仿真测试完成!")


if __name__ == "__main__":
    main()