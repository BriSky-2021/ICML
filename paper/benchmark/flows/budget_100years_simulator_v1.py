import numpy as np
import torch as th
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any

class BudgetSensitivityAnalyzer:
    """
    专门用于预算敏感性分析的评估器
    直接使用数据集中的原始状态，不进行长期仿真
    支持模型缓存，避免重复加载
    """
    
    def __init__(self, test_data_path, env_info_path, action_costs, model_base_path="paper/benchmark/saved_models"):
        """
        初始化预算敏感性分析器
        
        Args:
            test_data_path: 测试数据路径
            env_info_path: 环境信息路径
            action_costs: 动作成本字典 {0: 0, 1: 71.56, 2: 1643.31, 3: 2433.53}
            model_base_path: 模型文件基础路径
        """
        self.test_data_path = test_data_path
        self.env_info_path = env_info_path
        self.action_costs = action_costs
        self.model_base_path = model_base_path
        
        # ✅ 添加模型缓存
        self.model_cache = {}  # {algorithm_name: wrapped_model}
        self.model_paths_cache = {}  # {algorithm_name: model_path}
        
        # budget缩放因子和分配策略
        self.budget_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        self.allocation_strategies = ['uniform', 'expert', 'composite']
        
        # algorithm类型映射
        self.algorithm_types = {
            'multitask_bc': 'osrl',
            'multitask_cpq': 'osrl',
            'discrete_bc': 'marl',
            'iqlcql_marl': 'marl',
            'qmix_cql': 'marl'
        }
        
        # load数据
        self._load_test_data()
        
        print(f"预算敏感性分析器初始化完成")
        print(f"测试episodes数量: {self.n_episodes}")
        print(f"时间步数: {self.T}")
        print(f"智能体数量: {self.n_agents}")
        print(f"状态维度: {self.state_dim}")
    
    def _load_test_data(self):
        """加载测试数据"""
        # load测试buffer
        buffer = th.load(self.test_data_path)
        self.test_data = self._extract_data_from_buffer(buffer)
        
        # load环境信息
        with open(self.env_info_path, 'r') as f:
            env_info = json.load(f)
        self.norm_params = env_info.get('normalization_params', {})
        
        # get数据维度信息
        self.n_episodes, self.T, self.n_agents, self.state_dim = self.test_data['obs'].shape
        print(f"加载数据完成: {self.n_episodes} episodes, {self.T} steps, {self.n_agents} agents")
    
    def _extract_data_from_buffer(self, buffer):
        """从buffer中提取数据"""
        all_obs, all_actions, all_log_budgets, all_importance = [], [], [], []
        episode_budgets, actual_n_agents_list = [], []
        
        num_eps = buffer.episodes_in_buffer
        for i in range(num_eps):
            ep = buffer[i:i+1]
            
            # 提取观察数据
            obs = ep['obs'].cpu().numpy()  # [1, T, n_agents, state_dim]
            all_obs.append(obs[0])
            
            # 提取动作数据
            actions = ep['actions'].cpu().numpy()
            if actions.ndim == 4 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            all_actions.append(actions[0])
            
            # 提取预算数据
            if 'log_budget' in ep.data.transition_data:
                log_budgets = ep['log_budget'].cpu().numpy()

                if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
                    log_budgets = log_budgets.squeeze(-1)
                all_log_budgets.append(log_budgets[0])
            
            # 提取重要性数据
            if 'importance' in ep.data.transition_data:
                importance = ep['importance'].cpu().numpy()
                if importance.ndim == 4 and importance.shape[-1] == 1:
                    importance = importance.squeeze(-1)
                # 取第一个时间步的重要性
                all_importance.append(importance[0, 0, :])
            
            # 提取episode预算
            btotal = np.expm1(ep['btotal'].cpu().numpy()) if 'btotal' in ep.data.episode_data else 100000
            episode_budgets.append(btotal)
            
            # 提取实际智能体数量
            actual_n_agents = ep['n_bridges_actual'].cpu().numpy().squeeze()
            actual_n_agents_list.append(actual_n_agents)
        
        return {
            'obs': np.stack(all_obs),  # [num_eps, T, n_agents, state_dim]
            'actions': np.stack(all_actions),  # [num_eps, T, n_agents]
            'log_budgets': np.stack(all_log_budgets) if all_log_budgets else None,  # [num_eps, T, n_agents]
            'importance': np.stack(all_importance) if all_importance else None,  # [num_eps, n_agents]
            'episode_budgets': np.array(episode_budgets),
            'actual_n_agents': np.array(actual_n_agents_list)
        }
    
    def calculate_budget_allocation(self, total_budget, strategy, n_agents, 
                                   importance=None, obs_features=None, expert_budgets=None):
        """
        计算预算分配
        
        Args:
            total_budget: 总预算
            strategy: 分配策略 ('uniform', 'expert', 'composite')
            n_agents: 智能体数量
            importance: 重要性数组 [n_agents]
            obs_features: 观察特征 [n_agents, feature_dim]
            expert_budgets: 专家预算分配 [T, n_agents] (log格式)
        """
        if strategy == 'uniform':
            # 均匀分配
            return np.full(n_agents, total_budget / n_agents)
        
        elif strategy == 'expert':
            # 基于专家数据的分配
            if expert_budgets is None:
                print("警告: 缺少专家预算数据，回退到均匀分配")
                return np.full(n_agents, total_budget / n_agents)
            
            # compute每个智能体的平均log预算
            avg_log_budgets = np.mean(expert_budgets, axis=0)  # [n_agents]
            
            # usesoftmax计算比例（数值稳定）
            exp_budgets = np.exp(avg_log_budgets - np.max(avg_log_budgets))
            proportions = exp_budgets / np.sum(exp_budgets)
            
            return total_budget * proportions
        
        elif strategy == 'composite':
            # 基于观察特征的组合分配
            if obs_features is None:
                print("警告: 缺少观察特征，回退到均匀分配")
                return np.full(n_agents, total_budget / n_agents)
            
            # from观察特征中提取相关信息
            # obs_features: [n_agents, state_dim] - [health, age, adt, length, importance]
            health_states = obs_features[:, 0]  # 健康状态（已归一化）
            ages = obs_features[:, 1] if obs_features.shape[1] > 1 else np.ones(n_agents) * 50
            adts = obs_features[:, 2] if obs_features.shape[1] > 2 else np.ones(n_agents) * 1000
            
            # use重要性信息（如果可用）
            if importance is not None:
                importance_factor = importance
            elif obs_features.shape[1] > 4:
                importance_factor = obs_features[:, 4]
            else:
                importance_factor = np.ones(n_agents) * 0.5
            
            # normalize各因子到[0,1]范围
            health_factor = 1.0 - health_states  # 健康状态越低，需要越多预算
            age_factor = (ages - np.min(ages)) / (np.max(ages) - np.min(ages) + 1e-8)
            adt_factor = (adts - np.min(adts)) / (np.max(adts) - np.min(adts) + 1e-8)
            
            # 加权组合
            composite_score = (0.4 * health_factor + 
                             0.2 * age_factor + 
                             0.2 * adt_factor + 
                             0.2 * importance_factor)
            
            # ensure都是正数
            composite_score = np.maximum(composite_score, 1e-6)
            
            # compute比例
            proportions = composite_score / np.sum(composite_score)
            
            return total_budget * proportions
        
        else:
            raise ValueError(f"未知的分配策略: {strategy}")
    
    def normalize_budget(self, budget_allocation):
        """
        将预算分配归一化为log格式（与训练时一致）
        """
        # ensure预算都是正数
        budget_allocation = np.maximum(budget_allocation, 1e-6)
        
        # 转换为log格式
        log_budgets = np.log(budget_allocation)
        
        # use训练时的归一化参数
        if 'log_budget' in self.norm_params:
            mean = self.norm_params['log_budget']['mean']
            std = self.norm_params['log_budget']['std']
            normalized_log_budgets = (log_budgets - mean) / (std + 1e-8)
        else:
            # if没有归一化参数，使用简单的标准化
            normalized_log_budgets = (log_budgets - np.mean(log_budgets)) / (np.std(log_budgets) + 1e-8)
        
        return normalized_log_budgets
    
    def _find_model_path(self, algorithm_name):
        """查找模型路径（带缓存）"""
        if algorithm_name in self.model_paths_cache:
            return self.model_paths_cache[algorithm_name]
        
        try:
            # 搜索模型文件
            model_patterns = [
                f"*{algorithm_name}*checkpoint*",
                f"*{algorithm_name}*model*", 
                f"*{algorithm_name}*.pt",
                f"*{algorithm_name}*.pth"
            ]
            
            import glob
            model_files = []
            for pattern in model_patterns:
                search_path = os.path.join(self.model_base_path, "**", pattern)
                model_files.extend(glob.glob(search_path, recursive=True))
            
            if not model_files:
                print(f"未找到算法 {algorithm_name} 的模型文件")
                return None
            
            # select最新的模型文件
            latest_model = max(model_files, key=os.path.getmtime)
            
            # 缓存路径
            self.model_paths_cache[algorithm_name] = latest_model
            print(f"找到模型路径: {latest_model}")
            
            return latest_model
        
        except Exception as e:
            print(f"查找模型路径失败: {e}")
            return None
    
    def _load_algorithm_model(self, algorithm_name):
        """加载算法模型（带缓存）"""
        # check缓存
        if algorithm_name in self.model_cache:
            print(f"使用缓存的模型: {algorithm_name}")
            return self.model_cache[algorithm_name]
        
        # 查找模型路径
        model_path = self._find_model_path(algorithm_name)
        if model_path is None:
            return None
        
        try:
            print(f"首次加载模型: {algorithm_name} from {model_path}")
            
            # load模型
            model = th.load(model_path, map_location='cuda:0')
            
            # 判断算法类型并包装
            algo_type = self.algorithm_types.get(algorithm_name, 'osrl')
            wrapped_model = ModelWrapper(model, algo_type)
            
            # ✅ 缓存模型
            self.model_cache[algorithm_name] = wrapped_model
            print(f"模型已缓存: {algorithm_name}")
            
            return wrapped_model
        
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def preload_all_models(self, algorithm_names):
        """
        预加载所有需要的模型到缓存中
        
        Args:
            algorithm_names: 算法名称列表
        """
        print(f"开始预加载 {len(algorithm_names)} 个模型...")
        
        failed_models = []
        for algo_name in algorithm_names:
            print(f"\n预加载模型: {algo_name}")
            model = self._load_algorithm_model(algo_name)
            if model is None:
                failed_models.append(algo_name)
        
        if failed_models:
            print(f"\n预加载失败的模型: {failed_models}")
        else:
            print(f"\n所有模型预加载完成!")
        
        return len(algorithm_names) - len(failed_models)
    
    def clear_model_cache(self):
        """清空模型缓存（释放内存）"""
        print(f"清空模型缓存，释放 {len(self.model_cache)} 个模型")
        self.model_cache.clear()
        th.cuda.empty_cache()  # 清空GPU缓存
    
    def get_cache_info(self):
        """获取缓存信息"""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cached_paths': self.model_paths_cache,
            'cache_size': len(self.model_cache)
        }
    
    def evaluate_episode_with_budget(self, episode_idx, algorithm_name, budget_scale, allocation_strategy):
        """
        评估单个episode在特定预算配置下的表现
        现在使用缓存的模型，大大提升速度
        
        Args:
            episode_idx: episode索引
            algorithm_name: 算法名称
            budget_scale: 预算缩放因子
            allocation_strategy: 预算分配策略
        
        Returns:
            evaluation_results: 评估结果字典
        """
        # getepisode数据
        obs = self.test_data['obs'][episode_idx]  # [T, n_agents, state_dim]
        original_actions = self.test_data['actions'][episode_idx]  # [T, n_agents]
        expert_budgets = self.test_data['log_budgets'][episode_idx] if self.test_data['log_budgets'] is not None else None
        importance = self.test_data['importance'][episode_idx] if self.test_data['importance'] is not None else None
        
        # compute基础预算
        base_budget_per_agent = 10000
        total_budget = base_budget_per_agent * self.n_agents * budget_scale
        
        # compute预算分配
        budget_allocation = self.calculate_budget_allocation(
            total_budget=total_budget,
            strategy=allocation_strategy,
            n_agents=self.n_agents,
            importance=importance,
            obs_features=obs[0],  # use第一个时间步的观察特征
            expert_budgets=expert_budgets
        )
        
        # ✅ 使用缓存的模型（不会重复加载）
        model = self._load_algorithm_model(algorithm_name)
        if model is None:
            return None
        
        # 判断算法类型
        algo_type = self.algorithm_types.get(algorithm_name, 'osrl')
        
        # 准备模型输入数据
        if algo_type == 'marl':
            # MARL算法：需要将预算拼接到观察中
            model_obs = self._prepare_marl_observations(obs, budget_allocation)
        else:
            # OSRL算法：观察和预算分开
            model_obs = obs
            normalized_budgets = self.normalize_budget(budget_allocation)
        
        # evaluate模型性能
        results = {
            'episode_idx': episode_idx,
            'algorithm': algorithm_name,
            'algorithm_type': algo_type,
            'budget_scale': budget_scale,
            'allocation_strategy': allocation_strategy,
            'total_budget': total_budget,
            'budget_allocation': budget_allocation,
            'predicted_actions': [],
            'original_actions': original_actions,
            'action_costs': [],
            'health_evolution': [],
            'performance_metrics': {}
        }
        
        # 模拟时间步执行
        current_health = obs[0, :, 0]  # 初始健康状态 [n_agents]
        total_cost = 0
        
        for t in range(self.T):
            step_actions = []
            step_costs = []
            
            for agent_id in range(self.n_agents):
                # get当前观察
                if algo_type == 'marl':
                    agent_obs = model_obs[t, agent_id]  # contains budget information
                    action = self._get_marl_action(model, agent_obs, agent_id)
                else:
                    agent_obs = model_obs[t, agent_id]  # not包含预算
                    agent_budget = normalized_budgets[agent_id] / self.T  # 分摊到每个时间步
                    action = self._get_osrl_action(model, agent_obs, agent_budget)
                
                # compute成本
                cost = self.action_costs.get(action, 0)
                step_costs.append(cost)
                total_cost += cost
                
                step_actions.append(action)
                
                # update健康状态（简单的状态转移）
                current_health[agent_id] = self._update_health_state(
                    current_health[agent_id], action
                )
            
            results['predicted_actions'].append(step_actions)
            results['action_costs'].append(step_costs)
            results['health_evolution'].append(current_health.copy())
        
        # compute性能指标
        results['performance_metrics'] = self._calculate_performance_metrics(
            results, obs[0, :, 0], current_health
        )
        
        return results
    
    def _prepare_marl_observations(self, obs, budget_allocation):
        """
        为MARL算法准备包含预算信息的观察
        
        Args:
            obs: 原始观察 [T, n_agents, state_dim]
            budget_allocation: 预算分配 [n_agents]
        
        Returns:
            augmented_obs: 增强观察 [T, n_agents, state_dim+1]
        """
        T, n_agents, state_dim = obs.shape
        
        # normalize预算
        normalized_budgets = self.normalize_budget(budget_allocation)
        
        # will预算分摊到每个时间步
        budget_per_step = normalized_budgets / T
        
        # 扩展预算到时间维度
        budget_sequence = np.tile(budget_per_step.reshape(1, -1, 1), (T, 1, 1))  # [T, n_agents, 1]
        
        # concat观察和预算
        augmented_obs = np.concatenate([obs, budget_sequence], axis=-1)  # [T, n_agents, state_dim+1]
        
        return augmented_obs
    
    def _get_marl_action(self, model, obs_with_budget, agent_id):
        """获取MARL算法的动作"""
        try:
            # 构建输入格式
            max_n_agents = 500  # 假设最大智能体数量
            obs_in = np.zeros((1, max_n_agents, len(obs_with_budget)), dtype=np.float32)
            obs_in[0, agent_id, :] = obs_with_budget
            
            mask_in = np.zeros((1, max_n_agents), dtype=np.float32)
            mask_in[0, agent_id] = 1.0
            
            # 转换为tensor
            obs_tensor = th.tensor(obs_in, dtype=th.float32)
            
            # model预测（ModelWrapper已经处理了评估模式和无梯度计算）
            pred = model.act(obs_tensor, mask_in, legal_actions=None)
            
            return int(pred[0, agent_id])
        
        except Exception as e:
            print(f"MARL动作预测失败: {e}")
            return 0
    
    def _get_osrl_action(self, model, obs, budget):
        """获取OSRL算法的动作"""
        try:
            # 准备输入
            obs_in = obs.reshape(1, -1)
            obs_tensor = th.tensor(obs_in, dtype=th.float32)
            budget_tensor = th.tensor([budget], dtype=th.float32)
            
            # model预测（ModelWrapper已经处理了评估模式和无梯度计算）
            pred = model.act(obs_tensor, budget_tensor)
            
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            
            return int(pred.flatten()[0])
        
        except Exception as e:
            print(f"OSRL动作预测失败: {e}")
            return 0
    
    def _update_health_state(self, current_health, action):
        """
        简单的健康状态转移函数
        
        Args:
            current_health: 当前健康状态（归一化值）
            action: 维修动作 (0: 无动作, 1: 小修, 2: 大修, 3: 重建)
        
        Returns:
            new_health: 新的健康状态
        """
        # 反归一化到0-9范围
        health_level = int(current_health * 9)
        
        # 自然退化
        degradation = np.random.choice([0, 1], p=[0.7, 0.3])  # 30%概率退化1级
        health_level = max(0, health_level - degradation)
        
        # 维修效果
        if action == 1:  # 小修
            health_level = min(9, health_level + 1)
        elif action == 2:  # 大修
            health_level = min(9, health_level + 3)
        elif action == 3:  # 重建
            health_level = 9
        
        # 重新归一化
        return health_level / 9.0
    
    def _calculate_performance_metrics(self, results, initial_health, final_health):
        """计算性能指标"""
        # compute总成本
        total_cost = sum([sum(step_costs) for step_costs in results['action_costs']])
        
        # compute健康改善
        initial_avg_health = np.mean(initial_health)
        final_avg_health = np.mean(final_health)
        health_improvement = final_avg_health - initial_avg_health
        
        # compute动作分布
        all_actions = np.concatenate(results['predicted_actions'])
        action_distribution = {
            action: np.sum(all_actions == action) / len(all_actions)
            for action in range(4)
        }
        
        # compute成本效率
        cost_efficiency = health_improvement / (total_cost + 1e-8) * 1000
        
        return {
            'total_cost': total_cost,
            'initial_avg_health': initial_avg_health,
            'final_avg_health': final_avg_health,
            'health_improvement': health_improvement,
            'cost_efficiency': cost_efficiency,
            'action_distribution': action_distribution,
            'budget_utilization': total_cost / results['total_budget']
        }
    
    def run_budget_scale_sensitivity(self, algorithm_names, n_episodes=50):
        """
        运行预算缩放敏感性分析（使用模型缓存优化）
        
        Args:
            algorithm_names: 算法名称列表
            n_episodes: 评估的episode数量
        """
        print(f"开始预算缩放敏感性分析...")
        print(f"固定分配策略: expert")
        print(f"预算缩放因子: {self.budget_scales}")
        print(f"评估episodes: {n_episodes}")
        print("=" * 80)
        
        # ✅ 预加载所有模型
        successful_models = self.preload_all_models(algorithm_names)
        if successful_models == 0:
            print("没有成功加载任何模型，退出")
            return {}
        
        fixed_strategy = 'expert'
        results = {}
        
        total_experiments = len(algorithm_names) * len(self.budget_scales)
        experiment_count = 0
        
        for algo_name in algorithm_names:
            if algo_name not in self.model_cache:
                print(f"跳过未成功加载的算法: {algo_name}")
                continue
                
            print(f"\n评估算法: {algo_name} (使用缓存模型)")
            
            for budget_scale in self.budget_scales:
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] 预算缩放: {budget_scale}x")
                
                episode_results = []
                
                # evaluate多个episodes
                for ep_idx in range(min(n_episodes, self.n_episodes)):
                    if ep_idx % 10 == 0 and ep_idx > 0:
                        print(f"    处理episode {ep_idx}/{min(n_episodes, self.n_episodes)}")
                    
                    ep_result = self.evaluate_episode_with_budget(
                        episode_idx=ep_idx,
                        algorithm_name=algo_name,
                        budget_scale=budget_scale,
                        allocation_strategy=fixed_strategy
                    )
                    
                    if ep_result is not None:
                        episode_results.append(ep_result)
                
                if episode_results:
                    # compute汇总统计
                    aggregated_result = self._aggregate_episode_results(episode_results)
                    results[f"{algo_name}_scale{budget_scale}"] = aggregated_result
                    
                    print(f"    完成 {len(episode_results)} episodes, "
                          f"平均成本: {aggregated_result['avg_total_cost']:.0f}, "
                          f"平均健康改善: {aggregated_result['avg_health_improvement']:.3f}")
        
        print(f"\n预算缩放敏感性分析完成!")
        print(f"缓存信息: {self.get_cache_info()}")
        return results
    
    def run_allocation_strategy_comparison(self, algorithm_names, n_episodes=50):
        """
        运行预算分配策略对比分析（使用模型缓存优化）
        
        Args:
            algorithm_names: 算法名称列表
            n_episodes: 评估的episode数量
        """
        print(f"开始预算分配策略对比分析...")
        print(f"固定预算缩放: 1.0x")
        print(f"分配策略: {self.allocation_strategies}")
        print(f"评估episodes: {n_episodes}")
        print("=" * 80)
        
        # ✅ 如果模型还没有预加载，则预加载
        if not self.model_cache:
            successful_models = self.preload_all_models(algorithm_names)
            if successful_models == 0:
                print("没有成功加载任何模型，退出")
                return {}
        
        fixed_scale = 1.0
        results = {}
        
        total_experiments = len(algorithm_names) * len(self.allocation_strategies)
        experiment_count = 0
        
        for algo_name in algorithm_names:
            if algo_name not in self.model_cache:
                print(f"跳过未成功加载的算法: {algo_name}")
                continue
                
            print(f"\n评估算法: {algo_name} (使用缓存模型)")
            
            for strategy in self.allocation_strategies:
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] 分配策略: {strategy}")
                
                episode_results = []
                
                # evaluate多个episodes
                for ep_idx in range(min(n_episodes, self.n_episodes)):
                    if ep_idx % 10 == 0 and ep_idx > 0:
                        print(f"    处理episode {ep_idx}/{min(n_episodes, self.n_episodes)}")
                    
                    ep_result = self.evaluate_episode_with_budget(
                        episode_idx=ep_idx,
                        algorithm_name=algo_name,
                        budget_scale=fixed_scale,
                        allocation_strategy=strategy
                    )
                    
                    if ep_result is not None:
                        episode_results.append(ep_result)
                
                if episode_results:
                    # compute汇总统计
                    aggregated_result = self._aggregate_episode_results(episode_results)
                    results[f"{algo_name}_strategy{strategy}"] = aggregated_result
                    
                    print(f"    完成 {len(episode_results)} episodes, "
                          f"平均成本: {aggregated_result['avg_total_cost']:.0f}, "
                          f"平均健康改善: {aggregated_result['avg_health_improvement']:.3f}")
        
        print(f"\n预算分配策略对比分析完成!")
        print(f"缓存信息: {self.get_cache_info()}")
        return results
    
    def _aggregate_episode_results(self, episode_results):
        """聚合多个episode的结果"""
        # 提取各种指标
        metrics = [ep['performance_metrics'] for ep in episode_results]
        
        aggregated = {
            'algorithm': episode_results[0]['algorithm'],
            'algorithm_type': episode_results[0]['algorithm_type'],
            'budget_scale': episode_results[0]['budget_scale'],
            'allocation_strategy': episode_results[0]['allocation_strategy'],
            'n_episodes': len(episode_results),
            
            # mean指标
            'avg_total_cost': np.mean([m['total_cost'] for m in metrics]),
            'std_total_cost': np.std([m['total_cost'] for m in metrics]),
            'avg_health_improvement': np.mean([m['health_improvement'] for m in metrics]),
            'std_health_improvement': np.std([m['health_improvement'] for m in metrics]),
            'avg_cost_efficiency': np.mean([m['cost_efficiency'] for m in metrics]),
            'std_cost_efficiency': np.std([m['cost_efficiency'] for m in metrics]),
            'avg_budget_utilization': np.mean([m['budget_utilization'] for m in metrics]),
            'std_budget_utilization': np.std([m['budget_utilization'] for m in metrics]),
            
            # 详细结果
            'episode_results': episode_results
        }
        
        return aggregated
    
    def plot_results(self, scale_results=None, strategy_results=None, save_dir="budget_analysis_results"):
        """绘制分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if scale_results:
            self._plot_budget_scale_sensitivity(scale_results, save_dir, timestamp)
        
        if strategy_results:
            self._plot_allocation_strategy_comparison(strategy_results, save_dir, timestamp)
    
    def _plot_budget_scale_sensitivity(self, scale_results, save_dir, timestamp):
        """绘制预算缩放敏感性图表"""
        algorithms = list(set([r['algorithm'] for r in scale_results.values()]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = [
            ('avg_total_cost', '平均总成本', axes[0, 0]),
            ('avg_health_improvement', '平均健康改善', axes[0, 1]),
            ('avg_cost_efficiency', '平均成本效率', axes[1, 0]),
            ('avg_budget_utilization', '平均预算利用率', axes[1, 1])
        ]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        for metric_key, metric_name, ax in metrics:
            for i, algo in enumerate(algorithms):
                scales = []
                values = []
                errors = []
                
                for scale in self.budget_scales:
                    key = f"{algo}_scale{scale}"
                    if key in scale_results:
                        scales.append(scale)
                        values.append(scale_results[key][metric_key])
                        
                        # add误差棒
                        error_key = metric_key.replace('avg_', 'std_')
                        if error_key in scale_results[key]:
                            errors.append(scale_results[key][error_key])
                        else:
                            errors.append(0)
                
                if scales:
                    ax.errorbar(scales, values, yerr=errors, label=algo, 
                               color=colors[i], marker='o', linewidth=2, capsize=5)
            
            ax.set_xlabel('Budget Scale Factor')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Budget Scale\n(Expert Allocation)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'budget_scale_sensitivity_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预算缩放敏感性图表已保存: {save_path}")
        plt.show()
    
    def _plot_allocation_strategy_comparison(self, strategy_results, save_dir, timestamp):
        """绘制分配策略对比图表"""
        algorithms = list(set([r['algorithm'] for r in strategy_results.values()]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = [
            ('avg_total_cost', '平均总成本', axes[0, 0]),
            ('avg_health_improvement', '平均健康改善', axes[0, 1]),
            ('avg_cost_efficiency', '平均成本效率', axes[1, 0]),
            ('avg_budget_utilization', '平均预算利用率', axes[1, 1])
        ]
        
        x = np.arange(len(self.allocation_strategies))
        width = 0.8 / len(algorithms)
        
        for metric_key, metric_name, ax in metrics:
            for i, algo in enumerate(algorithms):
                values = []
                errors = []
                
                for strategy in self.allocation_strategies:
                    key = f"{algo}_strategy{strategy}"
                    if key in strategy_results:
                        values.append(strategy_results[key][metric_key])
                        
                        # add误差棒
                        error_key = metric_key.replace('avg_', 'std_')
                        if error_key in strategy_results[key]:
                            errors.append(strategy_results[key][error_key])
                        else:
                            errors.append(0)
                    else:
                        values.append(0)
                        errors.append(0)
                
                ax.bar(x + i * width, values, width, yerr=errors, 
                      label=algo, alpha=0.8, capsize=5)
            
            ax.set_xlabel('Allocation Strategy')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by Strategy\n(1.0x Budget Scale)')
            ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
            ax.set_xticklabels(self.allocation_strategies)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'allocation_strategy_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分配策略对比图表已保存: {save_path}")
        plt.show()
    
    def save_results(self, scale_results=None, strategy_results=None, save_dir="budget_analysis_results"):
        """保存结果到文件"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # save详细结果
        if scale_results:
            scale_path = os.path.join(save_dir, f'budget_scale_results_{timestamp}.json')
            with open(scale_path, 'w') as f:
                json.dump(self._serialize_results(scale_results), f, indent=2)
            print(f"预算缩放结果已保存: {scale_path}")
        
        if strategy_results:
            strategy_path = os.path.join(save_dir, f'allocation_strategy_results_{timestamp}.json')
            with open(strategy_path, 'w') as f:
                json.dump(self._serialize_results(strategy_results), f, indent=2)
            print(f"分配策略结果已保存: {strategy_path}")
        
        # save汇总表格
        self._save_summary_tables(scale_results, strategy_results, save_dir, timestamp)
    
    def _serialize_results(self, results):
        """序列化结果为JSON格式"""
        serialized = {}
        for key, result in results.items():
            # 移除episode_results中不能序列化的部分，只保留汇总统计
            clean_result = {k: v for k, v in result.items() if k != 'episode_results'}
            
            # ensure所有numpy数据类型都转换为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            serialized[key] = convert_numpy(clean_result)
        
        return serialized
    
    def _save_summary_tables(self, scale_results, strategy_results, save_dir, timestamp):
        """保存汇总表格"""
        if scale_results:
            # budget缩放汇总表
            scale_rows = []
            for key, result in scale_results.items():
                scale_rows.append({
                    'Algorithm': result['algorithm'],
                    'Budget_Scale': f"{result['budget_scale']}x",
                    'Avg_Total_Cost': f"{result['avg_total_cost']:.0f}",
                    'Std_Total_Cost': f"{result['std_total_cost']:.0f}",
                    'Avg_Health_Improvement': f"{result['avg_health_improvement']:.3f}",
                    'Avg_Cost_Efficiency': f"{result['avg_cost_efficiency']:.4f}",
                    'Avg_Budget_Utilization': f"{result['avg_budget_utilization']:.3f}",
                    'N_Episodes': result['n_episodes']
                })
            
            scale_df = pd.DataFrame(scale_rows)
            scale_csv_path = os.path.join(save_dir, f'budget_scale_summary_{timestamp}.csv')
            scale_df.to_csv(scale_csv_path, index=False)
            print(f"预算缩放汇总表已保存: {scale_csv_path}")
        
        if strategy_results:
            # 分配策略汇总表
            strategy_rows = []
            for key, result in strategy_results.items():
                strategy_rows.append({
                    'Algorithm': result['algorithm'],
                    'Allocation_Strategy': result['allocation_strategy'],
                    'Avg_Total_Cost': f"{result['avg_total_cost']:.0f}",
                    'Std_Total_Cost': f"{result['std_total_cost']:.0f}",
                    'Avg_Health_Improvement': f"{result['avg_health_improvement']:.3f}",
                    'Avg_Cost_Efficiency': f"{result['avg_cost_efficiency']:.4f}",
                    'Avg_Budget_Utilization': f"{result['avg_budget_utilization']:.3f}",
                    'N_Episodes': result['n_episodes']
                })
            
            strategy_df = pd.DataFrame(strategy_rows)
            strategy_csv_path = os.path.join(save_dir, f'allocation_strategy_summary_{timestamp}.csv')
            strategy_df.to_csv(strategy_csv_path, index=False)
            print(f"分配策略汇总表已保存: {strategy_csv_path}")

# create一个统一的模型wrapper来处理评估模式
class ModelWrapper:
    """统一的模型包装器，确保正确的评估模式管理"""
    
    def __init__(self, model, algorithm_type):
        self.model = model
        self.algorithm_type = algorithm_type
        self._set_eval_mode()
    
    def _set_eval_mode(self):
        """设置评估模式"""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, 'q_network'):
            # for于有多个网络的模型
            for attr_name in ['q_network', 'target_q_network', 'policy_network', 'mixer', 'target_mixer']:
                if hasattr(self.model, attr_name):
                    network = getattr(self.model, attr_name)
                    if hasattr(network, 'eval'):
                        network.eval()
    
    def act(self, *args, **kwargs):
        """统一的动作接口"""
        with th.no_grad():  # ensure无梯度计算
            return self.model.act(*args, **kwargs)

def main():
    """主函数 - 使用模型缓存优化"""
    print("预算敏感性分析（模型缓存优化版）")
    print("=" * 60)
    
    # init分析器
    analyzer = BudgetSensitivityAnalyzer(
        test_data_path="marl/data_benchmark/episodes/test_buffer.pt",
        env_info_path="marl/data_benchmark/episodes/train_env_info.json",
        action_costs={0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03},
        model_base_path="paper/benchmark/saved_models"
    )
    
    # 要分析的算法
    algorithm_names = [
        "discrete_bc"
    ]
    
    try:
        # 1. 预算缩放敏感性分析
        print("\n" + "="*60)
        print("第一部分：预算缩放敏感性分析")
        print("="*60)
        scale_results = analyzer.run_budget_scale_sensitivity(
            algorithm_names=algorithm_names,
            n_episodes=20  # evaluate20个episodes
        )
        
        # 2. 预算分配策略对比分析（复用已缓存的模型）
        print("\n" + "="*60)
        print("第二部分：预算分配策略对比分析（复用缓存模型）") 
        print("="*60)
        strategy_results = analyzer.run_allocation_strategy_comparison(
            algorithm_names=algorithm_names,
            n_episodes=20  # evaluate20个episodes
        )
        
        if not scale_results and not strategy_results:
            print("没有成功运行任何实验，退出")
            return
        
        # 3. 生成分析图表和保存结果
        print("\n3. 生成分析图表和报告")
        print("-" * 40)
        analyzer.plot_results(scale_results, strategy_results)
        analyzer.save_results(scale_results, strategy_results)
        
        print("\n预算敏感性分析完成!")
        print(f"预算缩放敏感性实验: {len(scale_results)} 个")
        print(f"分配策略对比实验: {len(strategy_results)} 个") 
        print(f"总实验数量: {len(scale_results) + len(strategy_results)} 个")
        
        # 显示缓存统计
        cache_info = analyzer.get_cache_info()
        print(f"模型缓存统计: {cache_info}")
        
    finally:
        # 4. 清理缓存（可选）
        print("\n清理模型缓存...")
        analyzer.clear_model_cache()


if __name__ == "__main__":
    main()