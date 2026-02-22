# evaluate_fqe_v2.py
"""
FQE 评估脚本（v2）
支持单模型与批处理：给定算法列表，自动查找最新模型，按算法配置 FQE 参数，一次性跑完并分算法保存结果。
"""
import os
import sys
import glob
import torch as th
import numpy as np
import json
import yaml
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARK_ROOT)

# 导入必要的模块
from algos.FQE import FQE
from Multi_Task_Run_v4 import (
    load_config, load_env_info, load_buffer,
    extract_data_from_buffer, build_legal_actions_from_log_budgets
)
from evaluate_unified_v4 import ModelWrapper
from heuristic_bc_base import project_health_to_categories

# --- 与 evaluate_with_budget_bc_v3_no_limit 对齐的配置 ---
ALGO_TYPE = {
    "qmix_cql": "marl",
    "iqlcql_marl": "marl",
    "multitask_cpq": "osrl",
    "multitask_offline_cpq": "osrl",
    "onestep": "osrl",
    "random_osrl": "osrl",
    "cdt": "cdt",
    "cql": "osrl",
    "cql_heuristic": "osrl",
    "random_osrl": "marl",
    "discrete_bc": "marl",
    "multitask_bc": "osrl",
    "multitask_bc_top20": "osrl",
    "multitask_bc_top50": "osrl",
}

# default要跑的算法列表（批处理模式使用）
TARGET_ALGORITHMS = [
    #"onestep",
    #"cql",
    #"multitask_bc",
    #"multitask_offline_cpq",
    #"cql_heuristic",
]

# 纯粹的启发式算法列表（没有模型，只根据health状态决定动作）
# 这些算法会使用FQE_heuristic进行评估
PURE_HEURISTIC_ALGORITHMS = [
    "heuristic_policy",
]

MODEL_DIR = "paper/benchmark/saved_models"

# 各算法 FQE 小参数：fqe_epochs, fqe_batch_size, gamma, fqe_lr
# not yet列出的算法使用 DEFAULT_FQE_PARAMS
DEFAULT_FQE_PARAMS = {
    "fqe_epochs": 10,
    "fqe_batch_size": 256,
    "gamma": 0.99,
    "fqe_lr": 5e-5,
}

FQE_PARAMS_BY_ALGO = {
    "onestep": {"fqe_epochs": 12, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "onestep_heuristic":{"fqe_epochs": 10, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "cql": {"fqe_epochs": 15, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "cql_heuristic": {"fqe_epochs": 15, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "multitask_bc": {"fqe_epochs": 12, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "multitask_bc_top20": {"fqe_epochs": 12, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "multitask_bc_top50": {"fqe_epochs": 12, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "multitask_cpq": {"fqe_epochs": 15, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "multitask_offline_cpq": {"fqe_epochs": 15, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "qmix_cql": {"fqe_epochs": 20, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "iqlcql_marl": {"fqe_epochs": 20, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "discrete_bc": {"fqe_epochs": 12, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "random_osrl": {"fqe_epochs": 10, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "random_marl": {"fqe_epochs": 10, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
    "heuristic_policy": {"fqe_epochs": 15, "fqe_batch_size": 512, "gamma": 0.99, "fqe_lr": 5e-5},
}


def find_latest_model(model_dir=MODEL_DIR, algorithm_name=None):
    """按时间戳查找最新模型文件"""
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return None
    if algorithm_name:
        pattern = os.path.join(model_dir, f"{algorithm_name}_model_*.pth")
    else:
        pattern = os.path.join(model_dir, "*_model_*.pth")
    files = glob.glob(pattern)
    if not files:
        print(f"未找到模型: {pattern}")
        return None
    latest = max(files, key=os.path.getmtime)
    print(f"找到最新模型: {latest}")
    return latest


def extract_algorithm_name(model_path):
    """从模型路径提取算法名：{algo}_model_{timestamp}.pth -> algo"""
    if model_path is None:
        return None
    basename = os.path.basename(model_path)
    if "_model_" in basename:
        return basename.split("_model_")[0]
    return None


def get_algorithm_type(model_path):
    """根据模型路径获取算法类型 (marl / osrl / cdt)"""
    if model_path is None:
        return None
    algo = extract_algorithm_name(model_path)
    if algo and algo in ALGO_TYPE:
        return ALGO_TYPE[algo]
    return "marl"


def get_fqe_params_for_algo(algo_name):
    """获取某算法的 FQE 参数，未配置则用默认"""
    base = dict(DEFAULT_FQE_PARAMS)
    overrides = FQE_PARAMS_BY_ALGO.get(algo_name, {})
    base.update(overrides)
    return base


'''
单模型示例:
  python paper/benchmark/flows/evaluate_fqe_v3.py \\
    --model_path paper/benchmark/saved_models/onestep_model_20260122_113606.pth \\
    --test_buffer marl/data_benchmark/episodes/test_buffer.pt \\
    --config paper/benchmark/flows/config.yaml --device_id 2 \\
    --fqe_epochs 12 --fqe_batch_size 512 --gamma 0.99 \\
    --output_dir paper/benchmark/fqe_results

批处理示例（自动找最新模型、按算法配置 FQE、分算法保存）:
  python paper/benchmark/flows/evaluate_fqe_v3.py --batch \\
    --target_algorithms onestep cql multitask_bc multitask_offline_cpq \\
    --test_buffer marl/data_benchmark/episodes/test_buffer.pt \\
    --config paper/benchmark/flows/config.yaml --device_id 0 \\
    --output_dir paper/benchmark/fqe_results
'''



class FQE_heuristic(FQE):
    """
    FQE的启发式版本，用于评估启发式算法。
    从obs的第一个维度（归一化的健康值）中提取健康状态，然后根据健康状态决定动作。
    启发式策略：
    - poor/fair (health <= 1 或 health == 2) -> action1
    - good (health == 3) -> action2
    - excellent (health > 3) -> action0 (不做处理)
    """
    def __init__(self, state_dim, action_dim, device='cpu', hidden_sizes=[256, 256], lr=5e-5, verbose=False, 
                 norm_params=None):
        """
        参数:
            norm_params: 归一化参数，包含policy_obs的mean和std，用于反归一化健康值
        """
        super(FQE_heuristic, self).__init__(state_dim, action_dim, device, hidden_sizes, lr, verbose)
        self.state_dim = state_dim  # savestate_dim以便后续使用
        self.norm_params = norm_params
        if norm_params is not None and 'policy_obs' in norm_params:
            self.health_mean = norm_params['policy_obs']['mean'][0]
            self.health_std = norm_params['policy_obs']['std'][0]
        else:
            # if没有提供归一化参数，使用默认值（从test_env_info.json中获取的）
            self.health_mean = 6.329548714232343
            self.health_std = 1.2265953639981502
    
    def _denormalize_health(self, normalized_health):
        """
        将归一化的健康值反归一化为原始健康值
        """
        return normalized_health * (self.health_std + 1e-8) + self.health_mean
    
    def _extract_health_from_obs(self, obs):
        """
        从obs中提取健康值（obs的第一个维度）
        如果obs包含budget，需要先分离出policy_obs部分
        """
        # obs的形状可能是 [obs_dim] 或 [B, obs_dim] 或 [B, obs_dim+1]（如果包含budget）
        # 健康值总是在第一个维度（索引0）
        if obs.dim() == 0:
            # scalar
            health_normalized = obs
        elif obs.dim() == 1:
            # 单个obs向量
            health_normalized = obs[0]
        else:
            # batch obs [B, obs_dim] 或 [B, obs_dim+1]
            health_normalized = obs[:, 0]
        return health_normalized
    
    def _get_heuristic_action_from_obs(self, obs):
        """
        从obs中提取健康值，然后返回启发式动作
        """
        # 提取归一化的健康值
        health_normalized = self._extract_health_from_obs(obs)
        
        # 转换为numpy标量（如果是tensor）
        if th.is_tensor(health_normalized):
            health_normalized = health_normalized.cpu().item() if health_normalized.numel() == 1 else health_normalized.cpu().numpy()
        
        # 反归一化
        health_raw = self._denormalize_health(health_normalized)
        
        # ensurehealth_raw是标量
        if isinstance(health_raw, (np.ndarray, th.Tensor)):
            if hasattr(health_raw, 'item'):
                health_raw = health_raw.item()
            else:
                health_raw = float(health_raw)
        
        # map到类别
        h_cat = project_health_to_categories(health_raw)
        
        # according to类别返回动作
        if h_cat <= 1:  # critical (0) 或 poor (1) -> action1
            return 1
        elif h_cat == 2:  # fair (2) -> action1
            return 1
        elif h_cat == 3:  # good (3) -> action2
            return 2
        else:  # excellent -> action0 (不做处理)
            return 0
    
    def train_step(self, batch, target_policy_model, gamma=0.99, is_continuous_action=False):
        """
        训练一步 FQE（启发式版本）
        从obs中提取健康状态来决定动作，而不是使用模型的动作
        """
        obs = batch['obs']
        actions = batch['actions']  # here不使用，而是根据health计算
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # ensurerewards和dones的形状正确
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # handle预算
        has_budgets = 'budgets' in batch and batch['budgets'] is not None
        if has_budgets:
            budgets = batch['budgets']
            next_budgets = batch.get('next_budgets', None)
            
            if budgets.dim() == 1: 
                budgets = budgets.unsqueeze(-1)
            if next_budgets is not None and next_budgets.dim() == 1:
                next_budgets = next_budgets.unsqueeze(-1)
        else:
            budgets = None
            next_budgets = None
        
        # fromobs中提取健康值并计算启发式动作
        batch_size = obs.shape[0]
        heuristic_actions = []
        heuristic_next_actions = []
        
        # ifobs包含budget，需要先分离出policy_obs部分
        # note：这里的state_dim是policy_obs的维度（通常是5），而obs可能包含budget（维度为6）
        # 但FQE的state_dim可能已经包含了budget（fqe_state_dim = state_dim + 1）
        # 所以我们需要检查：如果obs的最后一个维度是budget，那么policy_obs是前state_dim-1个维度
        # 但实际上，健康值总是在第一个维度（索引0），不管是否包含budget
        
        for i in range(batch_size):
            # from当前obs提取健康值并计算动作
            current_obs_single = obs[i]
            action = self._get_heuristic_action_from_obs(current_obs_single)
            heuristic_actions.append(action)
            
            # fromnext_obs提取健康值并计算动作
            next_obs_single = next_obs[i]
            next_action = self._get_heuristic_action_from_obs(next_obs_single)
            heuristic_next_actions.append(next_action)
        
        heuristic_actions = th.tensor(heuristic_actions, dtype=th.long, device=self.device)
        heuristic_next_actions = th.tensor(heuristic_next_actions, dtype=th.long, device=self.device)
        
        # 1. 计算 Target Q 值（使用启发式next_action）
        with th.no_grad():
            fqe_next_obs = next_obs if not has_budgets else th.cat([next_obs, next_budgets], dim=-1)
            next_qs = self.target_q_net(fqe_next_obs)
            
            if heuristic_next_actions.dim() == 1:
                heuristic_next_actions = heuristic_next_actions.unsqueeze(-1)
            
            heuristic_next_actions = heuristic_next_actions.clamp(0, self.action_dim - 1)
            next_v = next_qs.gather(1, heuristic_next_actions)
            
            target_q = rewards + gamma * (1 - dones) * next_v
        
        # 2. 计算 Current Q 值（使用启发式action）
        fqe_obs = obs if not has_budgets else th.cat([obs, budgets], dim=-1)
        current_qs = self.q_net(fqe_obs)
        
        if heuristic_actions.dim() == 1:
            heuristic_actions = heuristic_actions.unsqueeze(-1)
        
        heuristic_actions = heuristic_actions.long().clamp(0, self.action_dim - 1)
        current_q = current_qs.gather(1, heuristic_actions)
        
        # 3. 更新
        if current_q.shape != target_q.shape:
            if target_q.dim() == 2 and target_q.shape[1] != 1:
                target_q = target_q[:, 0:1]
            elif target_q.dim() == 1:
                target_q = target_q.unsqueeze(-1)
            if current_q.dim() == 2 and current_q.shape[1] != 1:
                current_q = current_q[:, 0:1]
            elif current_q.dim() == 1:
                current_q = current_q.unsqueeze(-1)
        
        loss = th.nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_policy_probs(self, model, obs, budgets=None, needs_budget_param=False):
        """
        获取启发式策略的动作概率分布
        对于启发式策略，我们根据obs中的健康值直接返回确定性分布
        """
        # handleobs的维度
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        probs = th.zeros((batch_size, self.action_dim), device=self.device)
        
        for i in range(batch_size):
            obs_single = obs[i]
            action = self._get_heuristic_action_from_obs(obs_single)
            probs[i, action] = 1.0
        
        return probs


def prepare_transitions_for_fqe(
    data, actions, rewards, budgets=None, health=None, 
    actual_n_agents=None, use_agent_mask=True
):
    """
    将episode格式的数据转换为transition格式，用于FQE训练
    
    参数:
        data: [num_eps, T, n_agents, state_dim]
        actions: [num_eps, T, n_agents]
        rewards: [num_eps, T, n_agents]
        budgets: 预算信息
        actual_n_agents: 每个episode的实际智能体数量
        use_agent_mask: 是否只使用激活的智能体
    
    返回:
        transitions: dict，包含obs, actions, rewards, next_obs, dones, budgets, next_budgets
    """
    num_eps, T, n_agents, state_dim = data.shape
    
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_dones = []
    all_budgets = []
    all_next_budgets = []
    
    for ep in range(num_eps):
        # 确定激活的智能体
        if use_agent_mask and actual_n_agents is not None:
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
        else:
            active_idx = list(range(n_agents))
        
        for t in range(T):
            for a in active_idx:
                obs = data[ep, t, a]  # [state_dim]
                action = int(actions[ep, t, a])
                
                # get奖励
                if rewards is not None:
                    reward = float(rewards[ep, t, a])
                else:
                    reward = 0.0
                
                # get下一个状态
                if t < T - 1:
                    next_obs = data[ep, t+1, a]
                    done = 0.0
                else:
                    next_obs = data[ep, t, a]  # last一个时间步
                    done = 1.0
                
                # handle预算
                budget = None
                next_budget = None
                if budgets is not None:
                    if budgets.ndim == 4:
                        budget = float(budgets[ep, t, a, 0]) if budgets.shape[-1] == 1 else float(budgets[ep, t, a])
                        if t < T - 1:
                            next_budget = float(budgets[ep, t+1, a, 0]) if budgets.shape[-1] == 1 else float(budgets[ep, t+1, a])
                        else:
                            next_budget = budget
                    elif budgets.ndim == 3:
                        budget = float(budgets[ep, t, a])
                        if t < T - 1:
                            next_budget = float(budgets[ep, t+1, a])
                        else:
                            next_budget = budget
                    elif budgets.ndim == 2:
                        # [num_eps, T] - 每个episode每个时间步一个预算值
                        budget = float(budgets[ep, t])
                        next_budget = float(budgets[ep, min(t+1, T-1)])
                
                all_obs.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_obs.append(next_obs)
                all_dones.append(done)
                if budget is not None:
                    all_budgets.append(budget)
                    all_next_budgets.append(next_budget)
    
    transitions = {
        'obs': np.array(all_obs),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'next_obs': np.array(all_next_obs),
        'dones': np.array(all_dones)
    }
    
    if len(all_budgets) > 0:
        transitions['budgets'] = np.array(all_budgets)
        transitions['next_budgets'] = np.array(all_next_budgets)
    
    return transitions


def train_fqe(
    fqe_model, transitions, target_policy_model, 
    device='cuda', num_epochs=50, batch_size=256, 
    gamma=0.99, verbose=True
):
    """
    训练FQE模型
    """
    # 转换为tensor
    obs_tensor = th.tensor(transitions['obs'], dtype=th.float32, device=device)
    actions_tensor = th.tensor(transitions['actions'], dtype=th.long, device=device)
    rewards_tensor = th.tensor(transitions['rewards'], dtype=th.float32, device=device)
    next_obs_tensor = th.tensor(transitions['next_obs'], dtype=th.float32, device=device)
    dones_tensor = th.tensor(transitions['dones'], dtype=th.float32, device=device)
    
    # handle预算
    has_budgets = 'budgets' in transitions
    if has_budgets:
        budgets_tensor = th.tensor(transitions['budgets'], dtype=th.float32, device=device)
        next_budgets_tensor = th.tensor(transitions['next_budgets'], dtype=th.float32, device=device)
        if budgets_tensor.dim() == 1:
            budgets_tensor = budgets_tensor.unsqueeze(-1)
            next_budgets_tensor = next_budgets_tensor.unsqueeze(-1)
    
    # create数据加载器
    dataset = TensorDataset(obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, dones_tensor)
    # ensureDataLoader不会将数据移到其他设备（pin_memory=False，num_workers=0）
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    
    training_losses = []
    fqe_model.train()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_idx, (obs_b, actions_b, rewards_b, next_obs_b, dones_b) in enumerate(dataloader):
            # ensure所有batch tensor都在正确的设备上
            batch = {
                'obs': obs_b.to(device),
                'actions': actions_b.to(device),
                'rewards': rewards_b.to(device),
                'next_obs': next_obs_b.to(device),
                'dones': dones_b.to(device)
            }
            
            if has_budgets:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(budgets_tensor))
                batch['budgets'] = budgets_tensor[start_idx:end_idx].to(device)
                batch['next_budgets'] = next_budgets_tensor[start_idx:end_idx].to(device)
            
            loss = fqe_model.train_step(batch, target_policy_model, gamma=gamma)
            epoch_losses.append(loss)
            fqe_model.update_target_network()
        
        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 1 == 0:
            print(f"FQE Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return training_losses


# evaluate_fqe.py
def estimate_policy_value_with_fqe(
    fqe_model, data, budgets=None, actual_n_agents=None,
    target_policy_model=None,  # added：策略模型参数
    device='cuda', batch_size=256, verbose=True, is_heuristic=False
):
    """
    使用训练好的FQE模型估计策略的初始状态价值
    
    参数:
        fqe_model: 训练好的FQE模型
        data: [num_eps, T, n_agents, state_dim] 格式的数据
        budgets: 预算信息
        actual_n_agents: 每个episode的实际智能体数量
        target_policy_model: 待评估的策略模型（必需！用于计算策略分布）
        device: 设备
        batch_size: 批次大小
        verbose: 是否打印详细信息
    
    返回:
        value_estimates: 每个episode的价值估计列表
    """
    # for于纯粹启发式算法，不需要target_policy_model
    if target_policy_model is None and not is_heuristic:
        raise ValueError("target_policy_model 是必需的！不能使用 max Q 值，必须使用策略分布计算expected价值。")
    
    num_eps, T, n_agents, state_dim = data.shape
    
    # 检测模型是否需要单独的budget参数（纯粹启发式算法不需要）
    needs_budget_param = False
    if not is_heuristic and target_policy_model is not None:
        import inspect
        act_signature = inspect.signature(target_policy_model.act)
        act_params = list(act_signature.parameters.keys())
        needs_budget_param = 'budget' in act_params
    
    # 收集所有episode的初始状态
    initial_obs_list = []
    initial_budgets_list = []
    
    for ep in range(num_eps):
        # 确定激活的智能体
        if actual_n_agents is not None:
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
        else:
            active_idx = list(range(n_agents))
        
        for a in active_idx:
            initial_obs = data[ep, 0, a]  # [state_dim]
            initial_obs_list.append(initial_obs)
            
            if budgets is not None:
                if budgets.ndim == 4:
                    initial_budget = budgets[ep, 0, a, 0] if budgets.shape[-1] == 1 else budgets[ep, 0, a]
                elif budgets.ndim == 3:
                    initial_budget = budgets[ep, 0, a]
                elif budgets.ndim == 2:
                    initial_budget = budgets[ep, 0]
                else:
                    initial_budget = None
                initial_budgets_list.append(initial_budget)
    
    initial_obs_array = np.array(initial_obs_list)  # [num_states, state_dim]
    initial_obs_tensor = th.tensor(initial_obs_array, dtype=th.float32, device=device)
    
    has_budgets = len(initial_budgets_list) > 0 and budgets is not None
    if has_budgets:
        initial_budgets_array = np.array(initial_budgets_list)
        initial_budgets_tensor = th.tensor(initial_budgets_array, dtype=th.float32, device=device)
        if initial_budgets_tensor.dim() == 1:
            initial_budgets_tensor = initial_budgets_tensor.unsqueeze(-1)
    else:
        initial_budgets_tensor = None
    
    # useFQE估计价值（使用策略分布，而不是max Q值）
    fqe_model.eval()
    all_values = []
    
    with th.no_grad():
        for i in range(0, len(initial_obs_tensor), batch_size):
            # ensurebatch tensor在正确的设备上
            batch_obs = initial_obs_tensor[i:i+batch_size].to(device)
            batch_budgets = initial_budgets_tensor[i:i+batch_size].to(device) if has_budgets else None
            
            # get策略的动作概率分布
            if is_heuristic and isinstance(fqe_model, FQE_heuristic):
                # for于纯粹启发式算法，使用FQE_heuristic的_get_policy_probs方法
                # not需要target_policy_model，直接从obs中提取健康状态
                probs = fqe_model._get_policy_probs(
                    None,  # 纯粹启发式算法不需要模型
                    batch_obs, 
                    batch_budgets, 
                    needs_budget_param
                )
            elif target_policy_model is not None and hasattr(target_policy_model, 'get_prob'):
                # if模型有act_prob方法，直接使用
                if needs_budget_param and batch_budgets is not None:
                    probs = target_policy_model.get_prob(batch_obs, batch_budgets)
                else:
                    probs = target_policy_model.get_prob(batch_obs)
            elif target_policy_model is not None:
                # use_get_policy_probs辅助函数
                probs = fqe_model._get_policy_probs(
                    target_policy_model, 
                    batch_obs, 
                    batch_budgets, 
                    needs_budget_param
                )
            else:
                # if没有模型，使用均匀分布（不应该到达这里）
                batch_size = batch_obs.shape[0]
                probs = th.ones((batch_size, fqe_model.action_dim), device=device) / fqe_model.action_dim
            
            # ensureprobs在正确的设备上
            if isinstance(probs, th.Tensor) and probs.device != device:
                probs = probs.to(device)
            
            # getQ值
            if batch_budgets is not None:
                obs_with_budget = th.cat([batch_obs, batch_budgets], dim=-1)
            else:
                obs_with_budget = batch_obs
            
            qs = fqe_model.q_net(obs_with_budget)  # [B, action_dim]
            
            # computeexpected价值：V(s) = sum_a π(a|s) * Q(s,a)
            # probs: [B, action_dim], qs: [B, action_dim]
            values = (probs * qs).sum(dim=1)  # [B]
            all_values.extend(values.cpu().numpy().tolist())
    
    # byepisode分组
    value_estimates = []
    idx = 0
    for ep in range(num_eps):
        if actual_n_agents is not None:
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
        else:
            ep_actual_n = n_agents
        
        ep_values = all_values[idx:idx+ep_actual_n]
        value_estimates.append({
            'episode': ep,
            'mean_value': float(np.mean(ep_values)),
            'std_value': float(np.std(ep_values)),
            'values': [float(v) for v in ep_values]
        })
        idx += ep_actual_n
    
    return value_estimates


def evaluate_fqe(
    model_path,
    test_buffer_path,
    config_path="paper/benchmark/flows/config.yaml",
    device_id=0,
    fqe_epochs=50,
    fqe_batch_size=256,
    gamma=0.99,
    fqe_lr=5e-5,
    algorithm_type=None,
    algorithm_name=None,
    result_timestamp=None,
    verbose=True,
    save_results=True,
    output_dir="paper/benchmark/fqe_results"
):
    """
    使用FQE评估训练好的模型

    参数:
        model_path: 训练好的模型路径
        test_buffer_path: 测试数据buffer路径
        config_path: config file path
        device_id: GPU设备ID
        fqe_epochs: FQE训练轮数
        fqe_batch_size: FQE批次大小
        gamma: 折扣因子
        fqe_lr: FQE学习率
        algorithm_type: marl/osrl/cdt，None则从模型路径推断
        algorithm_name: 算法名，用于保存文件名；None则用模型basename
        result_timestamp: 保存文件名时间戳；None则用当前时间
        verbose: 是否打印详细信息
        save_results: 是否保存结果
        output_dir: output directory

    返回:
        results: 评估结果字典
    """
    # 检测是否是纯粹的启发式算法（没有模型，只根据health状态决定动作）
    if model_path is None:
        # 纯粹启发式算法，使用algorithm_name参数
        algo_from_path = algorithm_name if algorithm_name is not None else None
        is_pure_heuristic = algo_from_path is not None and algo_from_path in PURE_HEURISTIC_ALGORITHMS
        if algorithm_type is None:
            algorithm_type = "heuristic" if is_pure_heuristic else None
    else:
        # 普通算法，从模型路径提取算法名
        if algorithm_type is None:
            algorithm_type = get_algorithm_type(model_path)
        algo_from_path = extract_algorithm_name(model_path)
        is_pure_heuristic = algo_from_path is not None and algo_from_path in PURE_HEURISTIC_ALGORITHMS
    
    # 1. 加载配置和环境信息
    config = load_config(config_path)
    env_info = load_env_info(config['data']['env_info_file'])
    device = th.device(f"cuda:{device_id}" if th.cuda.is_available() else "cpu")
    
    state_dim = env_info['obs_shape']
    action_dim = env_info['n_actions']
    max_bridges = env_info['max_bridges']
    action_costs = {0: 0, 1: 51.06, 2: 1819.24, 3: 3785.03}
    log_budget_norm_params = env_info['normalization_params']['log_budgets']
    norm_params = env_info.get('normalization_params', None)
    
    print(f"使用设备: {device}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 2. 加载训练好的模型（纯粹启发式算法不需要模型）
    model = None
    if not is_pure_heuristic:
        if verbose:
            print(f"加载模型: {model_path}")
        
        try:
            model = th.load(model_path, map_location=device)
            # only当模型有 .to 时移动（多智能体如 DiscreteBCMultiAgent 可能没有）
            if hasattr(model, 'to'):
                model = model.to(device)
            # key：OSRL 等算法保存了 self.device，act() 里用其创建 tensor，须统一为当前 device
            if hasattr(model, 'device'):
                model.device = device
            if hasattr(model, 'modules'):
                for m in model.modules():
                    if hasattr(m, 'device'):
                        m.device = device
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    if param.device != device:
                        param.data = param.data.to(device)
            if hasattr(model, 'buffers'):
                for buf in model.buffers():
                    if buf.device != device:
                        buf.data = buf.data.to(device)
            if hasattr(model, 'eval'):
                model.eval()
            elif hasattr(model, 'train'):
                model.train(False)
            print(f"模型加载成功，已移动到设备: {device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        if verbose:
            print(f"纯粹启发式算法，无需加载模型")
    
    # 3. 加载测试数据
    if verbose:
        print(f"加载测试数据: {test_buffer_path}")
    
    test_buffer = load_buffer(test_buffer_path, device)
    if test_buffer is None:
        print("错误: 无法加载测试 buffer")
        return None
    # ensure buffer 在正确设备上（不重新赋值，因部分 .to() 就地修改且返回 None）
    if hasattr(test_buffer, 'to'):
        _ = test_buffer.to(device)
    test_data, test_actions, test_budgets, test_health, test_raw_cost, test_reward, test_log_budgets, test_actual_n_agents = extract_data_from_buffer(test_buffer)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试episode数量: {len(test_data)}")
    
    # 4. 准备transition数据
    if verbose:
        print("准备transition数据...")
    
    transitions = prepare_transitions_for_fqe(
        data=test_data,
        actions=test_actions,
        rewards=test_reward,
        budgets=test_log_budgets,
        health=test_health,
        actual_n_agents=test_actual_n_agents,
        use_agent_mask=True
    )
    
    print(f"Transition数量: {len(transitions['obs'])}")
    
    # 5. 创建FQE模型
    # 确定state_dim（如果包含预算）
    fqe_state_dim = state_dim
    if 'budgets' in transitions:
        fqe_state_dim = state_dim + 1  # budget作为额外特征
    
    # according to是否是纯粹的启发式算法选择FQE模型
    if is_pure_heuristic:
        print(f"使用FQE_heuristic评估纯粹启发式算法: {algo_from_path}")
        fqe_model = FQE_heuristic(
            state_dim=fqe_state_dim,
            action_dim=action_dim,
            device=device,
            hidden_sizes=[256, 256],
            lr=fqe_lr,
            norm_params=norm_params
        )
    else:
        # 普通算法（包括cql_heuristic等）使用普通FQE
        fqe_model = FQE(
            state_dim=fqe_state_dim,
            action_dim=action_dim,
            device=device,
            hidden_sizes=[256, 256],
            lr=fqe_lr
        )
    
    # 6. 训练FQE
    if verbose:
        print(f"开始训练FQE ({fqe_epochs} epochs, lr={fqe_lr})...")
    
    # for于纯粹启发式算法，不需要模型
    if is_pure_heuristic:
        # 纯粹启发式算法不需要模型，传入None即可
        training_losses = train_fqe(
            fqe_model=fqe_model,
            transitions=transitions,
            target_policy_model=None,
            device=device,
            num_epochs=fqe_epochs,
            batch_size=fqe_batch_size,
            gamma=gamma,
            verbose=verbose
        )
    else:
        if hasattr(model, 'to'):
            model = model.to(device)
        with ModelWrapper(model, algorithm_type=algorithm_type) as wrapped_model:
            training_losses = train_fqe(
                fqe_model=fqe_model,
                transitions=transitions,
                target_policy_model=wrapped_model,
                device=device,
                num_epochs=fqe_epochs,
                batch_size=fqe_batch_size,
                gamma=gamma,
                verbose=verbose
            )
    

    # 7. 估计策略价值
    if verbose:
        print("估计策略价值...")

    # for于纯粹启发式算法，不需要模型
    if not is_pure_heuristic and model is not None:
        if hasattr(model, 'to'):
            model = model.to(device)
    # requires passing in the policy model!
    # for purely heuristic algorithms, pass in None
    target_model = None if is_pure_heuristic else model
    value_estimates = estimate_policy_value_with_fqe(
        fqe_model=fqe_model,
        data=test_data,
        budgets=test_log_budgets,
        actual_n_agents=test_actual_n_agents,
        target_policy_model=target_model,  # purely heuristic algorithms pass in None
        device=device,
        batch_size=fqe_batch_size,
        verbose=verbose,
        is_heuristic=is_pure_heuristic
    )
    
    # 8. 汇总结果
    all_mean_values = [v['mean_value'] for v in value_estimates]
    disp_algo = algorithm_name if algorithm_name is not None else algo_from_path
    results = {
        'model_path': model_path,
        'algorithm_name': disp_algo,
        'test_buffer_path': test_buffer_path,
        'fqe_config': {
            'fqe_epochs': fqe_epochs,
            'fqe_batch_size': fqe_batch_size,
            'gamma': gamma,
            'fqe_lr': fqe_lr,
            'fqe_state_dim': fqe_state_dim,
            'action_dim': action_dim,
            'algorithm_type': algorithm_type,
        },
        'fqe_training': {
            'final_loss': float(training_losses[-1]) if training_losses else None,
            'mean_loss': float(np.mean(training_losses)) if training_losses else None,
            'training_losses': [float(l) for l in training_losses]
        },
        'value_estimates': {
            'mean': float(np.mean(all_mean_values)),
            'std': float(np.std(all_mean_values)),
            'min': float(np.min(all_mean_values)),
            'max': float(np.max(all_mean_values)),
            'per_episode': value_estimates
        },
        'num_episodes': len(value_estimates),
        'num_transitions': len(transitions['obs'])
    }
    
    # 9. 打印结果
    if verbose:
        print("\n" + "="*60)
        print("FQE评估结果")
        print("="*60)
        print(f"模型路径: {model_path}")
        print(f"测试episode数量: {len(value_estimates)}")
        print(f"Transition数量: {len(transitions['obs'])}")
        print(f"\nFQE训练:")
        print(f"  最终损失: {results['fqe_training']['final_loss']:.4f}")
        print(f"  平均损失: {results['fqe_training']['mean_loss']:.4f}")
        print(f"\n策略价值估计:")
        print(f"  平均价值: {results['value_estimates']['mean']:.4f} ± {results['value_estimates']['std']:.4f}")
        print(f"  最小值: {results['value_estimates']['min']:.4f}")
        print(f"  最大值: {results['value_estimates']['max']:.4f}")
        print("="*60)
    
    # 10. 保存结果
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        dt_str = result_timestamp if result_timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')
        name_for_file = disp_algo if disp_algo else os.path.basename(model_path).replace('.pth', '')
        result_file = os.path.join(output_dir, f"fqe_{name_for_file}_{dt_str}.json")
        
        serializable_results = convert_to_serializable(results)
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n结果已保存到: {result_file}")
    
    return results


def run_fqe_batch(
    target_algorithms=None,
    test_buffer_path="marl/data_benchmark/episodes/test_buffer.pt",
    config_path="paper/benchmark/flows/config.yaml",
    device_id=0,
    output_dir="paper/benchmark/fqe_results",
    verbose=True,
):
    """
    批处理：对给定算法列表自动查找最新模型，按算法配置 FQE 参数，一次性跑完并分算法保存结果。
    CDT 不支持 FQE，会自动跳过。
    纯粹启发式算法不需要模型文件。
    """
    target_algorithms = target_algorithms or TARGET_ALGORITHMS
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    print(f"FQE 批处理: 算法列表 {target_algorithms}, 结果目录 {output_dir}, 时间戳 {dt_str}")

    collected = []
    for algo_name in target_algorithms:
        # check是否是纯粹启发式算法
        is_pure_heuristic = algo_name in PURE_HEURISTIC_ALGORITHMS
        
        if is_pure_heuristic:
            # 纯粹启发式算法不需要模型文件
            model_path = None
            algo_type = "heuristic"
            print(f"\n>>> 评估纯粹启发式算法 {algo_name} (无需模型)")
        else:
            # 普通算法需要查找模型文件
            model_path = find_latest_model(MODEL_DIR, algo_name)
            if not model_path:
                print(f"跳过 {algo_name}: 未找到模型")
                continue
            algo_type = get_algorithm_type(model_path)
            if algo_type == "cdt":
                print(f"跳过 {algo_name}: CDT 不支持 FQE 评估")
                continue

        params = get_fqe_params_for_algo(algo_name)
        if not is_pure_heuristic:
            print(f"\n>>> 评估 {algo_name} (模型: {os.path.basename(model_path)})")
        
        res = evaluate_fqe(
            model_path=model_path,  # 纯粹启发式算法传入None
            test_buffer_path=test_buffer_path,
            config_path=config_path,
            device_id=device_id,
            fqe_epochs=params["fqe_epochs"],
            fqe_batch_size=params["fqe_batch_size"],
            gamma=params["gamma"],
            fqe_lr=params["fqe_lr"],
            algorithm_type=algo_type,
            algorithm_name=algo_name,
            result_timestamp=dt_str,
            verbose=verbose,
            save_results=True,
            output_dir=output_dir,
        )
        if res is not None:
            collected.append(res)
            print(f"完成 {algo_name}: 平均价值 {res['value_estimates']['mean']:.4f}")

    print(f"\n批处理结束: 共评估 {len(collected)} 个算法, 结果已按算法保存到 {output_dir}")
    return collected


def convert_to_serializable(obj):
    """Convert numpy-containing objects to JSON-serializable"""
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FQE评估脚本 v2（单模型或批处理）')
    parser.add_argument('--batch', action='store_true', help='批处理模式：按算法列表自动找最新模型、跑FQE、分算法保存')
    parser.add_argument('--target_algorithms', type=str, nargs='+', default=None, help='批处理时的算法列表，覆盖 TARGET_ALGORITHMS')
    parser.add_argument('--model_path', type=str, default=None, help='单模型模式：训练好的模型路径（与 --batch 二选一）')
    parser.add_argument('--test_buffer', type=str, default='marl/data_benchmark/episodes/test_buffer.pt', help='测试buffer路径')
    parser.add_argument('--config', type=str, default='paper/benchmark/flows/config.yaml', help='config file path')
    parser.add_argument('--device_id', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--fqe_epochs', type=int, default=50, help='单模型模式: FQE训练轮数')
    parser.add_argument('--fqe_batch_size', type=int, default=256, help='单模型模式: FQE批次大小')
    parser.add_argument('--gamma', type=float, default=0.99, help='单模型模式: 折扣因子')
    parser.add_argument('--fqe_lr', type=float, default=5e-5, help='单模型模式: FQE学习率')
    parser.add_argument('--output_dir', type=str, default='paper/benchmark/fqe_results', help='output directory')
    args = parser.parse_args()

    if args.batch:
        run_fqe_batch(
            target_algorithms=args.target_algorithms,
            test_buffer_path=args.test_buffer,
            config_path=args.config,
            device_id=args.device_id,
            output_dir=args.output_dir,
            verbose=True,
        )
    else:
        if not args.model_path:
            parser.error('单模型模式需提供 --model_path；批处理请使用 --batch')
        evaluate_fqe(
            model_path=args.model_path,
            test_buffer_path=args.test_buffer,
            config_path=args.config,
            device_id=args.device_id,
            fqe_epochs=args.fqe_epochs,
            fqe_batch_size=args.fqe_batch_size,
            gamma=args.gamma,
            fqe_lr=args.fqe_lr,
            verbose=True,
            save_results=True,
            output_dir=args.output_dir,
        )


'''
# 批处理（用默认或 --target_algorithms 列表）
python paper/benchmark/flows/evaluate_fqe_v3.py --batch \
  --target_algorithms onestep cql multitask_offline_cpq \
  --test_buffer marl/data_benchmark/episodes/test_buffer.pt \
  --config paper/benchmark/flows/config.yaml --device_id 0 \
  --output_dir paper/benchmark/fqe_results

# 单模型
python paper/benchmark/flows/evaluate_fqe_v3.py \
  --model_path paper/benchmark/saved_models/onestep_model_20260122_113606.pth \
  --test_buffer marl/data_benchmark/episodes/test_buffer.pt \
  --config paper/benchmark/flows/config.yaml --device_id 0
'''