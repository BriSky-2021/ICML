# evaluate_unified.py
import os
import sys
import numpy as np
import torch as th

# Ensure utils can be imported (when imported from flows or elsewhere)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARK_ROOT)

from utils.transition_util import build_transition_matrices
from collections import defaultdict

# FIXME
# 1. Add hard constraints and evaluate with both constraint sets
# 2. Add more metrics when plotting

# TODO
# Investigate multi-agent budget allocation algorithms


class ModelWrapper:
    """Unified model wrapper; ensures correct eval mode handling."""

    def __init__(self, model,algorithm_type='marl'):
        self.model = model
        self.original_training_states = {}
        self.algorithm_type = algorithm_type
        self._set_eval_mode()

    def _set_eval_mode(self):
        """Set evaluation mode."""
        if hasattr(self.model, 'eval'):
            self.original_training_states['model'] = self.model.training
            print(f"Debug: get into eval mode")
            self.model.eval()

        network_attrs = ['q_network', 'target_q_network', 'policy_network', 'mixer', 'target_mixer',
                        'actor', 'critic', 'encoder', 'decoder', 'state_encoder']

        for attr_name in network_attrs:
            if hasattr(self.model, attr_name):
                network = getattr(self.model, attr_name)
                if hasattr(network, 'eval'):
                    self.original_training_states[attr_name] = network.training
                    network.eval()

        if hasattr(self.model, 'networks'):
            for i, network in enumerate(self.model.networks):
                if hasattr(network, 'eval'):
                    self.original_training_states[f'networks_{i}'] = network.training
                    network.eval()

        if hasattr(self.model, 'agents') and isinstance(self.model.agents, dict):
            for agent_id, agent in self.model.agents.items():
                if hasattr(agent, 'eval'):
                    self.original_training_states[f'agent_{agent_id}'] = agent.training
                    agent.eval()

    def _restore_training_mode(self):
        """Restore original training mode."""
        if hasattr(self.model, 'train') and 'model' in self.original_training_states:
            print(f"Debug: restore training mode")
            if self.original_training_states['model']:
                self.model.train()
            else:
                self.model.eval()

        network_attrs = ['q_network', 'target_q_network', 'policy_network', 'mixer', 'target_mixer',
                        'actor', 'critic', 'encoder', 'decoder', 'state_encoder']

        for attr_name in network_attrs:
            if hasattr(self.model, attr_name) and attr_name in self.original_training_states:
                network = getattr(self.model, attr_name)
                if hasattr(network, 'train'):
                    if self.original_training_states[attr_name]:
                        network.train()
                    else:
                        network.eval()
                        
        # Restore networks list
        if hasattr(self.model, 'networks'):
            for i, network in enumerate(self.model.networks):
                key = f'networks_{i}'
                if hasattr(network, 'train') and key in self.original_training_states:
                    if self.original_training_states[key]:
                        network.train()
                    else:
                        network.eval()
                        
        # Restore agents dict
        if hasattr(self.model, 'agents') and isinstance(self.model.agents, dict):
            for agent_id, agent in self.model.agents.items():
                key = f'agent_{agent_id}'
                if hasattr(agent, 'train') and key in self.original_training_states:
                    if self.original_training_states[key]:
                        agent.train()
                    else:
                        agent.eval()
    
    def act(self, *args, **kwargs):
        """Unified action interface."""
        with th.no_grad():
            return self.model.act(*args, **kwargs)
            
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit; restore training mode."""
        self._restore_training_mode()

def apply_hard_constraint(action, budget, action_costs, log_action_costs=None):
    """
    Apply hard constraint: if action cost exceeds budget, choose next action by cost order.

    Args:
        action: Original predicted action
        budget: Current budget (log scale)
        action_costs: Raw action cost dict
        log_action_costs: Log-scale action cost dict (if None, use action_costs)

    Returns:
        constrained_action: Action after constraint
    """
    if log_action_costs is None:
        log_action_costs = action_costs

    log_cost = float(log_action_costs.get(action, 0))
    if log_cost <= budget:
        return action

    action_cost_pairs = [(act, cost) for act, cost in log_action_costs.items()]
    action_cost_pairs.sort(key=lambda x: x[1])

    for act, cost in action_cost_pairs:
        if cost <= budget:
            return act

    return action_cost_pairs[0][0] if action_cost_pairs else action


def project_health_to_categories(health_value):
    """
    Map health value (0-8) to 4 categories (0-3). Same as data_processor.py:
    - 0-2: critical -> 0
    - 3-4: poor -> 1
    - 5-6: fair -> 2
    - 7-8: good -> 3

    Args:
        health_value: Raw health (0-8)

    Returns:
        category: Health category (0-3)
    """
    if health_value <= 2:
        return 0
    elif health_value <= 4:
        return 1
    elif health_value <= 6:
        return 2
    else:
        return 3


def calculate_expected_health_improvement(current_health, action, transition_matrices):
    """
    Compute expected health improvement from transition matrices.

    Args:
        current_health: Current health (0-8)
        action: Action
        transition_matrices: Dict of transition matrices

    Returns:
        expected_improvement: Expected health improvement (continuous)
    """
    cur_h_projected = project_health_to_categories(current_health)
    mat = transition_matrices[action]
    idx = np.clip(cur_h_projected, 0, mat.shape[0] - 1)
    probs = mat[idx]
    expected_next_state = sum(state * prob for state, prob in enumerate(probs))
    return expected_next_state - cur_h_projected


def calculate_comprehensive_score(metrics, mean_history_cost=1157744):
    """
    Compute comprehensive score.

    Args:
        metrics: Eval metrics dict
        mean_history_cost: Historical mean cost

    Returns:
        comprehensive_score: Composite score
    """
    cost_threshold = mean_history_cost * 1.2
    current_cost = metrics.get('mean_total_cost', float('inf'))
    if current_cost > cost_threshold:
        return 0.0
    behavioral_similarity = metrics.get('behavioral_similarity_mean', 0.0)
    health_gain_efficiency = min(1.0, metrics.get('health_gain_per_1000_dollars', 0.0) / 10.0)
    budget_usage = metrics.get('budget_usage_ratio', 1.0)
    budget_efficiency = max(0.0, 1.0 - budget_usage)
    health_improvement = min(1.0, max(0.0, (metrics.get('bridge_health_gain_absolute', 0.0) + 0.5) / 1.0))

    # Weighted composite score
    weights = {
        'behavioral_similarity': 0,
        'health_gain_efficiency': 0,
        'budget_efficiency': 0.3,
        'health_improvement': 0.7
    }
    
    score = (weights['behavioral_similarity'] * behavioral_similarity +
             weights['health_gain_efficiency'] * health_gain_efficiency +
             weights['budget_efficiency'] * budget_efficiency +
             weights['health_improvement'] * health_improvement)
    
    return float(score)


def evaluate_unified(
    model,
    data,
    actions,
    budgets=None,
    health=None,
    action_costs=None,
    raw_cost=None,
    verbose=True,
    use_agent_mask=True,
    budget_mode='uniform',
    actual_n_agents=None,
    # CDT-specific
    is_cdt_model=False,
    returns_to_go=None,
    costs_to_go=None,
    time_steps=None,
    episode_idx_arr=None,
    agent_idx_arr=None,
    max_bridges=None,
    log_budget_norm_params=None,
    test_episode_budgets=None,
    apply_hard_constraints=True,
    mean_history_cost=1494509,
    device='cuda',
    algorithm_type='marl'
):
    """
    Unified evaluation for all algorithm types.

    Args:
        model: Model
        data: [num_eps, T, n_agents, state_dim]
        actions: [num_eps, T, n_agents]
        budgets: Budget info
        health: [num_eps, T+1, n_agents]
        action_costs: Action cost dict
        raw_cost: [num_eps, T, n_agents]
        verbose: Whether to print details
        use_agent_mask: Use agent mask
        budget_mode: Budget allocation mode
        is_cdt_model, returns_to_go, costs_to_go, time_steps, episode_idx_arr, agent_idx_arr, max_bridges: CDT-specific
        apply_hard_constraints: Apply hard constraints
        mean_history_cost: Historical mean cost (for budget usage and scoring)

    Returns:
        metrics: Eval metrics dict (soft and hard constraint results)
    """
    

    with ModelWrapper(model,algorithm_type=algorithm_type) as wrapper_model:
        if is_cdt_model:
            return _evaluate_cdt_unified(
                model=wrapper_model, data=data, actions=actions, budgets=budgets, health=health, 
                action_costs=action_costs, raw_cost=raw_cost,
                returns_to_go=returns_to_go, costs_to_go=costs_to_go, time_steps=time_steps, 
                episode_idx_arr=episode_idx_arr, agent_idx_arr=agent_idx_arr, 
                max_bridges=max_bridges, verbose=verbose, actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params, 
                test_episode_budgets=test_episode_budgets,
                apply_hard_constraints=apply_hard_constraints,
                mean_history_cost=mean_history_cost,
                device=device
            )
        else:
            return _evaluate_standard_unified(
                wrapper_model, data, actions, budgets, health, action_costs, raw_cost,
                use_agent_mask, budget_mode, actual_n_agents, verbose,
                log_budget_norm_params, test_episode_budgets,
                apply_hard_constraints=apply_hard_constraints,
                mean_history_cost=mean_history_cost
            )


def _evaluate_standard_unified(
    model, data, actions, budgets, health, action_costs, raw_cost, 
    use_agent_mask, budget_mode, actual_n_agents, verbose,
    log_budget_norm_params=None,  # added：log预算变换参数
    test_episode_budgets=None,    # added：测试时的总原始预算值
    apply_hard_constraints=False, # added：是否应用硬约束
    mean_history_cost=1157744     # historical mean cost
):
    """
    标准统一评估函数（非CDT算法）
    - 不再推进健康轨迹
    - 仅使用数据集原始健康状态作为基底
    - 转移矩阵仅用于评估"每一步"的健康改善，并与历史下一年健康对比
    - 支持硬约束：当动作超过预算时选择下一个经费值顺位的动作
    - 同时进行软约束和硬约束两套评估
    - 新增桥梁级别的详细指标
    """
    data = np.asarray(data)
    actions = np.asarray(actions)
    num_eps, T, n_agents, state_dim = data.shape

    if action_costs is None:
        action_costs = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}

    # will convert action_costs to log format (for comparison)
    if log_budget_norm_params is not None:
        mean = log_budget_norm_params['mean']
        std = log_budget_norm_params['std']
        # will convert original action_costs to log format
        log_action_costs = {}
        for action, cost in action_costs.items():
            log_cost = np.log(cost + 1e-8) / 10  # and log_budgets use the same transformation
            log_action_costs[action] = log_cost
    else:
        log_action_costs = action_costs

    # budget processing - use log format uniformly
    if budget_mode == 'uniform':
        if budgets.ndim == 4:
            budget_seq = np.zeros((T, n_agents))
            for ep in range(num_eps):
                ep_budgets = budgets[ep]              # (T, n_agents, 1)
                btotal = float(ep_budgets.sum())
                per_step_agent_budget = btotal / (T * n_agents)
                # convert to log format
                if log_budget_norm_params is not None:
                    log_budget = np.log(per_step_agent_budget + 1e-8) / 10
                    budget_seq[:, :] = log_budget
                else:
                    budget_seq[:, :] = per_step_agent_budget
        elif budgets.ndim == 1:
            budget_seq = np.zeros((T, n_agents))
            for ep in range(num_eps):
                btotal = float(budgets[ep])
                per_step_agent_budget = btotal / (T * n_agents)
                # convert to log format
                if log_budget_norm_params is not None:
                    log_budget = np.log(per_step_agent_budget + 1e-8) / 10
                    budget_seq[:, :] = log_budget
                else:
                    budget_seq[:, :] = per_step_agent_budget
        else:
            raise ValueError(f"Unsupported budgets shape: {budgets.shape}")
    elif budget_mode == 'provided':
        if budgets.ndim == 4:
            budget_seq = budgets[0, :, :, 0]
        elif budgets.ndim == 3:
            budget_seq = budgets[0]
        else:
            raise ValueError(f"Unsupported budgets shape: {budgets.shape}")
    else:
        raise ValueError("budget_mode 必须是 'uniform' | 'provided'")

    # detect model type
    is_marl_model = hasattr(model, 'algorithm_type') and model.algorithm_type == 'marl'

    # 转移矩阵（仅用于单步改善评估）
    transition_matrices = build_transition_matrices()

    # 软约束评估指标 - 全局累积
    all_correct, all_total_cost, all_violation = [], [], []
    all_budget_utilization = []
    all_action_counts = []
    total_evaluations_num = 0
    
    # 桥梁级别指标 - 全局累积
    bridge_costs = []  # 每座桥的成本
    bridge_health_gains_pred = []  # 每座桥的预测健康改善
    bridge_health_gains_hist = []  # 每座桥的历史健康改善
    bridge_health_gains_vs_nothing = []  # 每座桥相对于什么都不做的健康改善
    bridge_initial_health = []  # 每座桥的初始健康状态

    # hard constraint评估指标（如果启用）
    all_correct_r, all_total_cost_r, all_violation_r = [], [], []
    all_budget_utilization_r = []
    all_action_counts_r = []
    total_evaluations_num_r = 0
    
    # hard constraint桥梁级别指标
    bridge_costs_r = []
    bridge_health_gains_pred_r = []
    bridge_health_gains_hist_r = []
    bridge_health_gains_vs_nothing_r = []

    for ep in range(num_eps):
        obs_seq = data[ep]                      # [T, n_agents, obs_dim]
        act_seq = actions[ep]                   # [T, n_agents]
        ep_raw_cost = raw_cost[ep] if raw_cost is not None else None
        ep_health = health[ep] if health is not None else None  # [T+1, n_agents] or None

        # 激活agent集合
        if use_agent_mask:
            if actual_n_agents is not None:
                ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
                active_idx = list(range(ep_actual_n))
            else:
                agent_mask = np.any(obs_seq.sum(axis=2) != 0, axis=0).astype(np.float32)
                active_idx = np.where(agent_mask == 1)[0].tolist()
            if not active_idx:
                continue
        else:
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))

        # 为MARL模型构建完整的agent_mask
        if is_marl_model:
            full_agent_mask = np.zeros(n_agents, dtype=np.float32)
            full_agent_mask[active_idx] = 1.0

        # 软约束评估
        correct = 0
        total_cost = 0.0
        violation = 0
        pred_actions = np.zeros((T, n_agents), dtype=int)
        action_counts = {}

        # hard constraint评估（如果启用）
        if apply_hard_constraints:
            correct_r = 0
            total_cost_r = 0.0
            violation_r = 0
            pred_actions_r = np.zeros((T, n_agents), dtype=int)
            action_counts_r = {}

        # 逐时间步决策与统计
        for t in range(T):

            if is_marl_model:
                # ==== MARL模型：批量处理所有智能体 ====
                obs_t = obs_seq[t]  # [n_agents, obs_dim]
                obs_batch = obs_t.reshape(1, n_agents, -1)  # [1, n_agents, obs_dim]
                
                # 构建legal_actions（如果模型需要）
                legal_actions_t = None
                if hasattr(model, 'action_dim'):
                    legal_actions_t = np.ones((1, n_agents, model.action_dim), dtype=bool)
                    can根据预算约束来设置legal_actions
                    for a in range(n_agents):
                        if a < len(active_idx):
                            bud_t_a = float(budget_seq[t, a])
                            for action_id, log_cost in log_action_costs.items():
                                if log_cost > bud_t_a:
                                    legal_actions_t[0, a, action_id] = False

                #try:
                # 调用MARL模型
                pred_actions = model.act(
                    obs=obs_batch,
                    agent_mask=full_agent_mask.reshape(1, -1),  # [1, n_agents]
                    legal_actions=None,#legal_actions_t,
                    training=False
                )
                

                #print(f"[DEBUG] 原始预测输出: {pred_actions}")
                #print(f"[DEBUG] 预测输出类型: {type(pred_actions)}")

                
                if isinstance(pred_actions, (tuple, list)):
                    pred_actions = pred_actions[0]
                    #print(f"[DEBUG] 提取后预测输出: {pred_actions}")
                
                pred_actions = np.asarray(pred_actions)
                #print(f"[DEBUG] 转换为numpy后形状: {pred_actions.shape}")
                #print(f"[DEBUG] 预测动作内容: {pred_actions}")
                if pred_actions.ndim == 2:
                    pred_actions = pred_actions[0]  # [n_agents]
                    #print(f"[DEBUG] 降维后预测动作: {pred_actions}")
                
                # if有负值变为0值
                pred_actions[pred_actions < 0] = 0

                #action_dist = np.bincount(pred_actions[:len(active_idx)], minlength=4)
                #print(f"[DEBUG] 当前时间步动作分布: {action_dist}")
                
                #exit(0)
                #except Exception as e:
                    #if verbose:
                    #    print(f"MARL模型调用失败: {e}")
                    # use随机动作作为fallback
                    #pred_actions = np.random.choice(len(action_costs), size=n_agents)

                # handle预测结果
                for i, a in enumerate(active_idx):
                    act = int(pred_actions[a]) if a < len(pred_actions) else 0
                    bud_t_a = float(budget_seq[t, a])
                    
                    # 软约束评估
                    action_counts[act] = action_counts.get(act, 0) + 1
                    
                    if act == int(act_seq[t, a]):
                        correct += 1
                    
                    cost = float(action_costs.get(act, 0))
                    total_cost += cost
                    
                    log_cost = float(log_action_costs.get(act, 0))
                    if log_cost > bud_t_a:
                        violation += 1

                    # hard constraint评估
                    if apply_hard_constraints:
                        act_r = apply_hard_constraint(act, bud_t_a, action_costs, log_action_costs)
                        action_counts_r[act_r] = action_counts_r.get(act_r, 0) + 1
                        
                        if act_r == int(act_seq[t, a]):
                            correct_r += 1
                        
                        cost_r = float(action_costs.get(act_r, 0))
                        total_cost_r += cost_r
                        
                        log_cost_r = float(log_action_costs.get(act_r, 0))
                        if log_cost_r > bud_t_a:
                            violation_r += 1

                    # 桥梁级别健康评估
                    if ep_health is not None and t + 1 < ep_health.shape[0]:
                        cur_h = int(ep_health[t, a])
                        
                        bridge_costs.append(cost)
                        bridge_initial_health.append(cur_h)
                        
                        pred_health_gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                        bridge_health_gains_pred.append(pred_health_gain)
                        
                        true_action = int(act_seq[t, a])
                        hist_health_gain = calculate_expected_health_improvement(cur_h, true_action, transition_matrices)
                        bridge_health_gains_hist.append(hist_health_gain)
                        
                        nothing_health_gain = calculate_expected_health_improvement(cur_h, 0, transition_matrices)
                        vs_nothing_gain = pred_health_gain - nothing_health_gain
                        bridge_health_gains_vs_nothing.append(vs_nothing_gain)

                        if apply_hard_constraints:
                            bridge_costs_r.append(cost_r)
                            pred_health_gain_r = calculate_expected_health_improvement(cur_h, act_r, transition_matrices)
                            bridge_health_gains_pred_r.append(pred_health_gain_r)
                            bridge_health_gains_hist_r.append(hist_health_gain)
                            vs_nothing_gain_r = pred_health_gain_r - nothing_health_gain
                            bridge_health_gains_vs_nothing_r.append(vs_nothing_gain_r)

            else:

                for a in active_idx:
                    obs_t_a = obs_seq[t, a]
                    bud_t_a = float(budget_seq[t, a])  # log格式的预算

                    # 调用模型（接口自适应）
                    try:
                        obs_in = obs_t_a.reshape(1, -1)
                        pred = model.act(obs_in, bud_t_a)  # 传入log格式预算
                    except TypeError:
                        try:
                            obs_in = obs_t_a.reshape(1, 1, -1)
                            mask_in = np.array([[1.0]], dtype=np.float32)
                            pred = model.act(obs_in, mask_in)
                        except TypeError:
                            obs_in = obs_t_a.reshape(1, -1)
                            pred = model.act(obs_in)
                            

                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    pred = np.asarray(pred)
                    act = int(pred[0, 0]) if pred.ndim >= 2 else int(pred)
                    
                    # 软约束评估
                    pred_actions[t, a] = act
                    action_counts[act] = action_counts.get(act, 0) + 1

                    if act == int(act_seq[t, a]):
                        correct += 1
                    
                    cost = float(action_costs.get(act, 0))
                    total_cost += cost

                    log_cost = float(log_action_costs.get(act, 0))
                    if log_cost > bud_t_a:
                        violation += 1

                    # hard constraint评估（如果启用）
                    if apply_hard_constraints:
                        act_r = apply_hard_constraint(act, bud_t_a, action_costs, log_action_costs)
                        pred_actions_r[t, a] = act_r
                        action_counts_r[act_r] = action_counts_r.get(act_r, 0) + 1

                        if act_r == int(act_seq[t, a]):
                            correct_r += 1
                        
                        cost_r = float(action_costs.get(act_r, 0))
                        total_cost_r += cost_r

                        log_cost_r = float(log_action_costs.get(act_r, 0))
                        if log_cost_r > bud_t_a:
                            violation_r += 1

                    # 桥梁级别健康改善评估（软约束）
                    if ep_health is not None and t + 1 < ep_health.shape[0]:
                        cur_h = int(ep_health[t, a])
                        
                        # record桥梁成本和初始健康状态
                        bridge_costs.append(cost)
                        bridge_initial_health.append(cur_h)
                        
                        # predicted健康改善（使用expected值而非随机采样）
                        pred_health_gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                        bridge_health_gains_pred.append(pred_health_gain)
                        
                        # 历史健康改善
                        true_action = int(act_seq[t, a])
                        hist_health_gain = calculate_expected_health_improvement(cur_h, true_action, transition_matrices)
                        bridge_health_gains_hist.append(hist_health_gain)
                        
                        # 相对于什么都不做的健康改善
                        nothing_health_gain = calculate_expected_health_improvement(cur_h, 0, transition_matrices)  # 假设动作0是什么都不做
                        vs_nothing_gain = pred_health_gain - nothing_health_gain
                        bridge_health_gains_vs_nothing.append(vs_nothing_gain)

                        # hard constraint桥梁级别评估（如果启用）
                        if apply_hard_constraints:
                            bridge_costs_r.append(cost_r)
                            
                            pred_health_gain_r = calculate_expected_health_improvement(cur_h, act_r, transition_matrices)
                            bridge_health_gains_pred_r.append(pred_health_gain_r)
                            bridge_health_gains_hist_r.append(hist_health_gain)  # 历史保持不变
                            
                            vs_nothing_gain_r = pred_health_gain_r - nothing_health_gain
                            bridge_health_gains_vs_nothing_r.append(vs_nothing_gain_r)

                total_evaluations_num += 1
                if apply_hard_constraints:
                    total_evaluations_num_r += 1

        # 参与评估的实际数量
        total_evaluations = len(active_idx) * T
        acc = (correct / total_evaluations) if total_evaluations > 0 else 0.0
        vio_rate = (violation / total_evaluations) if total_evaluations > 0 else 0.0

        all_correct.append(acc)
        all_total_cost.append(total_cost)
        all_violation.append(vio_rate)

        if apply_hard_constraints:
            acc_r = (correct_r / total_evaluations) if total_evaluations > 0 else 0.0
            vio_rate_r = (violation_r / total_evaluations) if total_evaluations > 0 else 0.0

            all_correct_r.append(acc_r)
            all_total_cost_r.append(total_cost_r)
            all_violation_r.append(vio_rate_r)

        # compute预算使用占比（修改计算方式）
        budget_utilization = total_cost / mean_history_cost
        all_budget_utilization.append(budget_utilization)

        if apply_hard_constraints:
            budget_utilization_r = total_cost_r / mean_history_cost
            all_budget_utilization_r.append(budget_utilization_r)

        # 存储动作计数
        all_action_counts.append(action_counts)
        if apply_hard_constraints:
            all_action_counts_r.append(action_counts_r)

    # compute各动作比例（软约束）
    action_proportions = {}
    if all_action_counts:
        total_action_counts = {}
        for action_counts in all_action_counts:
            for action, count in action_counts.items():
                total_action_counts[action] = total_action_counts.get(action, 0) + count
        
        total_actions = sum(total_action_counts.values())
        
        for action in sorted(total_action_counts.keys()):
            proportion = total_action_counts[action] / total_actions if total_actions > 0 else 0.0
            action_proportions[f'action_{action}_proportion'] = float(proportion)
    
    # compute各动作比例（硬约束）
    action_proportions_r = {}
    if apply_hard_constraints and all_action_counts_r:
        total_action_counts_r = {}
        for action_counts in all_action_counts_r:
            for action, count in action_counts.items():
                total_action_counts_r[action] = total_action_counts_r.get(action, 0) + count
        
        total_actions_r = sum(total_action_counts_r.values())
        
        for action in sorted(total_action_counts_r.keys()):
            proportion = total_action_counts_r[action] / total_actions_r if total_actions_r > 0 else 0.0
            action_proportions_r[f'action_{action}_proportion_r'] = float(proportion)

    # compute桥梁级别指标（软约束）
    bridge_metrics = {}
    if bridge_costs:
        # 基本统计
        bridge_metrics['bridge_avg_cost'] = float(np.mean(bridge_costs))
        bridge_metrics['bridge_avg_health_gain_absolute'] = float(np.mean(bridge_health_gains_pred))
        bridge_metrics['bridge_avg_health_gain_vs_history'] = float(np.mean(np.array(bridge_health_gains_pred) - np.array(bridge_health_gains_hist)))
        bridge_metrics['bridge_avg_health_gain_vs_nothing'] = float(np.mean(bridge_health_gains_vs_nothing))
        
        # ratio指标（相对于初始健康状态）
        initial_health_array = np.array(bridge_initial_health)
        valid_indices = initial_health_array > 0  # avoid除零
        if np.any(valid_indices):
            health_gains_ratio = np.array(bridge_health_gains_pred)[valid_indices] / initial_health_array[valid_indices]
            bridge_metrics['bridge_avg_health_gain_ratio'] = float(np.mean(health_gains_ratio))
        else:
            bridge_metrics['bridge_avg_health_gain_ratio'] = 0.0
        
        # normalize指标 (0-1范围)
        max_possible_gain = 3.0  # fromcritical(0)到good(3)的最大改善
        bridge_metrics['bridge_avg_health_gain_normalized'] = float(np.mean(bridge_health_gains_pred) / max_possible_gain)
        
        # 效率指标
        costs_array = np.array(bridge_costs)
        valid_cost_indices = costs_array > 0
        if np.any(valid_cost_indices):
            health_gain_per_1000 = (np.array(bridge_health_gains_pred)[valid_cost_indices] * 1000) / costs_array[valid_cost_indices]
            bridge_metrics['health_gain_per_1000_dollars'] = float(np.mean(health_gain_per_1000))
        else:
            bridge_metrics['health_gain_per_1000_dollars'] = 0.0

    # compute桥梁级别指标（硬约束）
    bridge_metrics_r = {}
    if apply_hard_constraints and bridge_costs_r:
        # 基本统计
        bridge_metrics_r['bridge_avg_cost_r'] = float(np.mean(bridge_costs_r))
        bridge_metrics_r['bridge_avg_health_gain_absolute_r'] = float(np.mean(bridge_health_gains_pred_r))
        bridge_metrics_r['bridge_avg_health_gain_vs_history_r'] = float(np.mean(np.array(bridge_health_gains_pred_r) - np.array(bridge_health_gains_hist_r)))
        bridge_metrics_r['bridge_avg_health_gain_vs_nothing_r'] = float(np.mean(bridge_health_gains_vs_nothing_r))
        
        # ratio指标
        initial_health_array = np.array(bridge_initial_health)
        valid_indices = initial_health_array > 0
        if np.any(valid_indices):
            health_gains_ratio_r = np.array(bridge_health_gains_pred_r)[valid_indices] / initial_health_array[valid_indices]
            bridge_metrics_r['bridge_avg_health_gain_ratio_r'] = float(np.mean(health_gains_ratio_r))
        else:
            bridge_metrics_r['bridge_avg_health_gain_ratio_r'] = 0.0
        
        # normalize指标
        bridge_metrics_r['bridge_avg_health_gain_normalized_r'] = float(np.mean(bridge_health_gains_pred_r) / max_possible_gain)
        
        # 效率指标
        costs_array_r = np.array(bridge_costs_r)
        valid_cost_indices_r = costs_array_r > 0
        if np.any(valid_cost_indices_r):
            health_gain_per_1000_r = (np.array(bridge_health_gains_pred_r)[valid_cost_indices_r] * 1000) / costs_array_r[valid_cost_indices_r]
            bridge_metrics_r['health_gain_per_1000_dollars_r'] = float(np.mean(health_gain_per_1000_r))
        else:
            bridge_metrics_r['health_gain_per_1000_dollars_r'] = 0.0

    # 汇总指标（软约束）
    metrics = {
        # 重命名的基础指标
        'behavioral_similarity_mean': float(np.mean(all_correct)) if all_correct else 0.0,
        'behavioral_similarity_std': float(np.std(all_correct)) if all_correct else 0.0,
        'mean_total_cost': float(np.mean(all_total_cost)) if all_total_cost else 0.0,
        'std_total_cost': float(np.std(all_total_cost)) if all_total_cost else 0.0,
        'mean_violation': float(np.mean(all_violation)) if all_violation else 0.0,
        'std_violation': float(np.std(all_violation)) if all_violation else 0.0,
        
        # 修改的预算使用率
        'budget_usage_ratio': float(np.mean(all_budget_utilization)) if all_budget_utilization else 0.0,
        'budget_usage_ratio_std': float(np.std(all_budget_utilization)) if all_budget_utilization else 0.0,
        
        # 违规率
        'violation_rate_mean': float(np.mean(all_violation)) if all_violation else 0.0,
        'violation_rate_std': float(np.std(all_violation)) if all_violation else 0.0,
        
        # 总评估数量
        'total_evaluations': total_evaluations_num,
        
        # 各动作比例
        **action_proportions,
        
        # 桥梁级别指标
        **bridge_metrics
    }

    # hard constraint指标（如果启用）
    if apply_hard_constraints:
        metrics.update({
            # 重命名的基础指标（硬约束）
            'behavioral_similarity_mean_r': float(np.mean(all_correct_r)) if all_correct_r else 0.0,
            'behavioral_similarity_std_r': float(np.std(all_correct_r)) if all_correct_r else 0.0,
            'mean_total_cost_r': float(np.mean(all_total_cost_r)) if all_total_cost_r else 0.0,
            'std_total_cost_r': float(np.std(all_total_cost_r)) if all_total_cost_r else 0.0,
            'mean_violation_r': float(np.mean(all_violation_r)) if all_violation_r else 0.0,
            'std_violation_r': float(np.std(all_violation_r)) if all_violation_r else 0.0,
            
            # 修改的预算使用率（硬约束）
            'budget_usage_ratio_r': float(np.mean(all_budget_utilization_r)) if all_budget_utilization_r else 0.0,
            'budget_usage_ratio_std_r': float(np.std(all_budget_utilization_r)) if all_budget_utilization_r else 0.0,
            
            # 违规率（硬约束）
            'violation_rate_mean_r': float(np.mean(all_violation_r)) if all_violation_r else 0.0,
            'violation_rate_std_r': float(np.std(all_violation_r)) if all_violation_r else 0.0,
            
            # 总评估数量（硬约束）
            'total_evaluations_r': total_evaluations_num_r,
            
            # 各动作比例（硬约束）
            **action_proportions_r,
            
            # 桥梁级别指标（硬约束）
            **bridge_metrics_r
        })

    # add综合评分
    metrics['comprehensive_score'] = calculate_comprehensive_score(metrics, mean_history_cost)
    if apply_hard_constraints:
        # 为硬约束创建临时metrics字典用于评分计算
        temp_metrics_r = {
            'behavioral_similarity_mean': metrics.get('behavioral_similarity_mean_r', 0.0),
            'health_gain_per_1000_dollars': metrics.get('health_gain_per_1000_dollars_r', 0.0),
            'budget_usage_ratio': metrics.get('budget_usage_ratio_r', 0.0),
            'bridge_health_gain_absolute': metrics.get('bridge_avg_health_gain_absolute_r', 0.0),
            'mean_total_cost': metrics.get('mean_total_cost_r', 0.0)
        }
        metrics['comprehensive_score_r'] = calculate_comprehensive_score(temp_metrics_r, mean_history_cost)

    if verbose:
        constraint_type = "Hard Constrained" if apply_hard_constraints else "Unconstrained"
        print(f"评估结果 ({constraint_type}): behavioral_similarity={metrics['behavioral_similarity_mean']:.4f}, "
              f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.4f}")
        
        if apply_hard_constraints:
            print(f"硬约束结果: behavioral_similarity={metrics['behavioral_similarity_mean_r']:.4f}, "
                  f"cost={metrics['mean_total_cost_r']:.1f}, vio={metrics['violation_rate_mean_r']:.4f}")
        
        print(f"预算使用占比 (vs历史): {metrics['budget_usage_ratio']:.4f}")
        
        if apply_hard_constraints:
            print(f"硬约束预算使用占比 (vs历史): {metrics['budget_usage_ratio_r']:.4f}")
        
        # print桥梁级别指标
        if 'bridge_avg_cost' in metrics:
            print(f"桥梁平均成本: ${metrics['bridge_avg_cost']:.2f}")
            print(f"桥梁平均健康改善: {metrics['bridge_avg_health_gain_absolute']:.4f}")
            print(f"桥梁平均相对历史改善: {metrics['bridge_avg_health_gain_vs_history']:.4f}")
            print(f"桥梁平均相对nothing改善: {metrics['bridge_avg_health_gain_vs_nothing']:.4f}")
            print(f"健康改善效率 (per $1000): {metrics['health_gain_per_1000_dollars']:.4f}")
        
        # print综合评分
        print(f"综合评分: {metrics['comprehensive_score']:.4f}")
        if apply_hard_constraints:
            print(f"硬约束综合评分: {metrics['comprehensive_score_r']:.4f}")
        
        # print各动作比例
        print("各动作比例:")
        for action in sorted(action_proportions.keys()):
            print(f"  {action}: {action_proportions[action]:.4f}")
        
        if apply_hard_constraints:
            print("硬约束各动作比例:")
            for action in sorted(action_proportions_r.keys()):
                print(f"  {action}: {action_proportions_r[action]:.4f}")

    return metrics


def _evaluate_cdt_unified(
    model, data, actions, returns_to_go, costs_to_go, time_steps,
    budgets=None, health=None, action_costs=None, raw_cost=None,
    verbose=True, episode_idx_arr=None, agent_idx_arr=None, max_bridges=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None,
    apply_hard_constraints=False,  # added：是否应用硬约束
    mean_history_cost=1157744,      #historical mean cost
    device='cuda'
):
    """
    修改后的CDT统一评估函数，支持预算和成本统一处理
    - 模型接收的预算都是log后的形式
    - action_cost对比时统一为相同格式（log格式）
    - 花费累加时统一采用原始值
    - 评估时输入测试时的总原始预算值，用于计算预算花费比例
    - 支持硬约束：当动作超过预算时选择下一个经费值顺位的动作
    - 同时进行软约束和硬约束两套评估
    - 新增桥梁级别的详细指标
    """
    # ensure所有输入都是numpy数组
    def to_numpy(data):
        if isinstance(data, th.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)
    
    data = to_numpy(data)
    actions = to_numpy(actions)
    returns_to_go = to_numpy(returns_to_go)
    costs_to_go = to_numpy(costs_to_go)
    time_steps = to_numpy(time_steps)

    
    # 现在数据格式是 [num_eps, T, n_agents, state_dim]
    num_eps, T, n_agents, state_dim = data.shape
    
    if verbose:
        print(f"Debug: data.shape = {data.shape}")
        print(f"Debug: returns_to_go.shape = {returns_to_go.shape}")
        print(f"Debug: actual_n_agents = {actual_n_agents}")
    
    # action shape 处理
    if actions.ndim == 4:  # [num_eps, T, n_agents, action_dim]
        action_dim = actions.shape[-1]
        is_onehot = True
    else:  # [num_eps, T, n_agents]
        action_dim = int(np.max(actions)) + 1
        is_onehot = False

    # action index
    if is_onehot:
        act_seq_index = actions.argmax(axis=-1)  # [num_eps, T, n_agents]
    else:
        act_seq_index = actions  # [num_eps, T, n_agents]

    # default成本
    if action_costs is None:
        action_costs = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}
    
    # willaction_costs转换为log格式（用于对比）
    if log_budget_norm_params is not None:
        mean = log_budget_norm_params['mean']
        std = log_budget_norm_params['std']
        # will原始action_costs转换为log格式
        log_action_costs = {}
        for action, cost in action_costs.items():
            log_cost = np.log(cost + 1e-8) / 10  # andlog_budgets使用相同的变换
            log_action_costs[action] = log_cost
    else:
        log_action_costs = action_costs
    
    transition_matrices = build_transition_matrices()

    # raw_cost 处理
    if raw_cost is not None:
        ep_raw_cost = raw_cost
        if ep_raw_cost.ndim > 3:
            ep_raw_cost = ep_raw_cost.squeeze()
    else:
        ep_raw_cost = None

    # budget处理
    if budgets is not None:
        if budgets.ndim == 4:
            budget_seq = budgets[0, :, :, 0]  # [T, n_agents]
        elif budgets.ndim == 3:
            budget_seq = budgets[0]  # [T, n_agents]
        else:
            raise ValueError(f"不支持的budgets形状: {budgets.shape}")
    else:
        # if没有提供预算，使用默认值
        budget_seq = np.zeros((T, n_agents))

    # 软约束评估指标
    all_correct = []
    all_violation = []
    all_total_cost = []
    all_budget_utilization = []
    all_action_counts = []
    total_evaluations_num = 0
    
    # 桥梁级别指标 - 全局累积
    bridge_costs = []
    bridge_health_gains_pred = []
    bridge_health_gains_hist = []
    bridge_health_gains_vs_nothing = []
    bridge_initial_health = []

    # hard constraint评估指标（如果启用）
    all_correct_r = []
    all_violation_r = []
    all_total_cost_r = []
    all_budget_utilization_r = []
    all_action_counts_r = []
    total_evaluations_num_r = 0
    
    # hard constraint桥梁级别指标
    bridge_costs_r = []
    bridge_health_gains_pred_r = []
    bridge_health_gains_hist_r = []
    bridge_health_gains_vs_nothing_r = []

    # byepisode处理
    for ep in range(num_eps):
        obs_seq = data[ep]  # [T, n_agents, obs_dim]
        act_seq = act_seq_index[ep]  # [T, n_agents]
        ret_seq = returns_to_go[ep]  # [T, n_agents]
        cost_seq = costs_to_go[ep]  # [T, n_agents]
        time_seq = time_steps[ep]  # [T, n_agents]
        ep_health = health[ep] if health is not None else None  # [T+1, n_agents] or None

        # 确定活跃的智能体
        if actual_n_agents is not None:
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
        else:
            agent_mask = np.any(obs_seq.sum(axis=2) != 0, axis=0).astype(np.float32)
            active_idx = np.where(agent_mask == 1)[0].tolist()
        
        if not active_idx:
            continue

        # 软约束评估
        correct = 0
        total_cost = 0.0
        violation = 0
        action_counts = {}

        # hard constraint评估（如果启用）
        if apply_hard_constraints:
            correct_r = 0
            total_cost_r = 0.0
            violation_r = 0
            action_counts_r = {}

        # CDT需要累积历史序列
        cdt_history = {
            'states': [],
            'actions': [],
            'returns_to_go': [],
            'costs_to_go': [],
            'time_steps': []
        }
        
        #print(ret_seq.shape)
        #print(cost_seq.shape)
        #print(time_seq.shape)
        #print(budget_seq.shape)
        #print(obs_seq.shape)
        #print(act_seq.shape)
        #print(ep_health.shape)
        #exit(0)
        #(15, 500)
        #(15, 500)
        #(15, 500)
        #(15, 500)
        #(15, 500, 5)
        #(15, 500)
        #(16, 500)

        def safe_get_value(tensor, t=None, a=None):
            """
            安全地从张量中提取值，处理不同维度的情况
            """
            if hasattr(tensor, 'ndim'):
                if tensor.ndim == 0:  # scalar张量
                    return tensor.item()
                elif tensor.ndim == 1 and t is not None:  # [T]
                    return tensor[t].item() if hasattr(tensor[t], 'item') else float(tensor[t])
                elif tensor.ndim == 2 and t is not None and a is not None:  # [T, n_agents]
                    return tensor[t, a].item() if hasattr(tensor[t, a], 'item') else float(tensor[t, a])
                else:
                    return tensor.item() if hasattr(tensor, 'item') else float(tensor)
            else:
                return float(tensor)

        # 逐时间步决策与统计
        for t in range(T):
            for a in active_idx:
                obs_t_a = obs_seq[t] if getattr(obs_seq, 'ndim', 2) == 2 else obs_seq[t, a]

                ret_t_a  = safe_get_value(ret_seq, t, a)
                cost_t_a = safe_get_value(cost_seq, t, a)
                time_t_a = safe_get_value(time_seq, t, a)
                bud_t_a = float(budget_seq[t, a])  # log格式的预算

                # 累积到历史序列
                cdt_history['states'].append(obs_t_a)
                cdt_history['returns_to_go'].append(ret_t_a)
                cdt_history['costs_to_go'].append(cost_t_a)
                cdt_history['time_steps'].append(t)
                
                # build动作序列（历史动作 + 当前零动作）
                if len(cdt_history['actions']) == 0:
                    # first时间步，创建零动作
                    actions_seq = np.zeros((len(cdt_history['states']), action_dim), dtype=np.float32)
                else:
                    # 后续时间步，使用历史动作
                    actions_seq = np.array(cdt_history['actions'])
                    if actions_seq.shape[0] < len(cdt_history['states']):
                        # if历史动作不够，用零动作填充
                        padding = np.zeros((len(cdt_history['states']) - actions_seq.shape[0], action_dim), dtype=np.float32)
                        actions_seq = np.vstack([actions_seq, padding])
                    elif actions_seq.shape[0] > len(cdt_history['states']):
                        # if历史动作太多，截取
                        actions_seq = actions_seq[:len(cdt_history['states'])]
                
                # limit序列长度，避免内存问题
                max_seq_len = 10
                if len(cdt_history['states']) > max_seq_len:
                    # 只保留最近的时间步
                    for key in cdt_history:
                        cdt_history[key] = cdt_history[key][-max_seq_len:]
                    actions_seq = actions_seq[-max_seq_len:]

                # build输入张量
                device = getattr(model, 'device', th.device(device))
                
                states = th.tensor(np.array(cdt_history['states']), dtype=th.float32, device=device)
                _actions = th.tensor(actions_seq, dtype=th.float32, device=device)
                _returns_to_go = th.tensor(np.array(cdt_history['returns_to_go']), dtype=th.float32, device=device)
                _costs_to_go = th.tensor(np.array(cdt_history['costs_to_go']), dtype=th.float32, device=device)
                _time_steps = th.tensor(np.array(cdt_history['time_steps']), dtype=th.long, device=device)

                # 调用CDT模型
                try:
                    with th.no_grad():
                        pred = model.act(
                            states=states,
                            actions=_actions,
                            returns_to_go=_returns_to_go,
                            costs_to_go=_costs_to_go,
                            time_steps=_time_steps,
                            deterministic=True
                        )
                except Exception as e:
                    if verbose:
                        print(f"CDT模型调用失败: {e}")
                    continue

                # handle输出
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                if hasattr(pred, 'detach'):
                    pred = pred.detach()
                if hasattr(pred, 'cpu'):
                    pred = pred.cpu()
                if hasattr(pred, 'numpy'):
                    pred = pred.numpy()

                if isinstance(pred, np.ndarray):
                    if pred.ndim == 2 and pred.shape[0] == 1:
                        act = int(np.argmax(pred[0]))
                    elif pred.ndim == 1:
                        act = int(np.argmax(pred))
                    else:
                        act = int(np.argmax(pred.reshape(-1)))
                else:
                    act = int(pred)

                # will选择的动作添加到历史中
                action_one_hot = np.zeros(action_dim, dtype=np.float32)
                action_one_hot[act] = 1.0
                cdt_history['actions'].append(action_one_hot)

                # 软约束评估
                action_counts[act] = action_counts.get(act, 0) + 1

                if act == int(act_seq[t, a]):
                    correct += 1
                
                cost = float(action_costs.get(act, 0))
                total_cost += cost
                
                log_cost = float(log_action_costs.get(act, 0))
                if log_cost > bud_t_a:
                    violation += 1

                # hard constraint评估（如果启用）
                if apply_hard_constraints:
                    act_r = apply_hard_constraint(act, bud_t_a, action_costs, log_action_costs)
                    action_counts_r[act_r] = action_counts_r.get(act_r, 0) + 1

                    if act_r == int(act_seq[t, a]):
                        correct_r += 1
                    
                    cost_r = float(action_costs.get(act_r, 0))
                    total_cost_r += cost_r
                    
                    log_cost_r = float(log_action_costs.get(act_r, 0))
                    if log_cost_r > bud_t_a:
                        violation_r += 1

                # 桥梁级别健康改善评估（软约束）
                if ep_health is not None and t + 1 < ep_health.shape[0]:
                    cur_h = int(ep_health[t, a])
                    
                    # record桥梁成本和初始健康状态
                    bridge_costs.append(cost)
                    bridge_initial_health.append(cur_h)
                    
                    # predicted健康改善（使用expected值而非随机采样）
                    pred_health_gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                    bridge_health_gains_pred.append(pred_health_gain)
                    
                    # 历史健康改善
                    true_action = int(act_seq[t, a])
                    hist_health_gain = calculate_expected_health_improvement(cur_h, true_action, transition_matrices)
                    bridge_health_gains_hist.append(hist_health_gain)
                    
                    # 相对于什么都不做的健康改善
                    nothing_health_gain = calculate_expected_health_improvement(cur_h, 0, transition_matrices)  # 假设动作0是什么都不做
                    vs_nothing_gain = pred_health_gain - nothing_health_gain
                    bridge_health_gains_vs_nothing.append(vs_nothing_gain)

                    # hard constraint桥梁级别评估（如果启用）
                    if apply_hard_constraints:
                        bridge_costs_r.append(cost_r)
                        
                        pred_health_gain_r = calculate_expected_health_improvement(cur_h, act_r, transition_matrices)
                        bridge_health_gains_pred_r.append(pred_health_gain_r)
                        bridge_health_gains_hist_r.append(hist_health_gain)  # 历史保持不变
                        
                        vs_nothing_gain_r = pred_health_gain_r - nothing_health_gain
                        bridge_health_gains_vs_nothing_r.append(vs_nothing_gain_r)
                
                total_evaluations_num += 1
                if apply_hard_constraints:
                    total_evaluations_num_r += 1

        # 参与评估的实际数量
        total_evaluations = len(active_idx) * T
        acc = (correct / total_evaluations) if total_evaluations > 0 else 0.0
        vio_rate = (violation / total_evaluations) if total_evaluations > 0 else 0.0

        all_correct.append(acc)
        all_total_cost.append(total_cost)
        all_violation.append(vio_rate)

        if apply_hard_constraints:
            acc_r = (correct_r / total_evaluations) if total_evaluations > 0 else 0.0
            vio_rate_r = (violation_r / total_evaluations) if total_evaluations > 0 else 0.0

            all_correct_r.append(acc_r)
            all_total_cost_r.append(total_cost_r)
            all_violation_r.append(vio_rate_r)

        # compute预算使用占比（修改计算方式）
        budget_utilization = total_cost / mean_history_cost
        all_budget_utilization.append(budget_utilization)

        if apply_hard_constraints:
            budget_utilization_r = total_cost_r / mean_history_cost
            all_budget_utilization_r.append(budget_utilization_r)

        # 存储动作计数
        all_action_counts.append(action_counts)
        if apply_hard_constraints:
            all_action_counts_r.append(action_counts_r)

    # compute各动作比例（软约束）
    action_proportions = {}
    if all_action_counts:
        total_action_counts = {}
        for action_counts in all_action_counts:
            for action, count in action_counts.items():
                total_action_counts[action] = total_action_counts.get(action, 0) + count
        
        total_actions = sum(total_action_counts.values())
        
        for action in sorted(total_action_counts.keys()):
            proportion = total_action_counts[action] / total_actions if total_actions > 0 else 0.0
            action_proportions[f'action_{action}_proportion'] = float(proportion)

    # compute各动作比例（硬约束）
    action_proportions_r = {}
    if apply_hard_constraints and all_action_counts_r:
        total_action_counts_r = {}
        for action_counts in all_action_counts_r:
            for action, count in action_counts.items():
                total_action_counts_r[action] = total_action_counts_r.get(action, 0) + count
        
        total_actions_r = sum(total_action_counts_r.values())
        
        for action in sorted(total_action_counts_r.keys()):
            proportion = total_action_counts_r[action] / total_actions_r if total_actions_r > 0 else 0.0
            action_proportions_r[f'action_{action}_proportion_r'] = float(proportion)

    # compute桥梁级别指标（软约束）
    bridge_metrics = {}
    if bridge_costs:
        # 基本统计
        bridge_metrics['bridge_avg_cost'] = float(np.mean(bridge_costs))
        bridge_metrics['bridge_avg_health_gain_absolute'] = float(np.mean(bridge_health_gains_pred))
        bridge_metrics['bridge_avg_health_gain_vs_history'] = float(np.mean(np.array(bridge_health_gains_pred) - np.array(bridge_health_gains_hist)))
        bridge_metrics['bridge_avg_health_gain_vs_nothing'] = float(np.mean(bridge_health_gains_vs_nothing))
        
        # ratio指标（相对于初始健康状态）
        initial_health_array = np.array(bridge_initial_health)
        valid_indices = initial_health_array > 0  # avoid除零
        if np.any(valid_indices):
            health_gains_ratio = np.array(bridge_health_gains_pred)[valid_indices] / initial_health_array[valid_indices]
            bridge_metrics['bridge_avg_health_gain_ratio'] = float(np.mean(health_gains_ratio))
        else:
            bridge_metrics['bridge_avg_health_gain_ratio'] = 0.0
        
        # normalize指标 (0-1范围)
        max_possible_gain = 3.0  # fromcritical(0)到good(3)的最大改善
        bridge_metrics['bridge_avg_health_gain_normalized'] = float(np.mean(bridge_health_gains_pred) / max_possible_gain)
        
        # 效率指标
        costs_array = np.array(bridge_costs)
        valid_cost_indices = costs_array > 0
        if np.any(valid_cost_indices):
            health_gain_per_1000 = (np.array(bridge_health_gains_pred)[valid_cost_indices] * 1000) / costs_array[valid_cost_indices]
            bridge_metrics['health_gain_per_1000_dollars'] = float(np.mean(health_gain_per_1000))
        else:
            bridge_metrics['health_gain_per_1000_dollars'] = 0.0

    # compute桥梁级别指标（硬约束）
    bridge_metrics_r = {}
    if apply_hard_constraints and bridge_costs_r:
        # 基本统计
        bridge_metrics_r['bridge_avg_cost_r'] = float(np.mean(bridge_costs_r))
        bridge_metrics_r['bridge_avg_health_gain_absolute_r'] = float(np.mean(bridge_health_gains_pred_r))
        bridge_metrics_r['bridge_avg_health_gain_vs_history_r'] = float(np.mean(np.array(bridge_health_gains_pred_r) - np.array(bridge_health_gains_hist_r)))
        bridge_metrics_r['bridge_avg_health_gain_vs_nothing_r'] = float(np.mean(bridge_health_gains_vs_nothing_r))
        
        # ratio指标
        initial_health_array = np.array(bridge_initial_health)
        valid_indices = initial_health_array > 0
        if np.any(valid_indices):
            health_gains_ratio_r = np.array(bridge_health_gains_pred_r)[valid_indices] / initial_health_array[valid_indices]
            bridge_metrics_r['bridge_avg_health_gain_ratio_r'] = float(np.mean(health_gains_ratio_r))
        else:
            bridge_metrics_r['bridge_avg_health_gain_ratio_r'] = 0.0
        
        # normalize指标
        bridge_metrics_r['bridge_avg_health_gain_normalized_r'] = float(np.mean(bridge_health_gains_pred_r) / max_possible_gain)
        
        # 效率指标
        costs_array_r = np.array(bridge_costs_r)
        valid_cost_indices_r = costs_array_r > 0
        if np.any(valid_cost_indices_r):
            health_gain_per_1000_r = (np.array(bridge_health_gains_pred_r)[valid_cost_indices_r] * 1000) / costs_array_r[valid_cost_indices_r]
            bridge_metrics_r['health_gain_per_1000_dollars_r'] = float(np.mean(health_gain_per_1000_r))
        else:
            bridge_metrics_r['health_gain_per_1000_dollars_r'] = 0.0

    # byepisode聚合（如果有episode_idx_arr）
    if episode_idx_arr is not None:
        ep_accs = defaultdict(list)
        ep_vios = defaultdict(list)
        ep_costs = defaultdict(list)
        ep_budget_util = defaultdict(list)
        
        # hard constraint聚合（如果启用）
        if apply_hard_constraints:
            ep_accs_r = defaultdict(list)
            ep_vios_r = defaultdict(list)
            ep_costs_r = defaultdict(list)
            ep_budget_util_r = defaultdict(list)
        
        for idx, ep in enumerate(episode_idx_arr):
            ep_accs[ep].append(all_correct[idx])
            ep_vios[ep].append(all_violation[idx])
            ep_costs[ep].append(all_total_cost[idx])
            if idx < len(all_budget_utilization):
                ep_budget_util[ep].append(all_budget_utilization[idx])
            
            # hard constraint聚合（如果启用）
            if apply_hard_constraints:
                ep_accs_r[ep].append(all_correct_r[idx])
                ep_vios_r[ep].append(all_violation_r[idx])
                ep_costs_r[ep].append(all_total_cost_r[idx])
                if idx < len(all_budget_utilization_r):
                    ep_budget_util_r[ep].append(all_budget_utilization_r[idx])
        
        # 软约束结果
        behavioral_similarity_mean = np.mean([np.mean(v) for v in ep_accs.values()])
        behavioral_similarity_std = np.std([np.mean(v) for v in ep_accs.values()])
        mean_total_cost = np.mean([np.mean(v) for v in ep_costs.values()])
        violation_rate_mean = np.mean([np.mean(v) for v in ep_vios.values()])
        
        # budget使用占比（软约束）
        budget_usage_ratio = np.mean([np.mean(v) for v in ep_budget_util.values()]) if ep_budget_util else 0.0
        
        result = {
            'behavioral_similarity_mean': behavioral_similarity_mean,
            'behavioral_similarity_std': behavioral_similarity_std,
            'mean_total_cost': mean_total_cost,
            'violation_rate_mean': violation_rate_mean,
            'budget_usage_ratio': budget_usage_ratio,
            **action_proportions,  # add各动作比例
            **bridge_metrics       # add桥梁级别指标
        }
        
        # hard constraint结果（如果启用）
        if apply_hard_constraints:
            behavioral_similarity_mean_r = np.mean([np.mean(v) for v in ep_accs_r.values()])
            behavioral_similarity_std_r = np.std([np.mean(v) for v in ep_accs_r.values()])
            mean_total_cost_r = np.mean([np.mean(v) for v in ep_costs_r.values()])
            violation_rate_mean_r = np.mean([np.mean(v) for v in ep_vios_r.values()])
            
            # budget使用占比（硬约束）
            budget_usage_ratio_r = np.mean([np.mean(v) for v in ep_budget_util_r.values()]) if ep_budget_util_r else 0.0
            
            result.update({
                'behavioral_similarity_mean_r': behavioral_similarity_mean_r,
                'behavioral_similarity_std_r': behavioral_similarity_std_r,
                'mean_total_cost_r': mean_total_cost_r,
                'violation_rate_mean_r': violation_rate_mean_r,
                'budget_usage_ratio_r': budget_usage_ratio_r,
                **action_proportions_r,  # add硬约束各动作比例
                **bridge_metrics_r       # add硬约束桥梁级别指标
            })
    else:
        # 软约束结果
        result = {
            'behavioral_similarity_mean': np.mean(all_correct),
            'behavioral_similarity_std': np.std(all_correct),
            'mean_total_cost': np.mean(all_total_cost),
            'violation_rate_mean': np.mean(all_violation),
            'budget_usage_ratio': np.mean(all_budget_utilization) if all_budget_utilization else 0.0,
            **action_proportions,  # add各动作比例
            **bridge_metrics       # add桥梁级别指标
        }
        
        # hard constraint结果（如果启用）
        if apply_hard_constraints:
            result.update({
                'behavioral_similarity_mean_r': np.mean(all_correct_r),
                'behavioral_similarity_std_r': np.std(all_correct_r),
                'mean_total_cost_r': np.mean(all_total_cost_r),
                'violation_rate_mean_r': np.mean(all_violation_r),
                'budget_usage_ratio_r': np.mean(all_budget_utilization_r) if all_budget_utilization_r else 0.0,
                **action_proportions_r,  # add硬约束各动作比例
                **bridge_metrics_r       # add硬约束桥梁级别指标
            })

    # add综合评分
    result['comprehensive_score'] = calculate_comprehensive_score(result, mean_history_cost)
    if apply_hard_constraints:
        # 为硬约束创建临时metrics字典用于评分计算
        temp_metrics_r = {
            'behavioral_similarity_mean': result.get('behavioral_similarity_mean_r', 0.0),
            'health_gain_per_1000_dollars': result.get('health_gain_per_1000_dollars_r', 0.0),
            'budget_usage_ratio': result.get('budget_usage_ratio_r', 0.0),
            'bridge_health_gain_absolute': result.get('bridge_avg_health_gain_absolute_r', 0.0),
            'mean_total_cost': result.get('mean_total_cost_r', 0.0)
        }
        result['comprehensive_score_r'] = calculate_comprehensive_score(temp_metrics_r, mean_history_cost)

    # add总评估数量
    result['total_evaluations'] = total_evaluations_num
    if apply_hard_constraints:
        result['total_evaluations_r'] = total_evaluations_num_r

    if verbose:
        constraint_type = "Hard Constrained" if apply_hard_constraints else "Unconstrained"
        print(f"评估结果 ({constraint_type}): behavioral_similarity={result['behavioral_similarity_mean']:.4f}, "
              f"cost={result['mean_total_cost']:.1f}, vio={result['violation_rate_mean']:.4f}")
        
        if apply_hard_constraints:
            print(f"硬约束结果: behavioral_similarity={result['behavioral_similarity_mean_r']:.4f}, "
                  f"cost={result['mean_total_cost_r']:.1f}, vio={result['violation_rate_mean_r']:.4f}")
        
        print(f"预算使用占比 (vs历史): {result['budget_usage_ratio']:.4f}")
        
        if apply_hard_constraints and 'budget_usage_ratio_r' in result:
            print(f"硬约束预算使用占比 (vs历史): {result['budget_usage_ratio_r']:.4f}")
        
        # print桥梁级别指标
        if 'bridge_avg_cost' in result:
            print(f"桥梁平均成本: ${result['bridge_avg_cost']:.2f}")
            print(f"桥梁平均健康改善: {result['bridge_avg_health_gain_absolute']:.4f}")
            print(f"桥梁平均相对历史改善: {result['bridge_avg_health_gain_vs_history']:.4f}")
            print(f"桥梁平均相对nothing改善: {result['bridge_avg_health_gain_vs_nothing']:.4f}")
            print(f"健康改善效率 (per $1000): {result['health_gain_per_1000_dollars']:.4f}")
        
        # print综合评分
        print(f"综合评分: {result['comprehensive_score']:.4f}")
        if apply_hard_constraints:
            print(f"硬约束综合评分: {result['comprehensive_score_r']:.4f}")
        
        # print各动作比例
        print("各动作比例:")
        for action in sorted(action_proportions.keys()):
            print(f"  {action}: {action_proportions[action]:.4f}")
        
        if apply_hard_constraints:
            print("硬约束各动作比例:")
            for action in sorted(action_proportions_r.keys()):
                print(f"  {action}: {action_proportions_r[action]:.4f}")

    return result