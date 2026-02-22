import os
import sys
import torch as th
import numpy as np
import yaml
import json
import random

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_PAPER_ROOT = os.path.abspath(os.path.join(_BENCHMARK_ROOT, '..'))
if _BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARK_ROOT)
for _sub in ('episodes', 'regions', 'transition_metrics', 'processed'):
    _d = os.path.join(_PAPER_ROOT, 'dataset', 'data', _sub)
    if _d not in sys.path:
        sys.path.append(_d)

from algos.MultiTaskBC import MultiTaskBC
from algos.CPQDiscreteMultiTask_v2 import CPQDiscreteMultiTask
from algos.CPQ_2 import OfflineCPQDiscrete
from algos.IQLCQLMultiAgent import IQLCQLMultiAgent
from algos.CDT import CDT
from algos.CQL import CQLDiscrete
from algos.DiscreteBCMultiAgent import DiscreteBCMultiAgent
from algos.QMIXCQLMultiAgent_v5 import QMIXCQLMultiAgent
from algos.RandomBaseline import RandomBaselineOSRL, RandomBaselineMARL
from algos.OneStepRLDiscrete import OneStepRLDiscrete
import torch.nn.functional as F  # for CDT training
from evaluate_unified_v4 import evaluate_unified
from datetime import datetime

# ================= Data preparation utilities =================

def convert_to_serializable(obj):
    """
    Convert objects containing numpy types to JSON-serializable format.
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

def save_metrics(metrics, algo_name, out_dir="paper/benchmark/metrics_results"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo_name}_{dt_str}.json"
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"[INFO] Metrics for {algo_name} saved to {fname}")

def save_training_history(training_history, algo_name, out_dir="paper/benchmark/training_results"):
    """Save training history."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo_name}_training_history_{dt_str}.json"
    
    # Convert to serializable format
    serializable_history = convert_to_serializable(training_history)
    
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"[INFO] Training history for {algo_name} saved to {fname}")

def save_model(model, algo_name, out_dir="paper/benchmark/saved_models"):
    """Save the trained full model."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo_name}_model_{dt_str}.pth"
    
    try:
        th.save(model, os.path.join(out_dir, fname))
        print(f"[INFO] Model for {algo_name} saved to {fname}")
        return os.path.join(out_dir, fname)
    except Exception as e:
        print(f"[ERROR] Failed to save model for {algo_name}: {str(e)}")
        return None

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True

def load_env_info(env_info_path):
    with open(env_info_path, 'r') as f:
        return json.load(f)

def load_buffer(buffer_path, device):
    buffer = th.load(buffer_path)
    buffer.to(device)
    return buffer

def extract_data_from_buffer(buffer):
    all_obs, all_actions, episode_budgets, all_health, all_raw_cost, all_rewards, all_log_budgets = [], [], [], [], [], [], []
    actual_n_agents_list = []  # added：记录每个episode的实际智能体数量
    
    num_eps = buffer.episodes_in_buffer
    for i in range(num_eps):
        ep = buffer[i:i+1]
        obs = ep['obs'].cpu().numpy()           # [1, T, n_agents, state_dim]
        actions = ep['actions'].cpu().numpy()   # [1, T, n_agents] or [1, T, n_agents, 1]
        btotal = np.expm1(ep['btotal'].cpu().numpy()) if 'btotal' in ep.data.episode_data else 100000
        log_budgets = ep['log_budget'].cpu().numpy()   # [1, T, n_agents]
        
        # record实际智能体数量，并squeeze掉多余的维度
        actual_n_agents = ep['n_bridges_actual'].cpu().numpy().squeeze()  # from [1, 1] 变成标量
        actual_n_agents_list.append(actual_n_agents)
        
        all_log_budgets.append(log_budgets[0])
        all_obs.append(obs[0])
        all_actions.append(actions[0])
        episode_budgets.append(btotal)
        
        if 'health_state' in ep.data.transition_data:
            health = ep['health_state'].cpu().numpy()  # [1, T+1, n_agents]
            all_health.append(health[0])
        # Raw cost: append always
        if 'raw_cost' in ep.data.transition_data:
            raw_cost = ep['raw_cost'].cpu().numpy()  # [1,T,n_agents] or [1,T,n_agents,1]
            all_raw_cost.append(raw_cost[0])
        else:
            print(ep['raw_cost'].cpu().numpy().shape)
            # note actions[0] shape 可能是 (T, n_agents) 或 (T, n_agents, 1)
            all_raw_cost.append(np.zeros_like(actions[0]))
        if 'reward' in ep.data.transition_data:
            rewards = ep['reward'].cpu().numpy()  # [1, T, n_agents]
            all_rewards.append(rewards[0])
        else:
            all_rewards.append(np.zeros_like(actions[0]))
    
    data = np.stack(all_obs)        # [num_eps, T, n_agents, state_dim]
    actions = np.stack(all_actions) # [num_eps,T,n_agents] or [num_eps,T,n_agents,1]
    episode_budgets = np.array(episode_budgets).squeeze()
    health = np.stack(all_health) if all_health else None
    raw_cost = np.stack(all_raw_cost) if all_raw_cost else None
    rewards = np.stack(all_rewards) if all_rewards else None
    log_budgets = np.stack(all_log_budgets) if all_log_budgets else None
    actual_n_agents = np.array(actual_n_agents_list)  # [num_eps] correct shape

    print(f'Generated episode length: {data.shape}')

    return data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents


def compute_episode_scores(data, actions, rewards=None, raw_cost=None, health=None, action_costs=None):
    """
    Compute per-episode scores for expert filter.
    Returns: dict of [num_eps] per metric.
    Metrics:
      - reward_sum: 每个 episode 的奖励和（若 rewards 可用）
      - neg_cost_sum: 每个 episode 的 -总成本（越大越好）。若 raw_cost 可用则用 raw_cost，否则用 action_costs 基于动作估计
      - health_improve: 健康改善（若 health 可用），按激活桥均值
    """
    data = np.asarray(data)
    actions = np.asarray(actions)
    num_eps, T, n_agents, _ = data.shape

    scores = {}


    # reward_sum
    if rewards is not None:
        rewards = np.asarray(rewards)
        if rewards.ndim == 4 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        ep_reward_sum = rewards.sum(axis=(1, 2))
        scores['reward_sum'] = ep_reward_sum

    # neg_cost_sum
    if raw_cost is not None or action_costs is not None:
        if raw_cost is not None:
            rc = np.asarray(raw_cost)
            if rc.ndim == 4 and rc.shape[-1] == 1:
                rc = rc.squeeze(-1)
            ep_cost_sum = rc.sum(axis=(1, 2))
        else:
            ac = actions
            if ac.ndim == 4 and ac.shape[-1] == 1:
                ac = ac.squeeze(-1)
            cost_map = np.vectorize(lambda a: action_costs.get(int(a), 0))
            est_cost = cost_map(ac)
            ep_cost_sum = est_cost.sum(axis=(1, 2))
        scores['neg_cost_sum'] = -ep_cost_sum

    # health_improve
    if health is not None:
        health = np.asarray(health)
        init_h = health[:, 0]
        final_h = health[:, -1]
        ep_improve = np.zeros((num_eps,), dtype=np.float32)
        for ep in range(num_eps):
            agent_mask = (data[ep].sum(axis=2) != 0).any(axis=0)
            if agent_mask.any():
                ih = init_h[ep][agent_mask]
                fh = final_h[ep][agent_mask]
                ih = np.maximum(ih, 1)
                fh = np.maximum(fh, 1)
                ep_improve[ep] = float((fh - ih).mean())
            else:
                ep_improve[ep] = 0.0
        scores['health_improve'] = ep_improve

    return scores


def select_expert_indices(scores, metric='reward_sum', top_percent=0.5, higher_is_better=True):
    """
    Select top_percent episodes by metric.
    scores: from compute_episode_scores
    metric: e.g. 'reward_sum' | 'neg_cost_sum' | 'health_improve'
    top_percent: 0~1
    higher_is_better: True if higher is better
    Returns: indices (ascending)
    """
    assert 0 < top_percent <= 1.0, 'top_percent must be in (0,1]'
    vals = np.asarray(scores[metric])
    num_eps = vals.shape[0]
    k = max(1, int(np.ceil(num_eps * top_percent)))
    order = np.argsort(vals)
    if higher_is_better:
        chosen = order[-k:]
    else:
        chosen = order[:k]
    return np.sort(chosen)


def build_legal_actions_from_log_budgets(log_budgets, action_costs, normalization_params):
    mean = normalization_params['mean']
    std = normalization_params['std']
    #print(log_budgets.shape)
    log_budgets_raw = log_budgets * std + mean
    budgets = np.expm1(log_budgets_raw).squeeze(-1)
    #print(budgets.shape)
    #exit(0)
    # (1599, 15, 500, 5)
    num_eps, T, n_agents = budgets.shape
    n_actions = len(action_costs)
    legal = np.zeros((num_eps, T, n_agents, n_actions), dtype=bool)
    for a in range(n_actions):
        legal[..., a] = budgets >= action_costs[a]
    return legal

def build_ma_batches(data, actions, rewards, dones, legal_actions, max_n_agents):
    """
    Pad variable n_agents to max_n_agents, build agent_mask
    data: [num_eps, T, n_agents_i, obs_dim]
    actions: [num_eps, T, n_agents_i]
    rewards: [num_eps, T, n_agents_i]
    dones: [num_eps, T]
    legal_actions: [num_eps, T, n_agents_i, action_dim]
    返回:
        obs: [num_eps, T, max_n_agents, obs_dim]
        actions: [num_eps, T, max_n_agents]
        rewards: [num_eps, T, max_n_agents]
        dones: [num_eps, T]
        legal_actions: [num_eps, T, max_n_agents, action_dim]
        agent_mask: [num_eps, max_n_agents]
    """
    num_eps = len(data)
    T = data[0].shape[0]
    obs_dim = data[0].shape[-1]
    action_dim = legal_actions[0].shape[-1]
    obs_arr = np.zeros((num_eps, T, max_n_agents, obs_dim), dtype=np.float32)
    actions_arr = np.zeros((num_eps, T, max_n_agents), dtype=np.int64)
    rewards_arr = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    legal_arr = np.zeros((num_eps, T, max_n_agents, action_dim), dtype=np.bool_)
    agent_mask = np.zeros((num_eps, max_n_agents), dtype=np.float32)
    for i in range(num_eps):
        n_agents = data[i].shape[1]
        obs_arr[i, :, :n_agents, :] = data[i]

        #generic squeeze (drop last dim if 1)
        a = actions[i]
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a.squeeze(-1)
        actions_arr[i, :, :n_agents] = a

        r = rewards[i]
        if r.ndim == 3 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        rewards_arr[i, :, :n_agents] = r

        l = legal_actions[i]
        if l.ndim == 4 and l.shape[-2] == 1:
            l = np.squeeze(l, axis=-2)  #rare, redundant pad
        legal_arr[i, :, :n_agents, :] = l

        agent_mask[i, :n_agents] = 1.
    return obs_arr, actions_arr, rewards_arr, dones, legal_arr, agent_mask


# =================== 训练函数（修改版，返回训练历史） ===================

def augment_data_with_budgets(data, budgets, budget_mode='provided', log_budget_norm_params=None, episode_budgets=None):
    """
    Concatenate budget into obs
    
    Args:
        data: [num_eps, T, n_agents, obs_dim] 原始观察数据
        budgets: 预算信息，格式根据budget_mode而定
        budget_mode: 'provided' 或 'uniform'
        log_budget_norm_params: log预算归一化参数
        episode_budgets: episode级别的预算（用于uniform模式）
    
    Returns:
        augmented_data: [num_eps, T, n_agents, obs_dim+1] 增强后的观察数据
    """
    data = np.asarray(data)
    num_eps, T, n_agents, obs_dim = data.shape
    
    if budget_mode == 'provided':
        # use提供的budgets（已经是log格式）
        if budgets.ndim == 4 and budgets.shape[-1] == 1:
            budget_arr = budgets  # [num_eps, T, n_agents, 1]
        elif budgets.ndim == 3:
            budget_arr = budgets[..., np.newaxis]  # add最后一维
        else:
            raise ValueError(f"不支持的budgets形状: {budgets.shape}")
            
    elif budget_mode == 'uniform':
        # according toepisode_budgets计算统一预算
        budget_arr = np.zeros((num_eps, T, n_agents, 1))
        for ep in range(num_eps):
            if episode_budgets is not None:
                total_budget = float(episode_budgets[ep])
                per_step_agent_budget = total_budget / (T * n_agents)
                
                #convert to log (if norm params)
                if log_budget_norm_params is not None:
                    log_budget = np.log(per_step_agent_budget + 1e-8) / 10
                    budget_arr[ep, :, :, 0] = log_budget
                else:
                    budget_arr[ep, :, :, 0] = per_step_agent_budget
            else:
                budget_arr[ep, :, :, 0] = 0.0  # default预算
    else:
        raise ValueError("budget_mode 必须是 'provided' 或 'uniform'")
    
    # check形状兼容性
    if budget_arr.shape[:3] != data.shape[:3]:
        raise ValueError(f"预算形状 {budget_arr.shape[:3]} 与数据形状 {data.shape[:3]} 不匹配")
    
    # concat
    augmented_data = np.concatenate([data, budget_arr], axis=-1)
    
    print(f"Data augmentation done:")
    print(f"  Original data: {data.shape}")
    print(f"  Budget data: {budget_arr.shape}")
    print(f"  Augmented data: {augmented_data.shape}")
    
    return augmented_data


def train_multitask_iqlcql(
    data, actions, rewards, dones, legal_actions, max_n_agents, state_dim, action_dim, device='cpu',
    num_epochs=20, batch_size=16, eval_interval=2,
    log_budgets=None, env_info=None, raw_cost=None,
    test_data=None, test_actions=None, test_log_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None
):

    # check if data already has budget
    if data.shape[-1] == state_dim + 1:
        print(f"Data already contains budget info: {data.shape}")
        actual_obs_dim = data.shape[-1]  # use实际的观察维度
        model_has_budget = True
    else:
        print(f"Data does not contain budget info: {data.shape}")
        actual_obs_dim = state_dim
        model_has_budget = False


    obs_arr, actions_arr, rewards_arr, dones_arr, legal_arr, agent_mask = build_ma_batches(
        data, actions, rewards, dones, legal_actions, max_n_agents
    )

    obs_arr = th.tensor(obs_arr, dtype=th.float32, device=device)
    actions_arr = th.tensor(actions_arr, dtype=th.long, device=device)
    rewards_arr = th.tensor(rewards_arr, dtype=th.float32, device=device)
    dones_arr = th.tensor(dones_arr, dtype=th.float32, device=device)
    legal_arr = th.tensor(legal_arr, dtype=th.bool, device=device)
    agent_mask = th.tensor(agent_mask, dtype=th.float32, device=device)

    num_eps, T, N, _ = obs_arr.shape

    model = IQLCQLMultiAgent(
        obs_dim=actual_obs_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device
    )

    # add标识
    model.algorithm_type = 'marl'
    model.needs_budget_input = False
    model.has_budget_in_obs = model_has_budget  # tag budget in obs

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs + 1):
        perm = th.randperm(num_eps)
        losses = []
        for i in range(0, num_eps, batch_size):
            idx = perm[i:i+batch_size]
            batch = {
                'observations': obs_arr[idx],              # [B, T, N, obs_dim]
                'actions': actions_arr[idx],               # [B, T, N]
                'rewards': rewards_arr[idx],               # [B, T, N]
                'dones': dones_arr[idx],                   # [B, T]
                'legal_actions': legal_arr[idx],           # [B, T, N, action_dim]
                'agent_mask': agent_mask[idx],             # [B, N]
            }
            loss_dict = model.train_step(batch)
            losses.append(float(loss_dict['loss']))  # ensure转换为 Python float
        
        avg_loss = float(np.mean(losses))  # ensure转换为 Python float
        print(f"[IQLCQL-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))  # ensure是 Python int
        training_history['losses'].append(avg_loss)

        # === 训练中评估 ===
        if ((epoch % eval_interval == 0 or epoch == num_epochs) and
            test_data is not None and env_info is not None):
            print(f"--- Evaluate at Epoch {epoch} ---")
            '''metrics = evaluate_marl_osrl_aligned(
                model,
                test_data,
                test_actions,
                budgets=test_log_budgets,                 # or budgets_provided
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                verbose=False,
                budget_mode='provided',                # or 'provided'
                uniform_active_only=False
            )'''
            # at train_multitask_iqlcql 等函数中
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                verbose=True,
                use_agent_mask=True,  # MARL: evaluate active agents only
                budget_mode='provided',
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets
            )
            print(metrics)
            # record评估数据（确保数据类型转换）
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)

    return model, training_history

def prepare_cpq_dataset(data, actions, budgets, log_budgets, rewards, reward_func=None):
    obs_list, budget_list, action_list, reward_list = [], [], [], []
    next_obs_list, next_budget_list, cost_list, done_list = [], [], [], []

    num_episodes, T, n_agents, state_dim = data.shape
    for ep in range(num_episodes):
        for t in range(T):
            done_flag = 1.0 if t == T-1 else 0.0
            for agent in range(n_agents):
                obs = data[ep, t, agent]
                act = actions[ep, t, agent]
                
                # use真实的log预算
                budget = float(log_budgets[ep, t, agent].squeeze() if hasattr(log_budgets[ep, t, agent], 'squeeze') else log_budgets[ep, t, agent])
                # next_budget
                if t < T-1:
                    next_budget = float(log_budgets[ep, t+1, agent].squeeze() if hasattr(log_budgets[ep, t+1, agent], 'squeeze') else log_budgets[ep, t+1, agent])
                else:
                    next_budget = 0.0
                # next_obs
                if t < T-1:
                    next_obs = data[ep, t+1, agent]
                else:
                    next_obs = np.zeros_like(obs)
                
                obs_list.append(obs)
                budget_list.append(budget)
                action_list.append(int(act))
                # ====== key separation ======
                if rewards is not None:
                    reward = float(rewards[ep, t, agent])
                elif reward_func:
                    reward = reward_func(obs, act)
                else:
                    reward = 0.0  # or raise 错误
                reward_list.append(reward)
                next_obs_list.append(next_obs)
                next_budget_list.append(next_budget)
                #cost from log_budgets only
                cost_list.append(budget)
                done_list.append(done_flag)
    return {
        'obs': np.stack(obs_list),
        'budget': np.array(budget_list),
        'actions': np.array(action_list),
        'rewards': np.array(reward_list),
        'next_obs': np.stack(next_obs_list),
        'next_budget': np.array(next_budget_list),
        'costs': np.array(cost_list),
        'dones': np.array(done_list)
    }


def train_multitask_offline_cpq(
    data, actions, budgets, log_budgets,reward, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None,test_log_budgets=None
):
    dataset = prepare_cpq_dataset(data, actions, budgets, log_budgets,reward)

    # ========= Action-balanced sampling =========
    actions_np = dataset['actions']

    #repair action: assume action > 0 means repair
    repair_mask = actions_np > 0
    no_repair_mask = actions_np == 0

    weights = np.ones_like(actions_np, dtype=np.float32)
    weights[repair_mask] = 100.0      # key hyperparam（建议 5~20）
    weights[no_repair_mask] = 1.0

    sample_weights = th.tensor(weights, device=device)
    # ===============================================
    
    N = len(dataset['obs'])
    # convert totensor
    for k in dataset:
        dataset[k] = th.tensor(dataset[k], dtype=th.float32 if dataset[k].ndim > 1 or k not in ['actions'] else th.long, device=device)
    # model初始化
    model = OfflineCPQDiscrete(
        state_dim=state_dim, n_actions=action_dim, device=device
    )
    optimizer_q = th.optim.Adam(model.q_net.parameters(), lr=3e-4)
    optimizer_qc = th.optim.Adam(model.qc_net.parameters(), lr=3e-4)

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'q_losses': [],
        'qc_losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs+1):
        #perm = th.randperm(N)
        perm = th.multinomial(
            sample_weights,
            num_samples=N,
            replacement=True
        )
        losses, losses_c = [], []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch = {k: v[idx] for k, v in dataset.items()}
            loss_q, loss_qc = model.update(batch, optimizer_q, optimizer_qc)
            losses.append(float(loss_q))  # ensure转换为 Python float
            losses_c.append(float(loss_qc))
        
        avg_loss_q = float(np.mean(losses))
        avg_loss_qc = float(np.mean(losses_c))
        print(f"[CPQ] Epoch {epoch}, Q Loss: {avg_loss_q:.4f}, QC Loss: {avg_loss_qc:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['q_losses'].append(avg_loss_q)
        training_history['qc_losses'].append(avg_loss_qc)
        training_history['losses'].append(avg_loss_q + avg_loss_qc)

        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # evaluate：注意这里要实现适配 evaluate_osrl 的 model.act 接口
            #metrics = evaluate_osrl(model, test_data, test_actions, budgets=test_budgets,health=test_health, verbose=True)
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,  #OSRL: evaluate all agents
                budget_mode='provided',
                algorithm_type='osrl'
            )
            print(f"[CPQ] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history

def train_multitask_cpq(
    data, actions, budgets, log_budgets,reward, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None,test_log_budgets=None
):
    dataset = prepare_cpq_dataset(data, actions, budgets, log_budgets,reward)
    N = len(dataset['obs'])
    # convert totensor
    for k in dataset:
        dataset[k] = th.tensor(dataset[k], dtype=th.float32 if dataset[k].ndim > 1 or k not in ['actions'] else th.long, device=device)
    # model初始化
    model = CPQDiscreteMultiTask(
        state_dim=state_dim, n_actions=action_dim, device=device
    )
    optimizer_q = th.optim.Adam(model.q_net.parameters(), lr=1e-4)
    optimizer_qc = th.optim.Adam(model.qc_net.parameters(), lr=1e-4)

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'q_losses': [],
        'qc_losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs+1):
        perm = th.randperm(N)
        losses, losses_c = [], []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch = {k: v[idx] for k, v in dataset.items()}
            loss_q, loss_qc = model.update(batch, optimizer_q, optimizer_qc)
            losses.append(float(loss_q))  # ensure转换为 Python float
            losses_c.append(float(loss_qc))
        
        avg_loss_q = float(np.mean(losses))
        avg_loss_qc = float(np.mean(losses_c))
        print(f"[CPQ] Epoch {epoch}, Q Loss: {avg_loss_q:.4f}, QC Loss: {avg_loss_qc:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['q_losses'].append(avg_loss_q)
        training_history['qc_losses'].append(avg_loss_qc)
        training_history['losses'].append(avg_loss_q + avg_loss_qc)

        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # evaluate：注意这里要实现适配 evaluate_osrl 的 model.act 接口
            #metrics = evaluate_osrl(model, test_data, test_actions, budgets=test_budgets,health=test_health, verbose=True)
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,  #OSRL: evaluate all agents
                budget_mode='provided',
                algorithm_type='osrl'
            )
            print(f"[CPQ] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history


import torch as th
import numpy as np

def train_multitask_onestep(
    data, actions, budgets, log_budgets, reward, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None, test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None, test_log_budgets=None
):
    # 1. prepare data
    dataset = prepare_cpq_dataset(data, actions, budgets, log_budgets, reward)
    N = len(dataset['obs'])
    
    # ========= Action-balanced sampling =========
    actions_np = dataset['actions']
    repair_mask = actions_np > 0

    weights = np.ones_like(actions_np, dtype=np.float32)
    weights[repair_mask] = 100.0
    sample_weights = th.tensor(weights, device=device)
    # ===============================================

    # convert to tensor
    for k in dataset:
        dataset[k] = th.tensor(
            dataset[k], 
            dtype=th.float32 if dataset[k].ndim > 1 or k not in ['actions'] else th.long, 
            device=device
        )

    # 2. init model
    # input维度 = 状态维度 + 1 (Budget维度)
    input_dim = state_dim + 1
    model = OneStepRLDiscrete(
        state_dim=input_dim, 
        n_actions=action_dim, 
        device=device,
        hidden_sizes=[256, 256]
    )
    
    optimizer_bc = th.optim.Adam(model.bc_net.parameters(), lr=3e-4)
    optimizer_q = th.optim.Adam(model.q_net.parameters(), lr=3e-4)

    # 3. adapt eval interface (monkey patch)
    '''
    original_act = model.act
    
    def act_wrapper(state, budget=None, deterministic=True, **kwargs):
        # --- A. 统一转为 Tensor ---
        if not th.is_tensor(state):
            state = th.tensor(state, dtype=th.float32, device=device)
        
        # --- B. 处理 Budget 拼接逻辑 ---
        if budget is not None:
            if not th.is_tensor(budget):
                budget = th.tensor(budget, dtype=th.float32, device=device)
            
            # 1. 确保 State 是 2D: (Batch_Size, State_Dim)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            
            # 2. 确保 Budget 是 2D: (Batch_Size, 1)
            if budget.ndim == 0:
                # if是标量 (Scalar)，变成 (1, 1)
                budget = budget.view(1, 1)
            elif budget.ndim == 1:
                # if是向量 (Batch,)，变成 (Batch, 1)
                budget = budget.unsqueeze(1)
            
            # 3. 关键修复：广播 (Broadcasting)
            # if State 有多条数据 (N > 1)，但 Budget 只有 1 条 (1, 1)
            # 说明所有 Agent 共享同一个 Budget，需要复制 Budget 以匹配 State
            if state.shape[0] != budget.shape[0] and budget.shape[0] == 1:
                budget = budget.expand(state.shape[0], -1)
            
            # 4. concat
            # state: (N, S), budget: (N, 1) -> (N, S+1)
            state = th.cat([state, budget], dim=-1)
            
        return original_act(state, deterministic=deterministic)
    
    # replace实例方法
    model.act = act_wrapper
    '''

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'bc_losses': [],
        'q_losses': [],
        'eval_metrics': []
    }

    print(f"[OneStepRL] Start training with input_dim={input_dim} (State {state_dim} + Budget 1)...")

    for epoch in range(1, num_epochs+1):
        #perm = th.randperm(N)
        perm = th.multinomial(
            sample_weights,
            num_samples=N,
            replacement=True
        )
        losses_bc, losses_q = [], []
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            
            # build包含 budget 的输入
            batch_obs = dataset['obs'][idx]
            batch_bud = dataset['budget'][idx]
            
            # training时的维度通常是对齐的，但为了保险起见也加上 unsqueeze
            if batch_bud.ndim == 1: batch_bud = batch_bud.unsqueeze(-1)
            
            full_obs = th.cat([batch_obs, batch_bud], dim=-1)

            batch_next_obs = dataset['next_obs'][idx]
            batch_next_bud = dataset['next_budget'][idx]
            if batch_next_bud.ndim == 1: batch_next_bud = batch_next_bud.unsqueeze(-1)
            
            full_next_obs = th.cat([batch_next_obs, batch_next_bud], dim=-1)

            # build OneStepRL 需要的 batch 字典
            train_batch = {
                'obs': full_obs,
                'actions': dataset['actions'][idx],
                'rewards': dataset['rewards'][idx],
                'next_obs': full_next_obs,
                'dones': dataset['dones'][idx]
            }

            loss_bc, loss_q = model.update(train_batch, optimizer_bc, optimizer_q)
            losses_bc.append(loss_bc)
            losses_q.append(loss_q)
        
        avg_loss_bc = float(np.mean(losses_bc))
        avg_loss_q = float(np.mean(losses_q))
        
        print(f"[OneStepRL] Epoch {epoch}, BC Loss: {avg_loss_bc:.4f}, Q Loss: {avg_loss_q:.4f}")
        
        # record
        training_history['epochs'].append(int(epoch))
        training_history['bc_losses'].append(avg_loss_bc)
        training_history['q_losses'].append(avg_loss_q)
        training_history['losses'].append(avg_loss_bc + avg_loss_q)

        # evaluate
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # note：这里调用 evaluate_unified 时，内部会调用被我们要修改过的 model.act
            # no论 evaluate_unified 传入的是标量 budget 还是 batch budget，act_wrapper 都能处理
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets, # here传入 log budgets
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,
                budget_mode='provided',
                algorithm_type='osrl'
            )
            print(f"[OneStepRL] Eval @ epoch {epoch}: behavioral_similarity={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}")
            
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    # restore原始的 act 方法，以便模型可以正常保存（pickle无法序列化本地函数）
    #model.act = original_act

    return model, training_history

def train_multitask_bc(
    data,actions, episode_budgets,log_budgets, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None,
    #expert trajectory filter（仅训练集）
    expert_percent=None,
    expert_metric='reward_sum',
    expert_higher_is_better=True,
    action_costs=None,
    rewards=None,
    raw_cost=None,
    health=None,
    log_budget_norm_params=None,
    test_episode_budgets=None,
    test_log_budgets=None
):
    """
    data: [num_episodes, T, n_agents, state_dim]
    actions: [num_episodes, T, n_agents]
    budgets: [num_episodes]
    """
    # optional：专家轨迹筛选（按 episode 粒度）
    if expert_percent is not None:
        try:

            score_dict = compute_episode_scores(
                data, actions, rewards=rewards, raw_cost=raw_cost, health=health, action_costs=action_costs
            )
            summarize_episode_metric_distribution(score_dict, metric=expert_metric, tag="multitask_bc_train")
            idx_sel = select_expert_indices(
                score_dict, metric=expert_metric, top_percent=float(expert_percent), higher_is_better=expert_higher_is_better
            )
            print(f"[MultiTaskBC] Using expert trajectories: metric={expert_metric}, top_percent={expert_percent}, selected {len(idx_sel)}/{len(data)}")
            print(f"data: {data.shape}, actions: {actions.shape}, log_budgets: {log_budgets.shape}, rewards: {rewards.shape}, raw_cost: {raw_cost.shape}, health: {health.shape}")
            
            data = data[idx_sel]
            actions = actions[idx_sel]
            log_budgets = log_budgets[idx_sel]
            #optional: if user provides rewards/raw_cost/health 也同步裁剪，避免后续可能使用
            if rewards is not None:
                rewards = rewards[idx_sel]
            if raw_cost is not None:
                raw_cost = raw_cost[idx_sel]
            if health is not None:
                health = health[idx_sel]
        except Exception as e:
            print(f"[MultiTaskBC] Expert filter failed, falling back to full data. Reason: {e}")


    #print(f'data: {data.shape}, actions: {actions.shape}, budgets: {log_budgets.shape}, rewards: {rewards.shape}, raw_cost: {raw_cost.shape}, health: {health.shape}')
    #exit(0)
    obs_list, budget_list, action_list = [], [], []
    num_episodes, T, n_agents, state_dim = data.shape
    for ep in range(num_episodes):
        for t in range(T):
            for agent in range(n_agents):
                obs = data[ep, t, agent]
                act = actions[ep, t, agent]
                
                # direct使用log_budgets，无需反归一化
                budget = log_budgets[ep, t, agent]
                if hasattr(budget, 'squeeze'):
                    budget = budget.squeeze()
                budget = float(budget)

                obs_list.append(obs)
                budget_list.append(float(budget))  # ensure是标量
                action_list.append(int(act))       # ensure是标量

    obs_arr = th.tensor(np.stack(obs_list), dtype=th.float32)         # (N, state_dim)
    budget_arr = th.tensor(np.array(budget_list), dtype=th.float32)   # (N,)
    action_arr = th.tensor(np.array(action_list), dtype=th.long)      # (N,)

    model = MultiTaskBC(state_dim=state_dim, action_dim=action_dim, device=device)
    model.setup_optimizers(actor_lr=1e-3)

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs+1):
        perm = th.randperm(obs_arr.shape[0])
        obs, budgets_, acts = obs_arr[perm], budget_arr[perm], action_arr[perm]
        losses = []
        for i in range(0, obs.shape[0], batch_size):
            batch_obs = obs[i:i+batch_size].to(device)
            batch_bud = budgets_[i:i+batch_size].to(device)
            batch_act = acts[i:i+batch_size].to(device)
            loss_actor, stats_actor = model.actor_loss(batch_obs, batch_bud, batch_act)
            losses.append(float(loss_actor.item()))  # ensure转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[MultiTaskBC] Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)
        
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # evaluate：注意这里要实现适配 evaluate_osrl 的 model.act 接口
            #metrics = evaluate_osrl(model, test_data, test_actions, budgets=test_budgets,health=test_health, verbose=True)
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=action_costs,
                raw_cost=test_raw_cost,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,  #OSRL: evaluate all agents
                budget_mode='provided',
                actual_n_agents=actual_n_agents,
                algorithm_type='osrl'
            )
            print(f"[MultiTaskBC] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history

def train_multitask_cql(
    data, actions, budgets, log_budgets, reward, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None, test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None, test_log_budgets=None
):
    """
    Train CQL (Conservative Q-Learning)
    
    Args:
        data: [num_episodes, T, n_agents, state_dim]
        actions: [num_episodes, T, n_agents]
        budgets: episode级别的预算 [num_episodes]
        log_budgets: log格式的预算 [num_episodes, T, n_agents]
        reward: 奖励 [num_episodes, T, n_agents]
        state_dim: 状态维度
        action_dim: 动作维度
        device: 设备
        num_epochs: 训练轮数
        batch_size: 批次大小
        eval_interval: 评估间隔
        test_data: 测试数据
        test_actions: 测试动作
        test_budgets: 测试预算
        test_raw_cost: 测试原始成本
        test_health: 测试健康状态
        actual_n_agents: 实际智能体数量
        log_budget_norm_params: log预算归一化参数
        test_episode_budgets: 测试episode预算
        test_log_budgets: 测试log预算
    """
    # 1. prepare data
    dataset = prepare_cpq_dataset(data, actions, budgets, log_budgets, reward)
    N = len(dataset['obs'])
    

    # ========= Action-balanced sampling =========
    actions_np = dataset['actions']
    repair_mask = actions_np > 0
    no_repair_mask = actions_np == 0
    weights = np.ones_like(actions_np, dtype=np.float32)
    weights[repair_mask] = 100.0
    sample_weights = th.tensor(weights, device=device)
    # ===============================================

    # convert totensor
    for k in dataset:
        dataset[k] = th.tensor(
            dataset[k], 
            dtype=th.float32 if dataset[k].ndim > 1 or k not in ['actions'] else th.long, 
            device=device
        )
    
    # 2. init model
    # input维度 = 状态维度 + 1 (Budget维度)
    input_dim = state_dim + 1
    model = CQLDiscrete(
        state_dim=input_dim,
        n_actions=action_dim,
        hidden_sizes=[256, 256],
        gamma=0.99,
        tau=0.005,
        cql_weight=1.0,
        device=device
    )
    
    optimizer = th.optim.Adam(model.q_net.parameters(), lr=3e-4)
    
    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'bellman_losses': [],
        'cql_losses': [],
        'eval_metrics': []
    }
    
    print(f"[CQL] Start training with input_dim={input_dim} (State {state_dim} + Budget 1)...")
    
    for epoch in range(1, num_epochs+1):
        #perm = th.randperm(N)
        perm = th.multinomial(
            sample_weights,
            num_samples=N,
            replacement=True
        )
        losses_total, losses_bellman, losses_cql = [], [], []
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            
            # build包含 budget 的输入
            batch_obs = dataset['obs'][idx]
            batch_bud = dataset['budget'][idx]
            
            # ensure维度正确
            if batch_bud.ndim == 1: 
                batch_bud = batch_bud.unsqueeze(-1)
            
            full_obs = th.cat([batch_obs, batch_bud], dim=-1)
            
            batch_next_obs = dataset['next_obs'][idx]
            batch_next_bud = dataset['next_budget'][idx]
            if batch_next_bud.ndim == 1: 
                batch_next_bud = batch_next_bud.unsqueeze(-1)
            
            full_next_obs = th.cat([batch_next_obs, batch_next_bud], dim=-1)
            
            # build CQL 需要的 batch 字典
            train_batch = {
                'obs': full_obs,
                'actions': dataset['actions'][idx],
                'rewards': dataset['rewards'][idx],
                'next_obs': full_next_obs,
                'dones': dataset['dones'][idx]
            }
            
            loss_total, loss_bellman, loss_cql = model.update(train_batch, optimizer)
            losses_total.append(loss_total)
            losses_bellman.append(loss_bellman)
            losses_cql.append(loss_cql)
        
        avg_loss_total = float(np.mean(losses_total))
        avg_loss_bellman = float(np.mean(losses_bellman))
        avg_loss_cql = float(np.mean(losses_cql))
        
        print(f"[CQL] Epoch {epoch}, Total Loss: {avg_loss_total:.4f}, Bellman Loss: {avg_loss_bellman:.4f}, CQL Loss: {avg_loss_cql:.4f}")
        
        # record
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss_total)
        training_history['bellman_losses'].append(avg_loss_bellman)
        training_history['cql_losses'].append(avg_loss_cql)
        
        # evaluate
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False, # OSRL: evaluate all agents
                budget_mode='provided',
                algorithm_type='osrl'
            )
            print(f"[CQL] Eval @ epoch {epoch}: behavioral_similarity={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}")
            
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history


def prepare_cdt_dataset(data, actions, rewards, raw_cost, health=None, max_n_agents=None):
    """
    CDT dataset prep (keep MARL format)
    
    data:     [num_episodes, T, n_agents, state_dim]
    actions:  [num_episodes, T, n_agents]
    rewards:  [num_episodes, T, n_agents]
    raw_cost: [num_episodes, T, n_agents, 1] or [num_episodes, T, n_agents] or None
    health:   [num_episodes, T+1, n_agents] or None
    max_n_agents: max agents for padding

    Returns:
        obs_arr:         [num_eps, T, max_n_agents, state_dim]
        acts_arr:        [num_eps, T, max_n_agents, action_dim]
        returns_to_go:   [num_eps, T, max_n_agents]
        costs_to_go:     [num_eps, T, max_n_agents]
        time_steps:      [num_eps, T, max_n_agents]
        raw_cost_arr:    [num_eps, T, max_n_agents]
        health_arr:      [num_eps, T+1, max_n_agents] or None
    """
    num_eps, T, n_agents, state_dim = data.shape
    
    if max_n_agents is None:
        max_n_agents = n_agents
    
    # init输出数组
    obs_arr = np.zeros((num_eps, T, max_n_agents, state_dim), dtype=np.float32)
    acts_arr = np.zeros((num_eps, T, max_n_agents, 4), dtype=np.float32)  # 假设action_dim=4
    returns_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    costs_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    time_steps = np.zeros((num_eps, T, max_n_agents), dtype=np.int32)
    raw_cost_arr = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    health_arr = np.zeros((num_eps, T+1, max_n_agents), dtype=np.float32) if health is not None else None
    
    for ep in range(num_eps):
        for agent in range(min(n_agents, max_n_agents)):
            # 观察数据
            obs_arr[ep, :, agent, :] = data[ep, :, agent, :]
            
            # action数据 - 转换为one-hot
            acts = actions[ep, :, agent]
            if acts.ndim > 1:
                acts = acts.squeeze(-1)
            action_dim = acts.max() + 1 if acts.max() > 0 else 1
            acts_oh = np.eye(action_dim)[acts.astype(int)]
            acts_arr[ep, :, agent, :acts_oh.shape[1]] = acts_oh
            
            # 奖励数据
            rews = rewards[ep, :, agent]
            returns_to_go[ep, :, agent] = rews[::-1].cumsum()[::-1]
            
            # 成本数据
            if raw_cost is not None:
                costs = raw_cost[ep, :, agent]
                if costs.ndim > 1:
                    costs = costs.squeeze(-1)
                costs_to_go[ep, :, agent] = costs[::-1].cumsum()[::-1]
                raw_cost_arr[ep, :, agent] = costs
            
            # 时间步
            time_steps[ep, :, agent] = np.arange(T)
            
            # health: fix time dim mismatch
            if health is not None:
                if health.shape[1] == T:  # health is [T, n_agents]
                    # fill last col: copy year T-1
                    health_ep_agent = health[ep, :, agent]  # [T] 或 [T, 1]
                    if health_ep_agent.ndim > 1:
                        health_ep_agent = health_ep_agent.squeeze(-1)  # ensure是 [T] 格式
                    health_arr[ep, :T, agent] = health_ep_agent
                    health_arr[ep, T, agent] = health_ep_agent[-1]  # copy最后一年的健康状态
                elif health.shape[1] == T+1:  # health is [T+1, n_agents]
                    health_ep_agent = health[ep, :, agent]  # [T+1] 或 [T+1, 1]
                    if health_ep_agent.ndim > 1:
                        health_ep_agent = health_ep_agent.squeeze(-1)  # ensure是 [T+1] 格式
                    health_arr[ep, :, agent] = health_ep_agent
                else:
                    raise ValueError(f"health time dim incorrect: {health.shape[1]}, expected {T} 或 {T+1}")
    
    return obs_arr, acts_arr, returns_to_go, costs_to_go, time_steps, raw_cost_arr, health_arr

def _prepare_cdt_dataset(data, actions, rewards, raw_cost, health=None):
    """
    data:     [num_episodes, T, n_agents, state_dim]
    actions:  [num_episodes, T, n_agents]
    rewards:  [num_episodes, T, n_agents]
    raw_cost: [num_episodes, T, n_agents, 1] or [num_episodes, T, n_agents] or None
    health:   [num_episodes, T+1, n_agents] or None

    Returns:
        obs_arr:         [N', T, state_dim]
        acts_arr:        [N', T, action_dim]
        returns_to_go:   [N', T]
        costs_to_go:     [N', T]
        time_steps:      [N', T]
        raw_cost_arr:    [N', T]
        health_arr:      [N', T+1] or None
        episode_idx_arr: [N',]      # added
        agent_idx_arr:   [N',]      # added
    """
    obs_list, act_list, retg_list, costg_list, t_list, raw_cost_list = [], [], [], [], [], []
    health_list = [] if health is not None else None
    episode_idx_list, agent_idx_list = [], []

    N, T, n_agents, state_dim = data.shape
    for ep in range(N):
        for agent in range(n_agents):
            obs = data[ep, :, agent, :]            # [T, state_dim]
            acts = actions[ep, :, agent]           # [T] or [T, 1]
            rews = rewards[ep, :, agent]           # [T]
            # fix raw_cost shape
            if raw_cost is not None:
                costs = raw_cost[ep, :, agent]     # [T, 1] or [T]
                if costs.ndim > 1:
                    costs = costs.squeeze(-1)      # [T]
            else:
                costs = np.zeros_like(rews)
            # return-to-go: right cumsum
            returns_to_go = rews[::-1].cumsum()[::-1]
            costs_to_go = costs[::-1].cumsum()[::-1]
            time_steps = np.arange(T)
            # action若 shape (T, 1)，转 (T,)
            if acts.ndim > 1:
                acts = acts.squeeze(-1)
            # action转 one-hot，统一 pad 到 max_adim，后处理
            action_dim = acts.max() + 1 if acts.max() > 0 else 1
            acts_oh = np.eye(action_dim)[acts.astype(int)]  # [T, action_dim]
            obs_list.append(obs)
            act_list.append(acts_oh)
            retg_list.append(returns_to_go)
            costg_list.append(costs_to_go)
            t_list.append(time_steps)
            raw_cost_list.append(costs)  # [T]
            episode_idx_list.append(ep)
            agent_idx_list.append(agent)
            if health is not None:
                health_this = health[ep, :, agent]
                health_list.append(health_this)

    # unified action_dim
    max_adim = max(a.shape[1] for a in act_list)
    for i in range(len(act_list)):
        if act_list[i].shape[1] < max_adim:
            pad = np.zeros((act_list[i].shape[0], max_adim - act_list[i].shape[1]))
            act_list[i] = np.concatenate([act_list[i], pad], axis=1)
    # convert to numpy
    obs_arr = np.stack(obs_list)         # [N', T, state_dim]
    acts_arr = np.stack(act_list)        # [N', T, action_dim]
    returns_to_go = np.stack(retg_list)  # [N', T]
    costs_to_go = np.stack(costg_list)   # [N', T]
    time_steps = np.stack(t_list)        # [N', T]
    raw_cost_arr = np.stack(raw_cost_list) # [N', T]
    health_arr = np.stack(health_list) if health is not None else None  # [N', T+1]
    episode_idx_arr = np.array(episode_idx_list)  # [N']
    agent_idx_arr = np.array(agent_idx_list)      # [N']
    return obs_arr, acts_arr, returns_to_go, costs_to_go, time_steps, raw_cost_arr, health_arr, episode_idx_arr, agent_idx_arr


def train_multitask_cdt(
    data, actions, returns_to_go, costs_to_go, time_steps,
    state_dim, action_dim, max_action,action_costs=None,
    device='cpu', num_epochs=20, batch_size=128, eval_interval=2,
    test_data=None, test_actions=None, test_returns=None, test_costs=None, test_time_steps=None,
    test_health=None, test_raw_cost=None,max_bridges=None,test_episode_idx=None,    # added
    test_agent_idx=None,  actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None,
    test_log_budgets=None
):
    """
    data: [N, T, state_dim]
    actions: [N, T, action_dim] 或 [N, T]
    returns_to_go: [N, T]
    costs_to_go: [N, T]
    time_steps: [N, T]
    test_health: [N, T+1] 或 None
    """
    model = CDT(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action,
        seq_len=data.shape[1], device=device
    ).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)

    

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }

    N = data.shape[0]
    for epoch in range(1, num_epochs+1):
        perm = np.random.permutation(N)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_obs = th.tensor(data[idx], dtype=th.float32, device=device)
            batch_act = th.tensor(actions[idx], dtype=th.float32, device=device)
            batch_ret = th.tensor(returns_to_go[idx], dtype=th.float32, device=device)
            batch_cost = th.tensor(costs_to_go[idx], dtype=th.float32, device=device)
            batch_time = th.tensor(time_steps[idx], dtype=th.long, device=device)
            action_preds, _, _ = model(batch_obs, batch_act, batch_ret, batch_cost, batch_time)
            loss = F.mse_loss(action_preds, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))  # ensure转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[CDT] Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === test set eval ===
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            '''metrics = evaluate_cdt_parallel(
                model,
                test_data,
                test_actions,
                test_returns,
                test_costs,
                test_time_steps,
                health=test_health,
                raw_cost=test_raw_cost,
                max_bridges=max_bridges,
                episode_idx_arr=test_episode_idx,   # added
                agent_idx_arr=test_agent_idx,       # added
                verbose=False
            )'''
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                verbose=True,
                is_cdt_model=True,
                returns_to_go=test_returns,
                costs_to_go=test_costs,
                time_steps=test_time_steps,
                episode_idx_arr=test_episode_idx,
                agent_idx_arr=test_agent_idx,
                max_bridges=max_bridges,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                device=device
            )


            print(
                f"[CDT] Eval @ epoch {epoch}: "
                f"behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, "
                f"vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else "")
            )
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)

    return model, training_history


def train_multitask_discrete_bc(
    data, actions, rewards, dones, legal_actions, max_n_agents, state_dim, action_dim, device='cpu',
    num_epochs=20, batch_size=16, eval_interval=2,log_budgets=None,
    test_data=None, test_actions=None, test_log_budgets=None, env_info=None,
    test_legal_actions=None, test_raw_cost=None, test_health=None, action_costs=None, verbose=False,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None,
    #expert trajectory filter开关（仅影响训练集）
    expert_percent=None,                 # e.g. 0.5 = top 50% as expert
    expert_metric='reward_sum',          # optional: 'reward_sum' | 'neg_cost_sum' | 'health_improve'
    expert_higher_is_better=True,         # True if higher is better
    train_health=None,
):
    # === optional: expert filter (training only) ===
    if expert_percent is not None:
        try:
            score_dict = compute_episode_scores(
                data, actions, rewards=rewards, raw_cost=None, health=train_health, action_costs=action_costs
            )
            summarize_episode_metric_distribution(score_dict, metric=expert_metric, tag="discrete_bc_train")
            idx_sel = select_expert_indices(
                score_dict, metric=expert_metric, top_percent=float(expert_percent), higher_is_better=expert_higher_is_better
            )
            print(f"[DiscreteBC-MA] Using expert trajectories: metric={expert_metric}, top_percent={expert_percent}, selected {len(idx_sel)}/{len(data)}")

            if verbose:
                print(f"[DiscreteBC-MA] Using expert trajectories: metric={expert_metric}, top_percent={expert_percent}, selected {len(idx_sel)}/{len(data)}")
            #slice training set by selected episode indices
            data = data[idx_sel]
            actions = actions[idx_sel]
            rewards = rewards[idx_sel] if rewards is not None else None
            dones = dones[idx_sel]
            legal_actions = legal_actions[idx_sel]
        except Exception as e:
            print(f"[DiscreteBC-MA] Expert filter failed, falling back to full data. Reason: {e}")
    # === align training batches ===
    obs_arr, actions_arr, rewards_arr, dones_arr, legal_arr, agent_mask = build_ma_batches(
        data, actions, rewards, dones, legal_actions, max_n_agents
    )

    # check if data already has budget
    if data.shape[-1] == state_dim + 1:
        print(f"Data already contains budget info: {data.shape}")
        actual_obs_dim = data.shape[-1]  # use实际的观察维度
        model_has_budget = True
    else:
        print(f"Data does not contain budget info: {data.shape}")
        actual_obs_dim = state_dim
        model_has_budget = False

    obs_arr = th.tensor(obs_arr, dtype=th.float32, device=device)
    actions_arr = th.tensor(actions_arr, dtype=th.long, device=device)
    agent_mask = th.tensor(agent_mask, dtype=th.float32, device=device)
    num_eps, T, N, _ = obs_arr.shape


    model = DiscreteBCMultiAgent(
        obs_dim=actual_obs_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device
    )

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs + 1):
        perm = th.randperm(num_eps)
        losses = []
        for i in range(0, num_eps, batch_size):
            idx = perm[i:i+batch_size]
            batch = {
                'observations': obs_arr[idx],      # [B, T, N, obs_dim]
                'actions': actions_arr[idx],       # [B, T, N]
                'agent_mask': agent_mask[idx],     # [B, N]
            }
            loss_dict = model.train_step(batch)
            losses.append(float(loss_dict['policy_loss']))  # ensure转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[DiscreteBC-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === evaluate ===
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None and env_info is not None:
            # by evaluate_marl 需求，直接用原始 test_data, test_actions, test_log_budgets
            model.policy_network.eval()
            with th.no_grad():
                metrics = evaluate_unified(
                    model=model,
                    data=test_data,
                    actions=test_actions,
                    budgets=test_log_budgets,
                    health=test_health,
                    action_costs=action_costs,
                    raw_cost=test_raw_cost,
                    verbose=verbose,
                    use_agent_mask=True, # MARL: evaluate active agents only
                    budget_mode='provided',
                    log_budget_norm_params=log_budget_norm_params,
                    test_episode_budgets=test_episode_budgets,
                )
                print(f"[DiscreteBC-MA] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                      f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                      + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
                # record评估数据
                eval_record = {
                    'epoch': int(epoch),
                    'metrics': convert_to_serializable(metrics)
                }
                training_history['eval_metrics'].append(eval_record)
            model.policy_network.train()

    return model, training_history

def train_multitask_qmixcql(
    data, actions, rewards, dones, legal_actions, max_n_agents, obs_dim, action_dim, device='cpu',
    num_epochs=20, batch_size=16, eval_interval=2,action_costs=None,
    log_budgets=None, env_info=None, raw_cost=None,
    test_data=None, test_actions=None, test_log_budgets=None, test_raw_cost=None, test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None
):
    # === 1. form batch ===
    obs_arr, actions_arr, rewards_arr, dones_arr, legal_arr, agent_mask = build_ma_batches(
        data, actions, rewards, dones, legal_actions, max_n_agents
    )
    # print("legal_arr.shape (from build_ma_batches):", legal_arr.shape)
    # === 2. pad obs_arr to T+1 ===
    if obs_arr.shape[1] == actions_arr.shape[1]:
        last_obs = obs_arr[:, -1:, :, :]
        obs_arr = np.concatenate([obs_arr, last_obs], axis=1)


    # check if data already has budget
    if data.shape[-1] == obs_dim + 1:
        print(f"Data already contains budget info: {data.shape}")
        actual_obs_dim = data.shape[-1]  # use实际的观察维度
        model_has_budget = True
    else:
        print(f"Data does not contain budget info: {data.shape}")
        actual_obs_dim = obs_dim
        model_has_budget = False

    # === 3. unify shape params ===
    B, Tp1, N, O = obs_arr.shape
    T = Tp1 - 1

    # === 4. check all arr shapes ===
    assert actions_arr.shape[1] == T, f"actions_arr T mismatch, expected{T}, got{actions_arr.shape[1]}"
    assert rewards_arr.shape[1] == T
    assert dones_arr.shape[1] == T
    assert legal_arr.shape[1] == T

    # === 5. to tensor ===
    obs_arr = th.tensor(obs_arr, dtype=th.float32, device=device)
    actions_arr = th.tensor(actions_arr, dtype=th.long, device=device)
    rewards_arr = th.tensor(rewards_arr, dtype=th.float32, device=device)
    dones_arr = th.tensor(dones_arr, dtype=th.float32, device=device)
    legal_arr = th.tensor(legal_arr, dtype=th.bool, device=device)
    agent_mask = th.tensor(agent_mask, dtype=th.float32, device=device)

    # === 6. QMIXCQLMultiAgent init ===
    model = QMIXCQLMultiAgent(
        obs_dim=actual_obs_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device,
        state_dim=max_n_agents * actual_obs_dim
    )

    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }

    for epoch in range(1, num_epochs + 1):
        perm = th.randperm(B)
        losses = []
        for i in range(0, B, batch_size):
            idx = perm[i:i+batch_size]
            b = idx.shape[0]
            # force用外层T/N/O
            batch = {
                'observations': obs_arr[idx],        # [b, T+1, N, O]
                'actions': actions_arr[idx],         # [b, T, N]
                'rewards': rewards_arr[idx],         # [b, T, N]
                'dones': dones_arr[idx],             # [b, T]
                'legal_actions': legal_arr[idx],     # [b, T, N, action_dim]
                'agent_mask': agent_mask[idx],       # [b, N]
                'env_states': obs_arr[idx, :-1].reshape(b, T, N*O),
                'next_env_states': obs_arr[idx, 1:].reshape(b, T, N*O),
            }
            #print(b,T,N*O)
            #exit(0)
            # debug print
            # print('obs_arr[idx, :-1].shape =', obs_arr[idx, :-1].shape)
            # print('reshape target =', (b, T, N*O))
            loss_dict = model.train_step(batch)
            losses.append(float(loss_dict['loss']))  # ensure转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[QMIXCQL-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # record训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === evaluate ===
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None and env_info is not None:
            print(f"--- Evaluate at Epoch {epoch} ---")
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=action_costs,
                raw_cost=test_raw_cost,
                verbose=True,
                use_agent_mask=True, # MARL: evaluate active agents only
                budget_mode='provided',
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets
            )
            print(f"[QMIX-CQL] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                      f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                      + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history


# add随机算法的训练函数
def train_random_baseline(
    data, actions, rewards, state_dim, action_dim, device='cpu',action_costs=None,
    num_epochs=20, eval_interval=2,seed=42,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None, test_health=None,
    algo_type='osrl',log_budget_norm_params=None, test_episode_budgets=None, **kwargs,
):
    """
    Random baseline 'training' (create model and evaluate)
    
    Args:
        algo_type: 'osrl' or 'marl'
    """
    # according to算法类型创建相应的随机基线
    if algo_type == 'osrl':
        model = RandomBaselineOSRL(
            state_dim=state_dim, 
            action_dim=action_dim, 
            device=device, 
            seed=seed
        )
    elif algo_type == 'marl':
        max_n_agents = kwargs.get('max_n_agents', 500)
        model = RandomBaselineMARL(
            obs_dim=state_dim,
            max_n_agents=max_n_agents,
            action_dim=action_dim,
            device=device,
            seed=seed,
            use_legal_constraint=False
        )
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    
    # training历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }
    
    print(f"[Random-{algo_type.upper()}] Random baseline requires no training, evaluating directly...")
    
    # Periodic eval (simulate training)
    for epoch in range(eval_interval, num_epochs + 1, eval_interval):
        # Random has no real loss, record 0
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(0.0)
        
        if test_data is not None:
            print(f"--- Evaluate Random Baseline at Epoch {epoch} ---")
            actual_n_agents = kwargs.get('actual_n_agents')
            # according to算法类型选择相应的评估函数
            if algo_type == 'osrl':
                metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_budgets,
                health=test_health,
                action_costs=action_costs,
                raw_cost=test_raw_cost,
                verbose=True,
                use_agent_mask=False,  #OSRL: evaluate all agents
                budget_mode='uniform',
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                algorithm_type='osrl'

            )
            elif algo_type == 'marl':
                
                env_info = kwargs.get('env_info')
                test_log_budgets = kwargs.get('test_log_budgets')
                # for齐到 OSRL 单智能体评估口径
                metrics = evaluate_unified(
                    model=model,
                    data=test_data,
                    actions=test_actions,
                    budgets=test_log_budgets,
                    health=test_health,
                    action_costs=action_costs,
                    raw_cost=test_raw_cost,
                    verbose=True,
                    use_agent_mask=True,  #MARL: evaluate active agents only
                    budget_mode='provided',
                    actual_n_agents=actual_n_agents,
                    log_budget_norm_params=log_budget_norm_params,
                    test_episode_budgets=test_episode_budgets
                )
            
            print(f"[Random-{algo_type.upper()}] Eval @ epoch {epoch}: behavioral_similarity_mean={metrics['behavioral_similarity_mean']:.4f}, "
                  f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                  + (f", health_improve={metrics['bridge_avg_health_gain_vs_history']:.3f}" if 'bridge_avg_health_gain_vs_history' in metrics else ""))
            
            # record评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history

def summarize_episode_metric_distribution(scores, metric, tag="train", save_dir="paper/benchmark/evaluation_plots"):
    """
    Print/save per-trajectory metric distribution (per 10% quantile).
    scores: from compute_episode_scores(...)
    metric: 'reward_sum' | 'neg_cost_sum' | 'health_improve'
    tag: train/test/algorithm name
    save_dir: save path
    """
    vals = np.asarray(scores[metric]).astype(float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"[MetricDist] {tag}/{metric}: no data")
        return None

    pcts = list(range(0, 101, 10))
    pct_vals = np.percentile(vals, pcts).tolist()

    # print
    print(f"[MetricDist] {tag}/{metric} distribution (per 10%):")
    for p, v in zip(pcts, pct_vals):
        print(f"  p{p:02d}: {v:.4f}")
    print(f"  mean: {float(np.mean(vals)):.4f}, std: {float(np.std(vals)):.4f}, min: {float(np.min(vals)):.4f}, max: {float(np.max(vals)):.4f}, n={len(vals)}")

    # save
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "tag": tag,
        "metric": metric,
        "percentiles": {f"p{p}": v for p, v in zip(pcts, pct_vals)},
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": int(len(vals))
    }
    out_path = os.path.join(save_dir, f"episode_metric_dist_{tag}_{metric}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[MetricDist] Saved to {out_path}")
    return out

# at Multi_Task_Run_v3.py 中添加测试数据集类

class UnifiedTestDataset:
    """
    Unified test dataset with all test data
    """
    def __init__(self, test_data, test_actions, test_budgets, test_health, test_raw_cost, 
                 test_reward, test_log_budgets, test_legal_actions, test_dones, 
                 action_costs, log_budget_norm_params, max_bridges, state_dim, action_dim,
                 test_actual_n_agents, test_data_with_budgets):  # added参数
        self.test_data = test_data  # [num_eps, T, n_agents, state_dim]
        self.test_actions = test_actions  # [num_eps, T, n_agents]
        self.test_budgets = test_budgets  # [num_eps]
        self.test_health = test_health  # [num_eps, T+1, n_agents]
        self.test_raw_cost = test_raw_cost  # [num_eps, T, n_agents]
        self.test_reward = test_reward  # [num_eps, T, n_agents]
        self.test_log_budgets = test_log_budgets  # [num_eps, T, n_agents]
        self.test_legal_actions = test_legal_actions  # [num_eps, T, n_agents, action_dim]
        self.test_dones = test_dones  # [num_eps, T]
        self.action_costs = action_costs
        self.log_budget_norm_params = log_budget_norm_params
        self.max_bridges = max_bridges
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.test_actual_n_agents = test_actual_n_agents  # [num_eps]
        self.test_data_with_budgets = test_data_with_budgets  # [num_eps, T, n_agents, state_dim+1]
        
        # print数据集信息
        print(f"Unified test dataset created:")
        print(f"  - Test data: {test_data.shape}")
        print(f"  - Test actions: {test_actions.shape}")
        print(f"  - Test budgets: {test_budgets.shape}")
        print(f"  - Test health: {test_health.shape}")
        print(f"  - Test raw cost: {test_raw_cost.shape}")
        print(f"  - Actual agent counts: {test_actual_n_agents}")
    
    def get_actual_n_agents(self):
        """Get actual n_agents per episode"""
        return self.test_actual_n_agents
    
    def get_osrl_data(self):
        """Get test data for OSRL"""
        return {
            'test_data': self.test_data,
            'test_actions': self.test_actions,
            'test_budgets': self.test_budgets,
            'test_log_budgets': self.test_log_budgets,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_marl_data(self, env_info):
        """Get test data for MARL"""
        return {
            'test_data': self.test_data_with_budgets,
            'test_actions': self.test_actions,
            'test_log_budgets': self.test_log_budgets,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_marl_data_without_budget(self, env_info):
        """Get test data for MARL"""
        return {
            'test_data': self.test_data,
            'test_actions': self.test_actions,
            'test_log_budgets': self.test_log_budgets,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_cdt_data(self):
        """Get test data for CDT (keep MARL format)"""
        test_cdt_data, test_cdt_actions, test_cdt_returns_to_go, test_cdt_costs_to_go, test_cdt_time_steps, test_cdt_raw_cost, test_cdt_health = prepare_cdt_dataset(
            self.test_data, self.test_actions, self.test_reward, self.test_raw_cost, 
            health=self.test_health, max_n_agents=self.max_bridges
        )
        return {
            'test_data': test_cdt_data,
            'test_actions': test_cdt_actions,
            'test_returns': test_cdt_returns_to_go,
            'test_costs': test_cdt_costs_to_go,
            'test_time_steps': test_cdt_time_steps,
            'test_raw_cost': test_cdt_raw_cost,
            'test_health': test_cdt_health,
            'max_bridges': self.max_bridges,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_discrete_bc_data(self, env_info):
        """Get test data for DiscreteBC"""
        return {
            'test_data': self.test_data_with_budgets,
            'test_actions': self.test_actions,
            'test_log_budgets': self.test_log_budgets,
            'test_legal_actions': self.test_legal_actions,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_qmix_data(self, env_info):
        """Get test data for QMIX"""
        return {
            'test_data': self.test_data_with_budgets,
            'test_actions': self.test_actions,
            'test_log_budgets': self.test_log_budgets,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    

    def get_qmix_data_without_budget(self, env_info):
        """Get test data for QMIX"""
        return {
            'test_data': self.test_data,
            'test_actions': self.test_actions,
            'test_log_budgets': self.test_log_budgets,
            'test_health': self.test_health,
            'test_raw_cost': self.test_raw_cost,
            'actual_n_agents': self.test_actual_n_agents,
            'log_budget_norm_params': self.log_budget_norm_params,
            'test_episode_budgets': self.test_budgets,
        }
    
    def get_random_data(self, algo_type, env_info=None):
        """Get test data for random baseline"""
        if algo_type == 'osrl':
            return {
                'test_data': self.test_data,
                'test_actions': self.test_actions,
                'test_budgets': self.test_budgets,
                'test_health': self.test_health,
                'test_raw_cost': self.test_raw_cost,
                'actual_n_agents': self.test_actual_n_agents,
                'log_budget_norm_params': self.log_budget_norm_params,
                'test_episode_budgets': self.test_budgets,
            }
        elif algo_type == 'marl':
            return {
                'test_data': self.test_data_with_budgets,
                'test_actions': self.test_actions,
                'test_budgets': self.test_budgets,
                'test_health': self.test_health,
                'test_raw_cost': self.test_raw_cost,
                'max_n_agents': self.max_bridges,
                'env_info': env_info,
                'test_log_budgets': self.test_log_budgets,
                'actual_n_agents': self.test_actual_n_agents,
                'log_budget_norm_params': self.log_budget_norm_params,
                'test_episode_budgets': self.test_budgets,
            }
        else:
            raise ValueError(f"Unknown algo_type: {algo_type}")

#can输入一个算法名称
def main(input_algo_name=None, device_id=3,seed=1024):
    # === config and env ===
    config_path = "paper/benchmark/flows/config.yaml"
    config = load_config(config_path)
    env_info = load_env_info(config['data']['env_info_file'])
    if seed is None:
        seed = config['training']['seed']
    else:
        seed = int(seed)
    set_seed(seed)
    device = th.device(f"cuda:{device_id}" if config['hardware']['use_cuda'] and th.cuda.is_available() else "cpu")
    print(f"Is CUDA available? {th.cuda.is_available()}")
    print(f"Number of GPUs: {th.cuda.device_count()}")


    # === load ReplayBuffer ===
    buffer = load_buffer(config['data']['buffer_file'], device)
    data, actions, episode_budgets, health, raw_cost ,reward,log_budgets,actual_n_agents= extract_data_from_buffer(buffer)
    state_dim = env_info['obs_shape']
    action_dim = env_info['n_actions']
    max_bridges=env_info['max_bridges']


    #print(f"log_budgets的shape为{log_budgets.shape}")
    #exit(0)

    # === build legal_actions ===
    action_costs = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}
    log_budget_norm_params = env_info['normalization_params']['log_budgets']
    legal_actions = build_legal_actions_from_log_budgets(
        log_budgets, action_costs, log_budget_norm_params
    )
    # === dones ===
    dones = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    dones[:, -1] = 1.0

    # load测试集buffer
    test_buffer = load_buffer(config['data']['test_buffer_file'], device)
    test_data, test_actions, test_budgets ,test_health, test_raw_cost,test_reward ,test_log_budgets,test_actual_n_agents= extract_data_from_buffer(test_buffer)

    # take first 20 episodes
    N = 20
    test_data = test_data[:N]
    test_actions = test_actions[:N]
    test_budgets = test_budgets[:N]
    test_health = test_health[:N]  # add这一行
    test_raw_cost = test_raw_cost[:N]  # add这一行
    test_reward = test_reward[:N]  # add这一行
    test_log_budgets = test_log_budgets[:N]  # add这一行
    test_actual_n_agents = test_actual_n_agents[:N]
    test_legal_actions = build_legal_actions_from_log_budgets(
        test_log_budgets, action_costs, log_budget_norm_params
    )
    test_dones = np.zeros((test_data.shape[0], test_data.shape[1]), dtype=np.float32)
    test_dones[:, -1] = 1.0


    # === prepare augmented training data for MARL ===
    train_data_with_budgets = augment_data_with_budgets(
        data=data,
        budgets=log_budgets,
        budget_mode='provided'
    )
    
    # === prepare augmented test data for MARL ===
    test_data_with_budgets = augment_data_with_budgets(
        data=test_data,
        budgets=test_log_budgets,
        budget_mode='provided'
    )

    # === create unified test dataset ===
    test_dataset = UnifiedTestDataset(
        test_data=test_data,
        test_actions=test_actions,
        test_budgets=test_budgets,
        test_health=test_health,
        test_raw_cost=test_raw_cost,
        test_reward=test_reward,
        test_log_budgets=test_log_budgets,
        test_legal_actions=test_legal_actions,
        test_dones=test_dones,
        action_costs=action_costs,
        log_budget_norm_params=log_budget_norm_params,
        max_bridges=max_bridges,
        state_dim=state_dim,
        action_dim=action_dim,
        test_actual_n_agents=test_actual_n_agents,
        test_data_with_budgets=test_data_with_budgets
    )

    # === CDT training data (training only) ===
    cdt_data, cdt_actions, cdt_returns_to_go, cdt_costs_to_go, cdt_time_steps, cdt_raw_cost, cdt_health,_,_ = _prepare_cdt_dataset(
        data, actions, reward, raw_cost, health=health
    )

    # === register algorithms, return training history ===
    algo_dict = {
        "multitask_bc": lambda: train_multitask_bc(
            data, actions, episode_budgets,log_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data()
        ),
        "multitask_bc_top20": lambda: train_multitask_bc(
            data, actions, episode_budgets,log_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data(),
            #expert filter
            expert_percent=0.2,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            action_costs=action_costs,
            rewards=reward,
            raw_cost=raw_cost,
            health=health
        ),
        "multitask_bc_top50": lambda: train_multitask_bc(
            data, actions, episode_budgets,log_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data(),
            #expert filter
            expert_percent=0.5,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            action_costs=action_costs,
            rewards=reward,
            raw_cost=raw_cost,
            health=health
        ),
        "multitask_cpq": lambda: train_multitask_cpq(
            data, actions, episode_budgets, log_budgets, reward, state_dim, action_dim, device,
            num_epochs=30, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data()
        ),
        "multitask_offline_cpq": lambda: train_multitask_offline_cpq(
            data, actions, episode_budgets, log_budgets, reward, state_dim, action_dim, device,
            num_epochs=30, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data()
        ),
        "onestep": lambda: train_multitask_onestep(
            data, actions, episode_budgets, log_budgets, reward, state_dim, action_dim, device,
            num_epochs=40, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data()
        ),
        "cql": lambda: train_multitask_cql(
            data, actions, episode_budgets, log_budgets, reward, state_dim, action_dim, device,
            num_epochs=30, batch_size=1024, eval_interval=5,
            **test_dataset.get_osrl_data()
        ),
        "iqlcql_marl": lambda: train_multitask_iqlcql(
            train_data_with_budgets, actions, reward, dones, legal_actions,
            env_info['max_bridges'], state_dim, action_dim, device,
            num_epochs=100, batch_size=16, eval_interval=10,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_marl_data(env_info)
        ),
        "iqlcql_marl_without_budget": lambda: train_multitask_iqlcql(
            data, actions, reward, dones, legal_actions,
            env_info['max_bridges'], state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_marl_data_without_budget(env_info)
        ),
        "cdt": lambda: train_multitask_cdt(
            cdt_data, cdt_actions, cdt_returns_to_go, cdt_costs_to_go, cdt_time_steps,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            device=device,
            num_epochs=20, batch_size=128, eval_interval=5,
            **test_dataset.get_cdt_data()
        ),
        "discrete_bc": lambda: train_multitask_discrete_bc(
            train_data_with_budgets, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=100, batch_size=16, eval_interval=10,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info)
        ),
        "discrete_bc_20": lambda: train_multitask_discrete_bc(
            train_data_with_budgets, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=100, batch_size=16, eval_interval=10,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info),
            action_costs=action_costs,
            expert_percent=0.2,
            verbose=True,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            train_health=health
        ),
        "discrete_bc_50": lambda: train_multitask_discrete_bc(
            train_data_with_budgets, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=100, batch_size=16, eval_interval=10,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info),
            action_costs=action_costs,
            expert_percent=0.5,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            train_health=health
        ),
        "qmix_cql": lambda: train_multitask_qmixcql(
            train_data_with_budgets, actions, reward, dones, legal_actions,
            max_bridges, state_dim, action_dim, device,
            num_epochs=100, batch_size=16, eval_interval=10,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_qmix_data(env_info)
        ),
        "qmix_cql_without_budget": lambda: train_multitask_qmixcql(
            data, actions, reward, dones, legal_actions,
            max_bridges, state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_qmix_data_without_budget(env_info)
        ),
        # === add random baseline ===
        "random_osrl": lambda: train_random_baseline(
            data, actions, reward, state_dim, action_dim, device,action_costs=action_costs,
            num_epochs=20, eval_interval=4,seed=seed,
            **test_dataset.get_random_data('osrl'),
            algo_type='osrl'
        ),
        "random_marl": lambda: train_random_baseline(
            train_data_with_budgets, actions, reward, state_dim, action_dim, device,action_costs=action_costs,
            num_epochs=20, eval_interval=4,seed=seed,
            **test_dataset.get_random_data('marl', env_info),
            algo_type='marl'
        ),
    }

    if input_algo_name:
        algorithms_to_run = [input_algo_name]
        print(f"using input algorithm: {input_algo_name}")
    else:
    # === specify algorithms to run ===
    #can选择运行全部算法或者指定算法
    #algorithms_to_run = ["random_osrl"]
    #algorithms_to_run = ["random_marl"] 
    #algorithms_to_run = ["cdt"] 
    #algorithms_to_run = ["qmix_cql"]# 
    #algorithms_to_run = ["discrete_bc_20"]# 
        algorithms_to_run = ["discrete_bc","qmix_cql","multitask_bc","random_osrl","iqlcql_marl","onestep","cql","multitask_cpq","multitask_offline_cpq",
        "cdt","discrete_bc_20","discrete_bc_50","multitask_bc_top20","multitask_bc_top50"]# 
    #algorithms_to_run = ["multitask_bc_top20","multitask_bc_top50","discrete_bc_20","discrete_bc_50"] 
    #algorithms_to_run = ["multitask_bc_top5","multitask_bc_top20"]
    #algorithms_to_run = ["discrete_bc_20","discrete_bc_50"] 
    #algorithms_to_run = ["random_marl","iqlcql_marl","discrete_bc","qmix_cql","discrete_bc_20","discrete_bc_50","discrete_bc_5"]  can修改这个列表
    #algorithms_to_run = list(algo_dict.keys())  # run all algorithms
    
    # store results for all algorithms
    all_results = {}
    
    print(f"Starting training for {len(algorithms_to_run)} algorithms: {algorithms_to_run}")
    
    for algo_name in algorithms_to_run:
        print(f"\n{'='*50}")
        print(f"Starting training for algorithm: {algo_name}")
        print(f"{'='*50}")
        
        try:
            # reset seed for reproducibility
            set_seed(seed)
            
            # run algorithm
            model, training_history = algo_dict[algo_name]()
            
            # save训练历史
            save_training_history(training_history, algo_name)
            
            # save训练好的模型
            model_path = save_model(model, algo_name)
            
            # store result
            all_results[algo_name] = {
                'model': model,
                'training_history': training_history
            }
            
            print(f"\nAlgorithm {algo_name} training completed!")
            
        except Exception as e:
            print(f"\nAlgorithm {algo_name} training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # === generate comparison report ===
    print(f"\n{'='*60}")
    print("Training summary")
    print(f"{'='*60}")
    
    # create汇总结果
    summary_results = {}
    
    for algo_name, result in all_results.items():
        training_history = result['training_history']
        
        # get最后一次评估结果
        if training_history['eval_metrics']:
            last_eval = training_history['eval_metrics'][-1]['metrics']
            summary_results[algo_name] = {
                'final_behavioral_similarity': last_eval.get('behavioral_similarity_mean', 0),
                'final_cost': last_eval.get('mean_total_cost', 0),
                'final_violation_rate': last_eval.get('violation_rate_mean', 0),
                'final_health_improve': last_eval.get('bridge_avg_health_gain_vs_history', 0) if 'bridge_avg_health_gain_vs_history' in last_eval else None,
                'num_epochs': len(training_history['epochs']),
                'budget_utilization_mean': last_eval.get('budget_utilization_mean', 0) if 'budget_utilization_mean' in last_eval else None,
                'cost_health_efficiency_mean': last_eval.get('cost_health_efficiency_mean', 0) if 'cost_health_efficiency_mean' in last_eval else None,
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None
            }
        else:
            summary_results[algo_name] = {
                'final_behavioral_similarity': None,
                'final_cost': None,
                'final_violation_rate': None,
                'final_health_improve': None,
                'num_epochs': len(training_history['epochs']),
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None
            }
    
    # print汇总表格
    print(f"{'算法名称':<15} {'准确率':<10} {'总成本':<10} {'违规率':<10} {'健康改善':<10} {'经费使用占比':<10} {'经费健康提升效率':<10}")
    print("-" * 80)
    
    for algo_name, metrics in summary_results.items():
        acc = f"{metrics['final_behavioral_similarity']:.4f}" if metrics['final_behavioral_similarity'] is not None else "N/A"
        cost = f"{metrics['final_cost']:.1f}" if metrics['final_cost'] is not None else "N/A"
        vio = f"{metrics['final_violation_rate']:.4f}" if metrics['final_violation_rate'] is not None else "N/A"
        health = f"{metrics['final_health_improve']:.4f}" if metrics['final_health_improve'] is not None else "N/A"
        loss = f"{metrics['final_loss']:.4f}" if metrics['final_loss'] is not None else "N/A"
        budget = f"{metrics['budget_utilization_mean']:.4f}" if metrics['budget_utilization_mean'] is not None else "N/A"
        cost_health = f"{metrics['cost_health_efficiency_mean']:.4f}" if metrics['cost_health_efficiency_mean'] is not None else "N/A"

        
        print(f"{algo_name:<15} {acc:<10} {cost:<10} {vio:<10} {health:<10} {budget:<10} {cost_health:<10}")
    
    # save汇总结果
    save_metrics(summary_results, "comparison_summary")
    
    print(f"\n所有结果已保存到 metrics_results/ 和 training_results/ 目录")
    print("可以使用保存的数据绘制训练曲线和对比图表")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results = main()
    elif len(sys.argv) < 3:
        results = main(sys.argv[1], sys.argv[2])
    else:
        results = main(sys.argv[1], sys.argv[2],sys.argv[3])
    