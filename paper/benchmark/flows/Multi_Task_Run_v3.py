import os
import sys
import torch as th
import numpy as np
import yaml
import json
import random

from algos.MultiTaskBC import MultiTaskBC
from algos.CPQDiscreteMultiTask import CPQDiscreteMultiTask
from algos.IQLCQLMultiAgent import IQLCQLMultiAgent
from algos.CDT import CDT
from algos.DiscreteBCMultiAgent import DiscreteBCMultiAgent
from algos.QMIXCQLMultiAgent import QMIXCQLMultiAgent
from algos.RandomBaseline import RandomBaselineOSRL, RandomBaselineMARL
import torch.nn.functional as F  # CDT 训练用
from evaluate_unified_v2 import evaluate_unified
from evaluate_osrl import evaluate_osrl
from evaluate_marl import evaluate_marl, evaluate_marl_osrl_aligned
from evaluate_marl_unified import evaluate_marl_unified
from evaluate_cdt import evaluate_cdt_parallel
from datetime import datetime

# ================= 数据准备工具 =================

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

def save_metrics(metrics, algo_name, out_dir="marl/new_module/metrics_results"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo_name}_{dt_str}.json"
    
    # 转换为可序列化的格式
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"[INFO] Metrics for {algo_name} saved to {fname}")

def save_training_history(training_history, algo_name, out_dir="marl/new_module/training_results"):
    """保存训练历史数据"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{algo_name}_training_history_{dt_str}.json"
    
    # 转换为可序列化的格式
    serializable_history = convert_to_serializable(training_history)
    
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"[INFO] Training history for {algo_name} saved to {fname}")

def save_model(model, algo_name, out_dir="marl/new_module/saved_models"):
    """保存训练好的完整模型"""
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
    
    data = np.stack(all_obs)        # [num_eps, T, n_agents, state_dim]
    actions = np.stack(all_actions) # [num_eps, T, n_agents] 或 [num_eps, T, n_agents, 1]
    episode_budgets = np.array(episode_budgets).squeeze()
    health = np.stack(all_health) if all_health else None
    raw_cost = np.stack(all_raw_cost) if all_raw_cost else None
    rewards = np.stack(all_rewards) if all_rewards else None
    log_budgets = np.stack(all_log_budgets) if all_log_budgets else None
    actual_n_agents = np.array(actual_n_agents_list)  # [num_eps] - 现在应该是正确的形状

    print(f'产生的ep的长度为{data.shape}')

    return data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents


def compute_episode_scores(data, actions, rewards=None, raw_cost=None, health=None, action_costs=None):
    """
    计算每个 episode 的若干评分指标用于专家轨迹筛选。
    返回: dict，其中包含每个指标对应的 [num_eps] 数组。
    指标:
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
    根据指定指标选择前 top_percent 的 episode 索引。
    scores: compute_episode_scores 的返回值
    metric: 指标名（如 'reward_sum' | 'neg_cost_sum' | 'health_improve'）
    top_percent: 0~1，选取前多少百分比
    higher_is_better: True 表示分数越高越好
    返回: numpy 数组的索引（升序）
    """
    assert 0 < top_percent <= 1.0, "top_percent 必须在 (0,1] 内"
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
    将变长n_agents的episode pad到max_n_agents，并生成agent_mask
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

        # 通用squeeze（最后一维为1时自动去掉）
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
            l = np.squeeze(l, axis=-2)  # 极少见，冗余pad
        legal_arr[i, :, :n_agents, :] = l

        agent_mask[i, :n_agents] = 1.
    return obs_arr, actions_arr, rewards_arr, dones, legal_arr, agent_mask


# =================== 训练函数（修改版，返回训练历史） ===================

def train_multitask_iqlcql(
    data, actions, rewards, dones, legal_actions, max_n_agents, state_dim, action_dim, device='cpu',
    num_epochs=20, batch_size=16, eval_interval=2,
    log_budgets=None, env_info=None, raw_cost=None,
    test_data=None, test_actions=None, test_log_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None
):
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
        obs_dim=state_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device
    )

    # 训练历史记录
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
            losses.append(float(loss_dict['loss']))  # 确保转换为 Python float
        
        avg_loss = float(np.mean(losses))  # 确保转换为 Python float
        print(f"[IQLCQL-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))  # 确保是 Python int
        training_history['losses'].append(avg_loss)

        # === 训练中评估 ===
        if ((epoch % eval_interval == 0 or epoch == num_epochs) and
            test_data is not None and env_info is not None):
            print(f"--- Evaluate at Epoch {epoch} ---")
            '''metrics = evaluate_marl_osrl_aligned(
                model,
                test_data,
                test_actions,
                budgets=test_log_budgets,                 # 或 budgets_provided
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                verbose=False,
                budget_mode='provided',                # 或 'provided'
                uniform_active_only=False
            )'''
            # 在 train_multitask_iqlcql 等函数中
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_log_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                verbose=True,
                use_agent_mask=True,  # 多智能体算法只对激活智能体评估
                budget_mode='provided',
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets
            )
            print(metrics)
            # 记录评估数据（确保数据类型转换）
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
        total_budget = budgets[ep]
        per_step_agent_budget = float(total_budget) / (T * n_agents)
        for t in range(T):
            done_flag = 1.0 if t == T-1 else 0.0
            for agent in range(n_agents):
                obs = data[ep, t, agent]
                act = actions[ep, t, agent]
                budget =  per_step_agent_budget
                # next_obs
                if t < T-1:
                    next_obs = data[ep, t+1, agent]
                    next_budget = per_step_agent_budget
                else:
                    next_obs = np.zeros_like(obs)
                    next_budget = 0.0
                obs_list.append(obs)
                budget_list.append(np.log(float(budget) + 1e-8)/10)
                action_list.append(int(act))
                # ====== 关键分离 ======
                if rewards is not None:
                    reward = float(rewards[ep, t, agent])
                elif reward_func:
                    reward = reward_func(obs, act)
                else:
                    reward = 0.0  # 或 raise 错误
                reward_list.append(reward)
                next_obs_list.append(next_obs)
                next_budget_list.append(np.log(float(next_budget) + 1e-8)/10)
                # cost 只来自log_budgets
                cost_list.append(float(log_budgets[ep, t, agent]))
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


def train_multitask_cpq(
    data, actions, budgets, log_budgets,reward, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None
):
    dataset = prepare_cpq_dataset(data, actions, budgets, log_budgets,reward)
    N = len(dataset['obs'])
    # 转为tensor
    for k in dataset:
        dataset[k] = th.tensor(dataset[k], dtype=th.float32 if dataset[k].ndim > 1 or k not in ['actions'] else th.long, device=device)
    # 模型初始化
    model = CPQDiscreteMultiTask(
        state_dim=state_dim, n_actions=action_dim, device=device
    )
    optimizer_q = th.optim.Adam(model.q_net.parameters(), lr=1e-3)
    optimizer_qc = th.optim.Adam(model.qc_net.parameters(), lr=1e-3)

    # 训练历史记录
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
            losses.append(float(loss_q))  # 确保转换为 Python float
            losses_c.append(float(loss_qc))
        
        avg_loss_q = float(np.mean(losses))
        avg_loss_qc = float(np.mean(losses_c))
        print(f"[CPQ] Epoch {epoch}, Q Loss: {avg_loss_q:.4f}, QC Loss: {avg_loss_qc:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))
        training_history['q_losses'].append(avg_loss_q)
        training_history['qc_losses'].append(avg_loss_qc)
        training_history['losses'].append(avg_loss_q + avg_loss_qc)

        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # 评估：注意这里要实现适配 evaluate_osrl 的 model.act 接口
            #metrics = evaluate_osrl(model, test_data, test_actions, budgets=test_budgets,health=test_health, verbose=True)
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_budgets,
                health=test_health,
                action_costs=None,
                raw_cost=test_raw_cost,
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,  # OSRL算法对所有智能体评估
                budget_mode='uniform'
            )
            print(f"[CPQ] Eval @ epoch {epoch}: acc={metrics['acc_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else ""))
            # 记录评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history

def train_multitask_bc(
    data,actions, budgets, state_dim, action_dim,
    device='cpu', num_epochs=20, batch_size=1024, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None,test_health=None,
    actual_n_agents=None,
    # 专家轨迹筛选（仅训练集）
    expert_percent=None,
    expert_metric='reward_sum',
    expert_higher_is_better=True,
    action_costs=None,
    rewards=None,
    raw_cost=None,
    health=None,
    log_budget_norm_params=None,
    test_episode_budgets=None
):
    """
    data: [num_episodes, T, n_agents, state_dim]
    actions: [num_episodes, T, n_agents]
    budgets: [num_episodes]
    """
    # 可选：专家轨迹筛选（按 episode 粒度）
    if expert_percent is not None:
        try:

            score_dict = compute_episode_scores(
                data, actions, rewards=rewards, raw_cost=raw_cost, health=health, action_costs=action_costs
            )
            summarize_episode_metric_distribution(score_dict, metric=expert_metric, tag="multitask_bc_train")
            idx_sel = select_expert_indices(
                score_dict, metric=expert_metric, top_percent=float(expert_percent), higher_is_better=expert_higher_is_better
            )
            print(f"[MultiTaskBC] 使用专家轨迹: metric={expert_metric}, top_percent={expert_percent}, 选中 {len(idx_sel)}/{len(data)} 条")
            print(f"data: {data.shape}, actions: {actions.shape}, budgets: {budgets.shape}, rewards: {rewards.shape}, raw_cost: {raw_cost.shape}, health: {health.shape}")
            
            data = data[idx_sel]
            actions = actions[idx_sel]
            budgets = budgets[idx_sel]
            # 附带：若用户提供 rewards/raw_cost/health 也同步裁剪，避免后续可能使用
            if rewards is not None:
                rewards = rewards[idx_sel]
            if raw_cost is not None:
                raw_cost = raw_cost[idx_sel]
            if health is not None:
                health = health[idx_sel]
        except Exception as e:
            print(f"[MultiTaskBC] 专家筛选失败，回退到全量数据。原因: {e}")

    obs_list, budget_list, action_list = [], [], []
    num_episodes, T, n_agents, state_dim = data.shape
    for ep in range(num_episodes):
        total_budget = budgets[ep]
        per_step_agent_budget = float(total_budget) / (T * n_agents)
        for t in range(T):
            for agent in range(n_agents):
                obs = data[ep, t, agent]
                act = actions[ep, t, agent]
                budget = per_step_agent_budget
                obs_list.append(obs)
                budget_list.append(float(budget))  # 保证是标量
                action_list.append(int(act))       # 保证是标量

    obs_arr = th.tensor(np.stack(obs_list), dtype=th.float32)         # (N, state_dim)
    budget_arr = th.tensor(np.array(budget_list), dtype=th.float32)   # (N,)
    action_arr = th.tensor(np.array(action_list), dtype=th.long)      # (N,)

    model = MultiTaskBC(state_dim=state_dim, action_dim=action_dim, device=device)
    model.setup_optimizers(actor_lr=1e-3)

    # 训练历史记录
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
            losses.append(float(loss_actor.item()))  # 确保转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[MultiTaskBC] Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)
        
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None:
            # 评估：注意这里要实现适配 evaluate_osrl 的 model.act 接口
            #metrics = evaluate_osrl(model, test_data, test_actions, budgets=test_budgets,health=test_health, verbose=True)
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=test_budgets,
                health=test_health,
                action_costs=action_costs,
                raw_cost=test_raw_cost,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets,
                verbose=True,
                use_agent_mask=False,  # OSRL算法对所有智能体评估
                budget_mode='uniform',
                actual_n_agents=actual_n_agents
            )
            print(f"[MultiTaskBC] Eval @ epoch {epoch}: acc={metrics['acc_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else ""))
            # 记录评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history


def prepare_cdt_dataset(data, actions, rewards, raw_cost, health=None, max_n_agents=None):
    """
    修改后的CDT数据集准备函数，保持多智能体格式
    
    data:     [num_episodes, T, n_agents, state_dim]
    actions:  [num_episodes, T, n_agents]
    rewards:  [num_episodes, T, n_agents]
    raw_cost: [num_episodes, T, n_agents, 1] or [num_episodes, T, n_agents] or None
    health:   [num_episodes, T+1, n_agents] or None
    max_n_agents: 最大智能体数量，用于padding

    返回：
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
    
    # 初始化输出数组
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
            
            # 动作数据 - 转换为one-hot
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
            
            # 健康数据 - 修复时间维度不匹配问题
            if health is not None:
                if health.shape[1] == T:  # health是[T, n_agents]格式
                    # 补全最后一列：复制上一年（第T-1年）的健康状态
                    health_ep_agent = health[ep, :, agent]  # [T] 或 [T, 1]
                    if health_ep_agent.ndim > 1:
                        health_ep_agent = health_ep_agent.squeeze(-1)  # 确保是 [T] 格式
                    health_arr[ep, :T, agent] = health_ep_agent
                    health_arr[ep, T, agent] = health_ep_agent[-1]  # 复制最后一年的健康状态
                elif health.shape[1] == T+1:  # health已经是[T+1, n_agents]格式
                    health_ep_agent = health[ep, :, agent]  # [T+1] 或 [T+1, 1]
                    if health_ep_agent.ndim > 1:
                        health_ep_agent = health_ep_agent.squeeze(-1)  # 确保是 [T+1] 格式
                    health_arr[ep, :, agent] = health_ep_agent
                else:
                    raise ValueError(f"health的时间维度不正确: {health.shape[1]}, 期望 {T} 或 {T+1}")
    
    return obs_arr, acts_arr, returns_to_go, costs_to_go, time_steps, raw_cost_arr, health_arr

def _prepare_cdt_dataset(data, actions, rewards, raw_cost, health=None):
    """
    data:     [num_episodes, T, n_agents, state_dim]
    actions:  [num_episodes, T, n_agents]
    rewards:  [num_episodes, T, n_agents]
    raw_cost: [num_episodes, T, n_agents, 1] or [num_episodes, T, n_agents] or None
    health:   [num_episodes, T+1, n_agents] or None

    返回：
        obs_arr:         [N', T, state_dim]
        acts_arr:        [N', T, action_dim]
        returns_to_go:   [N', T]
        costs_to_go:     [N', T]
        time_steps:      [N', T]
        raw_cost_arr:    [N', T]
        health_arr:      [N', T+1] or None
        episode_idx_arr: [N',]      # 新增
        agent_idx_arr:   [N',]      # 新增
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
            # 修正 raw_cost 形状
            if raw_cost is not None:
                costs = raw_cost[ep, :, agent]     # [T, 1] or [T]
                if costs.ndim > 1:
                    costs = costs.squeeze(-1)      # [T]
            else:
                costs = np.zeros_like(rews)
            # return-to-go: 右累加
            returns_to_go = rews[::-1].cumsum()[::-1]
            costs_to_go = costs[::-1].cumsum()[::-1]
            time_steps = np.arange(T)
            # 动作若 shape (T, 1)，转 (T,)
            if acts.ndim > 1:
                acts = acts.squeeze(-1)
            # 动作转 one-hot，统一 pad 到 max_adim，后处理
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

    # 统一 action_dim
    max_adim = max(a.shape[1] for a in act_list)
    for i in range(len(act_list)):
        if act_list[i].shape[1] < max_adim:
            pad = np.zeros((act_list[i].shape[0], max_adim - act_list[i].shape[1]))
            act_list[i] = np.concatenate([act_list[i], pad], axis=1)
    # 转为 numpy
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
    test_health=None, test_raw_cost=None,max_bridges=None,test_episode_idx=None,    # 新增
    test_agent_idx=None,  actual_n_agents=None, log_budget_norm_params=None, test_episode_budgets=None
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

    

    # 训练历史记录
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
            losses.append(float(loss.item()))  # 确保转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[CDT] Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === 测试集评估 ===
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
                episode_idx_arr=test_episode_idx,   # 新增
                agent_idx_arr=test_agent_idx,       # 新增
                verbose=False
            )'''
            metrics = evaluate_unified(
                model=model,
                data=test_data,
                actions=test_actions,
                budgets=None,
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
                test_episode_budgets=test_episode_budgets
            )
            print(
                f"[CDT] Eval @ epoch {epoch}: "
                f"acc={metrics['acc_mean']:.4f}, "
                f"cost={metrics['mean_total_cost']:.1f}, "
                f"vio={metrics['violation_rate_mean']:.3f}"
                + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else "")
            )
            # 记录评估数据
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
    # 专家轨迹筛选开关（仅影响训练集）
    expert_percent=None,                 # 例如 0.5 表示前 50% 作为专家
    expert_metric='reward_sum',          # 可选: 'reward_sum' | 'neg_cost_sum' | 'health_improve'
    expert_higher_is_better=True,         # True 表示分数越高越好
    train_health=None,
):
    # === 可选：专家轨迹筛选（仅训练集） ===
    if expert_percent is not None:
        try:
            score_dict = compute_episode_scores(
                data, actions, rewards=rewards, raw_cost=None, health=train_health, action_costs=action_costs
            )
            summarize_episode_metric_distribution(score_dict, metric=expert_metric, tag="discrete_bc_train")
            idx_sel = select_expert_indices(
                score_dict, metric=expert_metric, top_percent=float(expert_percent), higher_is_better=expert_higher_is_better
            )
            print(f"[DiscreteBC-MA] 使用专家轨迹: metric={expert_metric}, top_percent={expert_percent}, 选中 {len(idx_sel)}/{len(data)} 条")

            if verbose:
                print(f"[DiscreteBC-MA] 使用专家轨迹: metric={expert_metric}, top_percent={expert_percent}, 选中 {len(idx_sel)}/{len(data)} 条")
            # 依据选择的 episode 索引裁剪训练集
            data = data[idx_sel]
            actions = actions[idx_sel]
            rewards = rewards[idx_sel] if rewards is not None else None
            dones = dones[idx_sel]
            legal_actions = legal_actions[idx_sel]
        except Exception as e:
            print(f"[DiscreteBC-MA] 专家筛选失败，回退到全量数据。原因: {e}")
    # === 训练集批量对齐 ===
    obs_arr, actions_arr, rewards_arr, dones_arr, legal_arr, agent_mask = build_ma_batches(
        data, actions, rewards, dones, legal_actions, max_n_agents
    )
    obs_arr = th.tensor(obs_arr, dtype=th.float32, device=device)
    actions_arr = th.tensor(actions_arr, dtype=th.long, device=device)
    agent_mask = th.tensor(agent_mask, dtype=th.float32, device=device)
    num_eps, T, N, _ = obs_arr.shape


    model = DiscreteBCMultiAgent(
        obs_dim=state_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device
    )

    # 训练历史记录
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
            losses.append(float(loss_dict['policy_loss']))  # 确保转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[DiscreteBC-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === 评估 ===
        if (epoch % eval_interval == 0 or epoch == num_epochs) and test_data is not None and env_info is not None:
            # 按 evaluate_marl 需求，直接用原始 test_data, test_actions, test_log_budgets
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
                    use_agent_mask=True,  # 多智能体算法只对激活智能体评估
                    budget_mode='provided',
                    log_budget_norm_params=log_budget_norm_params,
                    test_episode_budgets=test_episode_budgets
                )
                print(f"[DiscreteBC-MA] Eval @ epoch {epoch}: acc={metrics['acc_mean']:.4f}, "
                      f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                      + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else ""))
                # 记录评估数据
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
    # === 1. 组batch ===
    obs_arr, actions_arr, rewards_arr, dones_arr, legal_arr, agent_mask = build_ma_batches(
        data, actions, rewards, dones, legal_actions, max_n_agents
    )
    # print("legal_arr.shape (from build_ma_batches):", legal_arr.shape)
    # === 2. 补齐 obs_arr 到 T+1 ===
    if obs_arr.shape[1] == actions_arr.shape[1]:
        last_obs = obs_arr[:, -1:, :, :]
        obs_arr = np.concatenate([obs_arr, last_obs], axis=1)

    # === 3. 统一shape参数 ===
    B, Tp1, N, O = obs_arr.shape
    T = Tp1 - 1

    # === 4. 检查所有arr的shape ===
    assert actions_arr.shape[1] == T, f"actions_arr T不对，期望{T}，实际{actions_arr.shape[1]}"
    assert rewards_arr.shape[1] == T
    assert dones_arr.shape[1] == T
    assert legal_arr.shape[1] == T

    # === 5. 转为tensor ===
    obs_arr = th.tensor(obs_arr, dtype=th.float32, device=device)
    actions_arr = th.tensor(actions_arr, dtype=th.long, device=device)
    rewards_arr = th.tensor(rewards_arr, dtype=th.float32, device=device)
    dones_arr = th.tensor(dones_arr, dtype=th.float32, device=device)
    legal_arr = th.tensor(legal_arr, dtype=th.bool, device=device)
    agent_mask = th.tensor(agent_mask, dtype=th.float32, device=device)

    # === 6. QMIXCQLMultiAgent 初始化 ===
    model = QMIXCQLMultiAgent(
        obs_dim=obs_dim,
        max_n_agents=max_n_agents,
        action_dim=action_dim,
        device=device,
        state_dim=max_n_agents * obs_dim
    )

    # 训练历史记录
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
            # 强制用外层T/N/O
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
            losses.append(float(loss_dict['loss']))  # 确保转换为 Python float
        
        avg_loss = float(np.mean(losses))
        print(f"[QMIXCQL-MA] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # 记录训练数据
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(avg_loss)

        # === 评估 ===
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
                use_agent_mask=True,  # 多智能体算法只对激活智能体评估
                budget_mode='provided',
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets
            )
            print(f"[QMIX-CQL] Eval @ epoch {epoch}: acc={metrics['acc_mean']:.4f}, "
                      f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                      + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else ""))
            # 记录评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history


# 添加随机算法的训练函数
def train_random_baseline(
    data, actions, rewards, state_dim, action_dim, device='cpu',action_costs=None,
    num_epochs=20, eval_interval=2,
    test_data=None, test_actions=None, test_budgets=None, test_raw_cost=None, test_health=None,
    algo_type='osrl',log_budget_norm_params=None, test_episode_budgets=None, **kwargs,
):
    """
    随机基线算法"训练"（实际上只是创建模型并进行评估）
    
    Args:
        algo_type: 'osrl' 或 'marl'
    """
    # 根据算法类型创建相应的随机基线
    if algo_type == 'osrl':
        model = RandomBaselineOSRL(
            state_dim=state_dim, 
            action_dim=action_dim, 
            device=device, 
            seed=42
        )
    elif algo_type == 'marl':
        max_n_agents = kwargs.get('max_n_agents', 500)
        model = RandomBaselineMARL(
            obs_dim=state_dim,
            max_n_agents=max_n_agents,
            action_dim=action_dim,
            device=device,
            seed=42,
            use_legal_constraint=False
        )
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    
    # 训练历史记录
    training_history = {
        'epochs': [],
        'losses': [],
        'eval_metrics': []
    }
    
    print(f"[Random-{algo_type.upper()}] 随机基线算法不需要训练，直接进行评估...")
    
    # 定期评估（模拟训练过程）
    for epoch in range(eval_interval, num_epochs + 1, eval_interval):
        # 随机算法没有真正的损失，记录为0
        training_history['epochs'].append(int(epoch))
        training_history['losses'].append(0.0)
        
        if test_data is not None:
            print(f"--- Evaluate Random Baseline at Epoch {epoch} ---")
            actual_n_agents = kwargs.get('actual_n_agents')
            # 根据算法类型选择相应的评估函数
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
                use_agent_mask=False,  # OSRL算法对所有智能体评估
                budget_mode='uniform',
                actual_n_agents=actual_n_agents,
                log_budget_norm_params=log_budget_norm_params,
                test_episode_budgets=test_episode_budgets

            )
            elif algo_type == 'marl':
                
                env_info = kwargs.get('env_info')
                test_log_budgets = kwargs.get('test_log_budgets')
                # 对齐到 OSRL 单智能体评估口径
                metrics = evaluate_unified(
                    model=model,
                    data=test_data,
                    actions=test_actions,
                    budgets=test_log_budgets,
                    health=test_health,
                    action_costs=action_costs,
                    raw_cost=test_raw_cost,
                    verbose=True,
                    use_agent_mask=True,  # 多智能体算法只对激活智能体评估
                    budget_mode='provided',
                    actual_n_agents=actual_n_agents,
                    log_budget_norm_params=log_budget_norm_params,
                    test_episode_budgets=test_episode_budgets
                )
            
            print(f"[Random-{algo_type.upper()}] Eval @ epoch {epoch}: acc={metrics['acc_mean']:.4f}, "
                  f"cost={metrics['mean_total_cost']:.1f}, vio={metrics['violation_rate_mean']:.3f}"
                  + (f", health_improve={metrics['health_improve_mean']:.3f}" if 'health_improve_mean' in metrics else ""))
            
            # 记录评估数据
            eval_record = {
                'epoch': int(epoch),
                'metrics': convert_to_serializable(metrics)
            }
            training_history['eval_metrics'].append(eval_record)
    
    return model, training_history

def summarize_episode_metric_distribution(scores, metric, tag="train", save_dir="marl/new_module/evaluation_plots"):
    """
    打印并可保存每条轨迹在指定metric上的分布(每10%分位)。
    scores: compute_episode_scores(...) 的返回
    metric: 'reward_sum' | 'neg_cost_sum' | 'health_improve'
    tag:    标记是train/test/某算法名
    save_dir: 保存路径
    """
    vals = np.asarray(scores[metric]).astype(float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"[MetricDist] {tag}/{metric}: 空数据")
        return None

    pcts = list(range(0, 101, 10))
    pct_vals = np.percentile(vals, pcts).tolist()

    # 打印
    print(f"[MetricDist] {tag}/{metric} 分布(每10%):")
    for p, v in zip(pcts, pct_vals):
        print(f"  p{p:02d}: {v:.4f}")
    print(f"  mean: {float(np.mean(vals)):.4f}, std: {float(np.std(vals)):.4f}, min: {float(np.min(vals)):.4f}, max: {float(np.max(vals)):.4f}, n={len(vals)}")

    # 保存
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
    print(f"[MetricDist] 保存到 {out_path}")
    return out

# 在 Multi_Task_Run_v3.py 中添加测试数据集类

class UnifiedTestDataset:
    """
    统一的测试数据集，包含所有测试相关的数据
    """
    def __init__(self, test_data, test_actions, test_budgets, test_health, test_raw_cost, 
                 test_reward, test_log_budgets, test_legal_actions, test_dones, 
                 action_costs, log_budget_norm_params, max_bridges, state_dim, action_dim,
                 test_actual_n_agents):  # 新增参数
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
        self.test_actual_n_agents = test_actual_n_agents  # [num_eps] - 每个episode的实际智能体数量
        
        # 打印数据集信息
        print(f"统一测试数据集创建完成:")
        print(f"  - 测试数据: {test_data.shape}")
        print(f"  - 测试动作: {test_actions.shape}")
        print(f"  - 测试预算: {test_budgets.shape}")
        print(f"  - 测试健康: {test_health.shape}")
        print(f"  - 测试成本: {test_raw_cost.shape}")
        print(f"  - 实际智能体数量: {test_actual_n_agents}")
    
    def get_actual_n_agents(self):
        """获取每个episode的实际智能体数量"""
        return self.test_actual_n_agents
    
    def get_osrl_data(self):
        """获取OSRL算法需要的测试数据"""
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
    
    def get_marl_data(self, env_info):
        """获取MARL算法需要的测试数据"""
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
        """获取CDT算法需要的测试数据（保持多智能体格式）"""
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
        """获取DiscreteBC算法需要的测试数据"""
        return {
            'test_data': self.test_data,
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
        """获取QMIX算法需要的测试数据"""
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
        """获取随机算法需要的测试数据"""
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
                'test_data': self.test_data,
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


def main():
    # === 配置和环境信息 ===
    config_path = "marl/new_module/config.yaml"
    config = load_config(config_path)
    env_info = load_env_info(config['data']['env_info_file'])
    set_seed(config['training']['seed'])
    device = th.device("cuda" if config['hardware']['use_cuda'] and th.cuda.is_available() else "cpu")

    # === 加载ReplayBuffer ===
    buffer = load_buffer(config['data']['buffer_file'], device)
    data, actions, episode_budgets, health, raw_cost ,reward,log_budgets,actual_n_agents= extract_data_from_buffer(buffer)
    state_dim = env_info['obs_shape']
    action_dim = env_info['n_actions']
    max_bridges=env_info['max_bridges']

    # === 构造 legal_actions ===
    action_costs = {0: 0, 1: 71.56, 2: 1643.31, 3: 2433.53}
    log_budget_norm_params = env_info['normalization_params']['log_budgets']
    legal_actions = build_legal_actions_from_log_budgets(
        log_budgets, action_costs, log_budget_norm_params
    )
    # === dones ===
    dones = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    dones[:, -1] = 1.0

    # 加载测试集buffer
    test_buffer = load_buffer(config['data']['test_buffer_file'], device)
    test_data, test_actions, test_budgets ,test_health, test_raw_cost,test_reward ,test_log_budgets,test_actual_n_agents= extract_data_from_buffer(test_buffer)

    # 只取前20个 episode
    N = 20
    test_data = test_data[:N]
    test_actions = test_actions[:N]
    test_budgets = test_budgets[:N]
    test_health = test_health[:N]  # 添加这一行
    test_raw_cost = test_raw_cost[:N]  # 添加这一行
    test_reward = test_reward[:N]  # 添加这一行
    test_log_budgets = test_log_budgets[:N]  # 添加这一行
    test_actual_n_agents = test_actual_n_agents[:N]
    test_legal_actions = build_legal_actions_from_log_budgets(
        test_log_budgets, action_costs, log_budget_norm_params
    )
    test_dones = np.zeros((test_data.shape[0], test_data.shape[1]), dtype=np.float32)
    test_dones[:, -1] = 1.0

    # === 创建统一测试数据集 ===
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
        test_actual_n_agents=test_actual_n_agents
    )

    # === CDT训练数据准备（仅用于训练） ===
    cdt_data, cdt_actions, cdt_returns_to_go, cdt_costs_to_go, cdt_time_steps, cdt_raw_cost, cdt_health,_,_ = _prepare_cdt_dataset(
        data, actions, reward, raw_cost, health=health
    )

    # === 多算法注册，返回训练历史 ===
    algo_dict = {
        "multitask_bc": lambda: train_multitask_bc(
            data, actions, episode_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=2,
            **test_dataset.get_osrl_data()
        ),
        "multitask_bc_top20": lambda: train_multitask_bc(
            data, actions, episode_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=2,
            **test_dataset.get_osrl_data(),
            # 专家筛选
            expert_percent=0.2,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            action_costs=action_costs,
            rewards=reward,
            raw_cost=raw_cost,
            health=health
        ),
        "multitask_bc_top50": lambda: train_multitask_bc(
            data, actions, episode_budgets, state_dim, action_dim, device,
            num_epochs=20, batch_size=1024, eval_interval=2,
            **test_dataset.get_osrl_data(),
            # 专家筛选
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
            num_epochs=20, batch_size=1024, eval_interval=2,
            **test_dataset.get_osrl_data()
        ),
        "iqlcql_marl": lambda: train_multitask_iqlcql(
            data, actions, reward, dones, legal_actions,
            env_info['max_bridges'], state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_marl_data(env_info)
        ),
        "cdt": lambda: train_multitask_cdt(
            cdt_data, cdt_actions, cdt_returns_to_go, cdt_costs_to_go, cdt_time_steps,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            device=device,
            num_epochs=20, batch_size=128, eval_interval=2,
            **test_dataset.get_cdt_data()
        ),
        "discrete_bc": lambda: train_multitask_discrete_bc(
            data, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info)
        ),
        "discrete_bc_20": lambda: train_multitask_discrete_bc(
            data, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info),
            action_costs=action_costs,
            expert_percent=0.2,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            train_health=health
        ),
        "discrete_bc_50": lambda: train_multitask_discrete_bc(
            data, actions, reward, dones, legal_actions, max_bridges, state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,log_budgets=log_budgets,env_info=env_info,
            **test_dataset.get_discrete_bc_data(env_info),
            action_costs=action_costs,
            expert_percent=0.5,
            expert_metric='reward_sum',
            expert_higher_is_better=True,
            train_health=health
        ),
        "qmix_cql": lambda: train_multitask_qmixcql(
            data, actions, reward, dones, legal_actions,
            max_bridges, state_dim, action_dim, device,
            num_epochs=20, batch_size=16, eval_interval=2,
            log_budgets=log_budgets, env_info=env_info, raw_cost=raw_cost,
            **test_dataset.get_qmix_data(env_info)
        ),
        # === 添加随机基线算法 ===
        "random_osrl": lambda: train_random_baseline(
            data, actions, reward, state_dim, action_dim, device,action_costs=action_costs,
            num_epochs=20, eval_interval=4,
            **test_dataset.get_random_data('osrl'),
            algo_type='osrl'
        ),
        "random_marl": lambda: train_random_baseline(
            data, actions, reward, state_dim, action_dim, device,action_costs=action_costs,
            num_epochs=20, eval_interval=4,
            **test_dataset.get_random_data('marl', env_info),
            algo_type='marl'
        ),
    }

    # === 指定要运行的算法列表 ===
    # 可以选择运行全部算法或者指定算法
    #algorithms_to_run = ["random_osrl"]
    #algorithms_to_run = ["random_marl"] 
    #algorithms_to_run = ["cdt"] 
    #algorithms_to_run = ["iqlcql_marl"] 
    #algorithms_to_run = ["multitask_bc_top20","multitask_bc_top50","discrete_bc_20","discrete_bc_50"] 
    #algorithms_to_run = ["multitask_bc_top5","multitask_bc_top20"]
    #algorithms_to_run = ["discrete_bc_20","discrete_bc_50"] 
    #algorithms_to_run = ["random_marl","iqlcql_marl","discrete_bc","qmix_cql","discrete_bc_20","discrete_bc_50","discrete_bc_5"]  # 可以修改这个列表
    algorithms_to_run = list(algo_dict.keys())  # 运行全部算法
    
    # 存储所有算法的结果
    all_results = {}
    
    print(f"开始训练 {len(algorithms_to_run)} 个算法: {algorithms_to_run}")
    
    for algo_name in algorithms_to_run:
        print(f"\n{'='*50}")
        print(f"开始训练算法: {algo_name}")
        print(f"{'='*50}")
        
        try:
            # 重新设置随机种子确保每个算法的训练环境一致
            set_seed(config['training']['seed'])
            
            # 运行算法
            model, training_history = algo_dict[algo_name]()
            
            # 保存训练历史
            save_training_history(training_history, algo_name)
            
            # 保存训练好的模型
            model_path = save_model(model, algo_name)
            
            # 存储结果
            all_results[algo_name] = {
                'model': model,
                'training_history': training_history
            }
            
            print(f"\n算法 {algo_name} 训练完成!")
            
        except Exception as e:
            print(f"\n算法 {algo_name} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # === 生成对比报告 ===
    print(f"\n{'='*60}")
    print("训练完成总结")
    print(f"{'='*60}")
    
    # 创建汇总结果
    summary_results = {}
    
    for algo_name, result in all_results.items():
        training_history = result['training_history']
        
        # 获取最后一次评估结果
        if training_history['eval_metrics']:
            last_eval = training_history['eval_metrics'][-1]['metrics']
            summary_results[algo_name] = {
                'final_accuracy': last_eval.get('acc_mean', 0),
                'final_cost': last_eval.get('mean_total_cost', 0),
                'final_violation_rate': last_eval.get('violation_rate_mean', 0),
                'final_health_improve': last_eval.get('health_improve_mean', 0) if 'health_improve_mean' in last_eval else None,
                'num_epochs': len(training_history['epochs']),
                'budget_utilization_mean': last_eval.get('budget_utilization_mean', 0) if 'budget_utilization_mean' in last_eval else None,
                'cost_health_efficiency_mean': last_eval.get('cost_health_efficiency_mean', 0) if 'cost_health_efficiency_mean' in last_eval else None,
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None
            }
        else:
            summary_results[algo_name] = {
                'final_accuracy': None,
                'final_cost': None,
                'final_violation_rate': None,
                'final_health_improve': None,
                'num_epochs': len(training_history['epochs']),
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None
            }
    
    # 打印汇总表格
    print(f"{'算法名称':<15} {'准确率':<10} {'总成本':<10} {'违规率':<10} {'健康改善':<10} {'经费使用占比':<10} {'经费健康提升效率':<10}")
    print("-" * 80)
    
    for algo_name, metrics in summary_results.items():
        acc = f"{metrics['final_accuracy']:.4f}" if metrics['final_accuracy'] is not None else "N/A"
        cost = f"{metrics['final_cost']:.1f}" if metrics['final_cost'] is not None else "N/A"
        vio = f"{metrics['final_violation_rate']:.4f}" if metrics['final_violation_rate'] is not None else "N/A"
        health = f"{metrics['final_health_improve']:.4f}" if metrics['final_health_improve'] is not None else "N/A"
        loss = f"{metrics['final_loss']:.4f}" if metrics['final_loss'] is not None else "N/A"
        budget = f"{metrics['budget_utilization_mean']:.4f}" if metrics['budget_utilization_mean'] is not None else "N/A"
        cost_health = f"{metrics['cost_health_efficiency_mean']:.4f}" if metrics['cost_health_efficiency_mean'] is not None else "N/A"

        
        print(f"{algo_name:<15} {acc:<10} {cost:<10} {vio:<10} {health:<10} {budget:<10} {cost_health:<10}")
    
    # 保存汇总结果
    save_metrics(summary_results, "comparison_summary")
    
    print(f"\n所有结果已保存到 metrics_results/ 和 training_results/ 目录")
    print("可以使用保存的数据绘制训练曲线和对比图表")
    
    return all_results


if __name__ == "__main__":
    results = main()