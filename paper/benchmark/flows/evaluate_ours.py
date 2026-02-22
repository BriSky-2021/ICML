import os
import sys
import numpy as np
import torch as th

# ensure utils 可导入
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARK_ROOT)

from utils.transition_util import build_transition_matrices


def convert_budget_levels_to_values(budget_levels,log_btotal, sample):
        """将预算级别转换为对数空间的预算值"""
        log_budget_values = th.zeros_like(budget_levels, dtype=th.float32)
        budget_thresholds = sample["budget_thresholds"].squeeze()

        #print(budget_thresholds)
        #print(log_btotal)
        #exit(0)
        
        for level in range(8):
            mask = (budget_levels == level)
            
            if level == 0:
                # 最低级别
                level_value = 0.0
            elif level == 8 - 1:
                # 最高级别
                if len(budget_thresholds) > 0:
                    level_value = (budget_thresholds[-1] + log_btotal) / 2.0
                else:
                    level_value = log_btotal * 0.9
            else:
                # 中间级别
                if level < len(budget_thresholds):
                    if level == 1:
                        level_value = budget_thresholds[0] / 2.0
                    else:
                        level_value = (budget_thresholds[level-2] + budget_thresholds[level-1]) / 2.0
                else:
                    level_value = log_btotal * (level / 8)
            
            level_value = float(level_value)
            log_budget_values[mask] = level_value
        
        return log_budget_values


def evaluate_on_test_set(
    budget_learner, decision_learner, shared_graph_builder, 
    test_replay_buffer, env_info, verbose=True, logger=None
):
    """
    简明评估主函数
    1. 平均健康改善
    2. 平均实际花费
    3. 平均预算分配值
    4. 动作满足预算分配限制百分比（预算违规率）
    5. 动作与原动作相似比例
    """
    def log_message(msg):
        if logger:
            if hasattr(logger, 'log_eval'):
                logger.log_eval(msg)
            else:
                logger.info(msg)
        elif verbose:
            print(msg)

    transition_matrices = build_transition_matrices()
    action_costs = {0: 0, 1: 71.56, 2: 1643.31, 3: 2433.53}
    n_actions = len(action_costs)
    
    n_episodes = test_replay_buffer.episodes_in_buffer
    max_test_episodes = min(15, n_episodes)
    
    health_improvements = []
    total_costs = []
    total_allocated_budgets = []
    violation_rates = []
    action_similarity_rates = []

    for ep_idx in range(max_test_episodes):
        sample = test_replay_buffer[ep_idx:ep_idx+1]
        batch_size = sample.batch_size
        max_seq_length = sample.max_seq_length
        device = sample.device
        
        # get实际桥梁数量
        if hasattr(sample, 'n_bridges_actual') and 'n_bridges_actual' in sample.data.episode_data:
            actual_bridges = sample.data.episode_data['n_bridges_actual'][0].item()
        else:
            actual_bridges = sample["obs"].shape[2]
        
        # get总预算
        if "btotal" in sample.data.episode_data:
            log_btotal = sample["btotal"].item()
            total_budget = np.expm1(log_btotal)
        else:
            total_budget = 100000
        
        # get初始健康状态
        initial_obs = sample["obs"][:, 0]
        initial_health = initial_obs[:, :, 0].cpu().numpy()  # [batch, bridges]
        cur_health = initial_health.copy()
        
        # raw动作序列（用于相似性度量）
        if "actions" in sample.data.episode_data:
            true_action_seq = sample["actions"].cpu().numpy()  # [batch, T, bridges]
        else:
            true_action_seq = None
        
        total_correct = 0
        total_steps = 0
        total_cost = 0
        total_allocated = 0
        total_violation = 0

        for t in range(max_seq_length - 1):
            # budget分配
            budget_out, *_ = budget_learner.mac.forward(sample, t)
            budget_levels = budget_out.max(dim=2)[1]

            # use类方法转换预算级别为log预算值（如果有的话，建议放在self里）
            log_budget_values = convert_budget_levels_to_values(
                budget_levels, th.tensor(np.log1p(total_budget)), sample
            )
            budget_allocations = th.expm1(log_budget_values).cpu().numpy()  # [batch, bridges]

            

            # normalize分配
            total_alloc = np.sum(budget_allocations, axis=1, keepdims=True)  # [batch, 1]
            scale = (total_budget / (total_alloc + 1e-8)).reshape(-1, 1)
            budget_allocations = budget_allocations * scale
            total_allocated += np.sum(budget_allocations)

            # print(total_alloc,budget_allocations)

            # updatebudget_obs
            budget_obs = th.log1p(th.tensor(budget_allocations, device=device))
            sample.update({"budget_obs": budget_obs.unsqueeze(1).unsqueeze(3)}, ts=t)

            # action决策
            decision_out, *_ = decision_learner.mac.forward(sample, t)
            pred_actions = decision_out.max(dim=2)[1].cpu().numpy()

            # budget违规检测
            violation = (np.vectorize(lambda a, b: action_costs[a] > b)(pred_actions, budget_allocations)).sum()
            total_violation += violation

            # action相似性
            if true_action_seq is not None:
                correct = (pred_actions == true_action_seq[:, t, :actual_bridges]).sum()
                total_correct += correct
                total_steps += np.prod(pred_actions.shape)

            # 花费统计
            step_cost = np.vectorize(lambda a: action_costs[a])(pred_actions).sum()
            total_cost += step_cost

            # 健康推进
            for b in range(batch_size):
                for a in range(actual_bridges):
                    action = pred_actions[b, a]
                    state = int(cur_health[b, a]) - 1
                    state = np.clip(state, 0, transition_matrices[action].shape[0]-1)
                    probs = transition_matrices[action][state]
                    next_state = np.random.choice(len(probs), p=probs)
                    cur_health[b, a] = next_state + 1
        
        final_health = cur_health
        health_improve = (final_health - initial_health).mean()
        health_improvements.append(health_improve)
        total_costs.append(total_cost / batch_size)
        total_allocated_budgets.append(total_allocated / (batch_size * (max_seq_length-1)))
        violation_rates.append(total_violation / (batch_size * (max_seq_length-1) * actual_bridges))
        if total_steps > 0:
            action_similarity_rates.append(total_correct / total_steps)
        else:
            action_similarity_rates.append(np.nan)

        if verbose:
            log_message(
                f"[EVAL] Ep{ep_idx+1}: 健康改善:{health_improve:.3f} "
                f"花费:{total_cost / batch_size:.1f} 分配:{total_allocated / (batch_size ):.1f} "  # * (max_seq_length-1)
                f"预算违规率:{violation_rates[-1]:.3f} 动作相似率:{action_similarity_rates[-1]:.3f}"
            )
    
    results = {
        "health_improve_mean": np.mean(health_improvements),
        "total_cost_mean": np.mean(total_costs),
        "budget_allocated_mean": np.mean(total_allocated_budgets),
        "violation_rate_mean": np.mean(violation_rates),
        "action_similarity_mean": np.nanmean(action_similarity_rates)
    }
    if verbose:
        log_message("========== 简明测试集评估 ==========")
        for k, v in results.items():
            log_message(f"{k}: {v:.4f}")
        log_message("=================================")
    return results