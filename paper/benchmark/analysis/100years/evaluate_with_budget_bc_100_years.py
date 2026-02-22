import os
import sys
import torch as th
import numpy as np
import json
import glob
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm  # addtqdm导入

# get当前脚本文件的绝对路径，并设置 sys.path 以便导入 flows 与 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
flows_dir = os.path.join(benchmark_root, 'flows')
for _path in (flows_dir, benchmark_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Import necessary modules
from Multi_Task_Run_v4 import (
    load_config, load_buffer, 
    extract_data_from_buffer, load_env_info
)
from evaluate_unified_v4 import (
    ModelWrapper,
    calculate_expected_health_improvement,
    project_health_to_categories,
)
from utils.transition_util import build_transition_matrices


def apply_health_transition(current_health, action, transition_matrices):
    """
    使用给定的状态转移矩阵对单个桥梁的健康状态进行转移。
    
    Args:
        current_health: 当前健康状态（原始值0-9）
        action: 执行的动作
        transition_matrices: 各动作对应的转移矩阵字典
    
    Returns:
        new_health: 新的健康状态（原始值0-9）
    """
    if action not in transition_matrices:
        action = 0
    mat = transition_matrices[action]

    # will0-9的健康值映射到4个健康类别（与bridge_maintenance_simulator_v2一致）
    if current_health <= 2:       # 0-2: critical
        old_cat = 0
    elif current_health <= 4:     # 3-4: poor
        old_cat = 1
    elif current_health <= 6:     # 5-6: fair
        old_cat = 2
    else:                         # 7-9: good
        old_cat = 3

    probs = mat[old_cat]
    new_cat = np.random.choice(len(probs), p=probs)

    # if健康类别未发生变化，则保持原始健康分数不变，提高轨迹稳定性
    if new_cat == old_cat:
        return float(current_health)

    # else，仅在类别发生变化时，在新类别对应区间内重新采样具体健康分数
    if new_cat == 0:        # critical
        new_health = np.random.randint(0, 3)   # 0-2
    elif new_cat == 1:      # poor
        new_health = np.random.randint(3, 5)   # 3-4
    elif new_cat == 2:      # fair
        new_health = np.random.randint(5, 7)   # 5-6
    else:                   # good
        new_health = np.random.randint(7, 10)  # 7-9

    return float(new_health)

# --- Configuration ---
ACTION_COSTS = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}

ALGO_TYPE = {
    "qmix_cql": "marl",
    "iqlcql_marl": "marl",
    "multitask_cpq": "osrl",
    "multitask_offline_cpq": "osrl",
    "onestep": "osrl",
    "onestep_heuristic": "osrl",
    "random_osrl": "osrl",
    "cdt": "cdt",
    "cql": "osrl",
    "random_marl": "marl",
    "discrete_bc": "marl",
    "multitask_bc": "osrl",
    "multitask_bc_top20": "osrl",
    "multitask_bc_top50": "osrl",
}

# List of algorithms to evaluate automatically
TARGET_ALGORITHMS = [
    #"qmix_cql",
    #"iqlcql_marl",
    #"multitask_offline_cpq",
    #"random_osrl",
    #"onestep",
    #"multitask_cpq"
    #"cql",
    "cql_heuristic",
    #"discrete_bc"
    #"multitask_bc",
    #"onestep_heuristic",
    #"multitask_cpq_heuristic"
    #"cdt",
    #"heuristic",
]

# The algorithm name used for the Budget Oracle (must be multi-agent BC)
BC_ALGORITHM_NAME = "discrete_bc"

# Directory containing the models
MODEL_DIR = "paper/benchmark/saved_models"


def get_action_cost(action):
    """Get action cost from action ID"""
    return ACTION_COSTS.get(int(action), 0.0)


def mask_invalid_q_values(q_values, health_categories, active_indices):
    """
    屏蔽无效的Q值
    
    需要屏蔽的情况：
    - poor(0)状态下的action2和action3
    - excellent(3)状态下的action1
    
    Args:
        q_values: [n_active, n_actions] 或 [n_agents, n_actions] Q值矩阵
        health_categories: [n_active] 健康类别数组 (0-3)，按active_indices的顺序排列
        active_indices: 激活的智能体索引列表
    
    Returns:
        masked_q_values: 屏蔽后的Q值矩阵（无效动作的Q值设为-inf）
    """
    if q_values is None:
        return None
    
    masked_q_values = q_values.copy()
    
    # ifq_values是[n_active, n_actions]格式（最常见的情况）
    if q_values.shape[0] == len(active_indices):
        # q_values已经是[n_active, n_actions]格式
        # health_categories也应该是[n_active]格式，按active_indices的顺序排列
        for i, h_cat in enumerate(health_categories):
            if i >= masked_q_values.shape[0]:
                continue
            if h_cat == 0:  # poor(0)状态
                # 屏蔽action2和action3
                if masked_q_values.shape[1] > 2:
                    masked_q_values[i, 2] = -np.inf
                if masked_q_values.shape[1] > 3:
                    masked_q_values[i, 3] = -np.inf
            elif h_cat == 3:  # excellent(3)状态
                # 屏蔽action1
                if masked_q_values.shape[1] > 1:
                    masked_q_values[i, 1] = -np.inf
    else:
        # q_values是[n_agents, n_actions]格式，需要映射active_indices
        # at这种情况下，health_categories应该是按active_indices的顺序排列的
        for idx, agent_idx in enumerate(active_indices):
            if idx < len(health_categories) and agent_idx < masked_q_values.shape[0]:
                h_cat = health_categories[idx]
                if h_cat == 0:  # poor(0)状态
                    # 屏蔽action2和action3
                    if masked_q_values.shape[1] > 2:
                        masked_q_values[agent_idx, 2] = -np.inf
                    if masked_q_values.shape[1] > 3:
                        masked_q_values[agent_idx, 3] = -np.inf
                elif h_cat == 3:  # excellent(3)状态
                    # 屏蔽action1
                    if masked_q_values.shape[1] > 1:
                        masked_q_values[agent_idx, 1] = -np.inf
    
    return masked_q_values


def find_latest_model(model_dir="paper/benchmark/saved_models", algorithm_name=None):
    """Find the latest model file based on timestamp"""
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

    # by修改时间排序，返回最新的
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"找到最新模型: {latest_model}")
    return latest_model


def extract_algorithm_name(model_path):
    """从模型路径中提取算法名称"""
    basename = os.path.basename(model_path)
    # format: {algorithm_name}_model_{timestamp}.pth
    if '_model_' in basename:
        algo_name = basename.split('_model_')[0]
        return algo_name
    return None


def get_algorithm_type(model_path):
    """根据模型路径获取算法类型"""
    algo_name = extract_algorithm_name(model_path)
    print(f"algo_name: {algo_name}")
    if algo_name and algo_name in ALGO_TYPE:
        return ALGO_TYPE[algo_name]
    return "marl"  # default


class HeuristicBudgetController:
    """
    启发式预算约束求解器
    根据Q值依次选择每一座桥梁的动作，直到动作对应的成本值到达预算
    注意：此版本不按健康状态排序，仅按Q值优先级排序
    """
    def __init__(self, action_costs):
        self.action_costs = action_costs

    def enforce_budget(self, actions, q_values, budget_limit, active_indices, health_states):
        """
        根据预算约束调整动作
        
        Args:
            actions: [n_agents] 原始动作数组
            q_values: [n_agents, n_actions] Q值矩阵，或None
            budget_limit: 预算限制
            active_indices: 激活的智能体索引列表
            health_states: [n_agents] 当前所有智能体的健康状态 (raw health 0-3 or 0-9)
        
        Returns:
            adjusted_actions: [n_agents] 调整后的动作数组
        """
        actions = np.asarray(actions)
        n_agents = len(actions)
        
        # compute当前总成本
        current_cost = sum([self.action_costs[int(actions[i])] for i in active_indices])
        
        # if未超出预算，直接返回
        if current_cost <= budget_limit + 1e-5:
            return actions.copy()
        
        # 超出预算，需要根据Q值选择动作
        adjusted_actions = np.zeros(n_agents, dtype=int)  # default全部为0（不做任何动作）
        
        # 收集所有维护请求（action > 0）
        maintenance_requests = []
        for idx, i in enumerate(active_indices):
            act = int(actions[i])
            if act > 0:
                cost = self.action_costs[act]
                
                # compute优先级数值：如果有Q值，使用Q(s,a) - Q(s,0)；否则使用成本作为代理
                if q_values is not None:
                    # 判断Q值的格式：如果是[n_active, n_actions]，使用idx；如果是[n_agents, n_actions]，使用i
                    if q_values.shape[0] == len(active_indices):
                        # Q值是[n_active, n_actions]格式
                        if idx < q_values.shape[0] and act < q_values.shape[1]:
                            # Q值增益：选择该动作相对于不做动作的增益
                            priority = float(q_values[idx, act] - q_values[idx, 0])
                        else:
                            priority = float(cost)
                    else:
                        # Q值是[n_agents, n_actions]格式
                        if i < q_values.shape[0] and act < q_values.shape[1]:
                            # Q值增益：选择该动作相对于不做动作的增益
                            priority = float(q_values[i, act] - q_values[i, 0])
                        else:
                            priority = float(cost)
                else:
                    # 没有Q值时，使用成本作为优先级（假设成本越高越重要）
                    priority = float(cost)
                
                maintenance_requests.append({
                    'agent_id': i,
                    'action': act,
                    'cost': cost,
                    'priority': priority
                })
        
        # by优先级排序（仅按Q值增益排序，不按健康状态）
        # Priority (Q-gain) 越大越靠前 (使用负号实现降序)
        maintenance_requests.sort(key=lambda x: -x['priority'])
        
        # 贪心选择：按优先级依次批准动作，直到预算用完
        current_spending = 0.0
        for req in maintenance_requests:
            if current_spending + req['cost'] <= budget_limit + 1e-5:
                adjusted_actions[req['agent_id']] = req['action']
                current_spending += req['cost']
            # if超出预算，该动作被拒绝（保持为0）
        
        return adjusted_actions


def load_model_safe(path, device):
    """安全加载模型，确保所有参数都在指定设备上"""
    try:
        # load模型到指定设备
        model = th.load(path, map_location=device)
        
        # 显式将模型移动到设备（确保所有参数和缓冲区都在正确设备上）
        if hasattr(model, 'to'):
            model = model.to(device)
        
        # if模型有device属性，更新它
        if hasattr(model, 'device'):
            model.device = device
            print(f"模型内部的 'device' 属性已更新为: {device}")
        
        # ensure所有子模块也在正确设备上
        if hasattr(model, 'modules'):
            for module in model.modules():
                if hasattr(module, 'to'):
                    module.to(device)
        
        # 设置评估模式
        if hasattr(model, 'eval'):
            model.eval()
        elif hasattr(model, 'train'):
            model.train(False)
        
        # 验证模型参数是否在正确设备上
        if hasattr(model, 'parameters'):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                actual_device = first_param.device
                if actual_device != device:
                    print(f"警告: 模型参数在 {actual_device}，expected在 {device}，尝试修复...")
                    model = model.to(device)
        
        print(f"模型已加载到设备: {device}")
        return model
    except Exception as e:
        print(f"加载模型失败 {path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_model_q_values(model, obs, budget, active_indices, algorithm_type, device):
    """
    从模型中提取Q值（支持批处理）
    
    Args:
        model: 模型对象
        obs: 观测 [n_active, obs_dim] (批处理) 或 [n_agents, obs_dim] (多智能体)
        budget: 预算值 [n_active] (批处理) 或标量
        active_indices: 激活的智能体索引列表
        algorithm_type: 'marl' 或 'osrl' 或 'cdt'
        device: 设备
    
    Returns:
        q_values: [n_active, n_actions] 或 [n_agents, n_actions] 或 None
    """
    try:
        with th.no_grad():
            # method1: 模型有get_q_values方法
            if hasattr(model, 'get_q_values'):
                if algorithm_type == 'marl':
                    # 多智能体：obs应该是 [n_agents, obs_dim] 或 [1, n_agents, obs_dim]
                    #requires转换为 [B, N, obs_dim] 格式，其中B=1
                    obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
                    
                    # handle不同维度的输入
                    if obs_tensor.ndim == 1:
                        # [obs_dim] -> [1, 1, obs_dim]
                        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                    elif obs_tensor.ndim == 2:
                        # [n_agents, obs_dim] -> [1, n_agents, obs_dim]
                        obs_tensor = obs_tensor.unsqueeze(0)
                    elif obs_tensor.ndim == 3:
                        # already经是 [B, N, obs_dim] 格式，保持不变
                        pass
                    else:
                        raise ValueError(f"不支持的obs维度: {obs_tensor.ndim}")
                    
                    # 现在obs_tensor应该是 [B, N, obs_dim] 格式
                    q_vals = model.get_q_values(obs_tensor)  # [B, N, n_actions]
                    if isinstance(q_vals, th.Tensor):
                        q_vals = q_vals.cpu().numpy()
                    
                    # handle输出维度
                    if q_vals.ndim == 3:
                        # [B, N, n_actions] -> [N, n_actions]
                        q_vals = q_vals[0]  # 取第一个batch
                    elif q_vals.ndim == 2:
                        # already经是 [N, n_actions]
                        pass
                    else:
                        raise ValueError(f"不支持的q_vals维度: {q_vals.ndim}")
                    
                    return q_vals
                elif algorithm_type == 'osrl':
                    # 单智能体：支持批处理，obs是 [n_active, obs_dim]
                    obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)  # [1, obs_dim]
                    
                    # handlebudget
                    if budget is not None:
                        budget_tensor = th.tensor(budget, dtype=th.float32, device=device)
                        if budget_tensor.ndim == 0:
                            budget_tensor = budget_tensor.unsqueeze(0)  # [1]
                        if budget_tensor.ndim == 1 and budget_tensor.shape[0] != obs_tensor.shape[0]:
                            # ifbudget是标量，扩展到batch大小
                            if budget_tensor.shape[0] == 1:
                                budget_tensor = budget_tensor.expand(obs_tensor.shape[0])
                        q_vals = model.get_q_values(obs_tensor, budget=budget_tensor)
                    else:
                        q_vals = model.get_q_values(obs_tensor)
                    
                    if isinstance(q_vals, th.Tensor):
                        q_vals = q_vals.cpu().numpy()
                    # return [n_active, n_actions]
                    return q_vals
            
            # method2: 模型有q_network属性
            if hasattr(model, 'q_network'):
                if algorithm_type == 'marl':
                    obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    q_vals = model.q_network(obs_tensor)
                    if isinstance(q_vals, th.Tensor):
                        q_vals = q_vals.cpu().numpy()
                    return q_vals
                elif algorithm_type == 'osrl':
                    # 单智能体：支持批处理
                    obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    # if有budget，需要拼接
                    if budget is not None:
                        budget_tensor = th.tensor(budget, dtype=th.float32, device=device)
                        if budget_tensor.ndim == 0:
                            budget_tensor = budget_tensor.unsqueeze(0)
                        if budget_tensor.ndim == 1:
                            budget_tensor = budget_tensor.unsqueeze(-1)  # [n_active, 1]
                        # concatstate和budget
                        obs_tensor = th.cat([obs_tensor, budget_tensor], dim=-1)
                    q_vals = model.q_network(obs_tensor)
                    if isinstance(q_vals, th.Tensor):
                        q_vals = q_vals.cpu().numpy()
                    return q_vals
            
            return None
    except Exception as e:
        print(f"提取Q值失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_single_model(
    model_path,
    bc_model,
    test_data_dict,
    device,
    transition_matrices,
    controller,
    action_costs,
    budget_factor=1.0,
    n_years=100,
    env_info=None
):
    """
    评估单个目标模型
    
    Args:
        model_path: 目标模型路径
        bc_model: BC模型（用于生成预算）
        test_data_dict: 测试数据字典
        device: 设备
        transition_matrices: 状态转移矩阵
        controller: 启发式预算控制器
        action_costs: 动作成本字典
    """
    print(f"\n{'='*60}")
    print(f"评估目标模型: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    # load目标模型
    target_model = load_model_safe(model_path, device)
    if target_model is None:
        print(f"无法加载模型: {model_path}")
        return None
    
    # 确定算法类型
    algorithm_type = get_algorithm_type(model_path)
    print(f"算法类型: {algorithm_type}")

    # if是CDT算法，使用专门的CDT评估函数
    if algorithm_type == 'cdt':
        return evaluate_single_model_cdt(
            model_path=model_path,
            target_model=target_model,
            bc_model=bc_model,
            test_data_dict=test_data_dict,
            device=device,
            transition_matrices=transition_matrices,
            controller=controller,
            action_costs=action_costs
        )
    
    # 解包数据（仅使用第1年的真实数据，其余年份自己生成）
    data = test_data_dict['data']          # [num_eps, T, n_agents, state_dim]
    health = test_data_dict['health']      # [num_eps, T+1, n_agents]
    actual_n_agents = test_data_dict['actual_n_agents']
    log_budgets = test_data_dict.get('log_budgets', None)  # [num_eps, T, n_agents, 1] 或 [num_eps, T, n_agents]
    actions_history = test_data_dict.get('actions', None)  # [num_eps, T, n_agents] 历史动作
    
    # handlelog_budgets维度：如果最后一维是1，squeeze掉
    if log_budgets is not None:
        if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
            log_budgets = log_budgets.squeeze(-1)  # [num_eps, T, n_agents]
    
    num_eps, T, max_n_agents, state_dim = data.shape

    # normalize参数（用于构造后续年份的观测）
    policy_mean = None
    policy_std = None
    if env_info is not None:
        norm_params = env_info.get('normalization_params', {})
        if 'policy_obs' in norm_params:
            policy_mean = np.array(norm_params['policy_obs']['mean'], dtype=np.float32)
            policy_std = np.array(norm_params['policy_obs']['std'], dtype=np.float32)
    
    # evaluate指标
    total_metrics = defaultdict(float)
    bridge_metrics = defaultdict(list)
    episode_metrics = []

    # action分布统计（整体）
    original_action_counts = defaultdict(int)  # raw动作（应用中心规划器之前）
    final_action_counts = defaultdict(int)     # 最终动作（应用中心规划器之后）

    # by健康等级统计原始/最终动作分布：{health_level: {action: count}}
    health_original_action_counts = defaultdict(lambda: defaultdict(int))  # model原始动作(屏蔽前)
    health_final_action_counts = defaultdict(lambda: defaultdict(int))     # budget约束后动作
    # by健康等级统计“原始数据集（历史动作）”的动作分布：{health_level: {action: count}}
    dataset_health_action_counts = defaultdict(lambda: defaultdict(int))   # 历史动作
    
    # 包装模型
    bc_algorithm_type = get_algorithm_type(BC_ALGORITHM_NAME) if BC_ALGORITHM_NAME in ALGO_TYPE else 'osrl'
    
    with ModelWrapper(target_model, algorithm_type=algorithm_type) as wrapped_target, \
         ModelWrapper(bc_model, algorithm_type=bc_algorithm_type) as wrapped_bc:
        
        # for每个episode（区域）进行评估
        for ep in tqdm(range(num_eps), desc=f"评估 {os.path.basename(model_path)}", unit="episode"):
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
            
            # createagent mask
            agent_mask = np.zeros(max_n_agents, dtype=np.float32)
            agent_mask[active_idx] = 1.0
            
            # Episode级别的指标
            ep_total_cost = 0.0
            ep_total_budget = 0.0
            ep_total_health_gain = 0.0
            ep_total_health_gain_vs_nothing = 0.0
            ep_total_health_gain_vs_history = 0.0
            ep_steps = 0

            # record该episode内每一年的预算、花费与平均健康等级
            ep_yearly_budgets = []
            ep_yearly_costs = []
            ep_yearly_avg_health = []

            # === 使用第1年的数据初始化后续长期模拟所需的状态 ===
            # 基础观测（第1年）
            base_obs_ep = data[ep, 0].copy()  # [max_n_agents, state_dim]

            # 利用归一化参数反推第1年原始特征（只针对policy_obs前几个维度）
            if policy_mean is not None and policy_std is not None:
                policy_dim = len(policy_mean)
                # 第1年policy_obs的归一化值
                base_policy_norm = base_obs_ep[:, :policy_dim]
                base_policy_raw = base_policy_norm * (policy_std + 1e-8) + policy_mean  # [max_n_agents, policy_dim]
                # raw健康、年龄、流量、长度、重要性
                init_health_raw = base_policy_raw[:, 0]
                init_age_raw = base_policy_raw[:, 1]
                base_adt_raw = base_policy_raw[:, 2]
                base_len_raw = base_policy_raw[:, 3]
                base_imp_raw = base_policy_raw[:, 4]
            else:
                # if缺少归一化参数，退化为直接使用数据中的health
                print("警告: 缺少policy_obs归一化参数，长期模拟将直接使用health数组初始化")
                if health is not None:
                    init_health_raw = health[ep, 0].astype(np.float32)
                else:
                    init_health_raw = np.full(max_n_agents, 6.0, dtype=np.float32)
                init_age_raw = np.full(max_n_agents, 50.0, dtype=np.float32)
                base_adt_raw = np.full(max_n_agents, 1000.0, dtype=np.float32)
                base_len_raw = np.full(max_n_agents, 100.0, dtype=np.float32)
                base_imp_raw = np.full(max_n_agents, 0.5, dtype=np.float32)

            # use数据中的health作为初始健康（更接近原始结构评估）
            if health is not None:
                init_health_raw = health[ep, 0]
                # 兼容形状 [max_n_agents] 或 [max_n_agents, 1]
                if hasattr(init_health_raw, "ndim") and init_health_raw.ndim > 1:
                    init_health_raw = init_health_raw.squeeze(-1)
                init_health_raw = init_health_raw.astype(np.float32)

            # build健康和年龄的长期轨迹容器
            sim_health = np.zeros((n_years + 1, max_n_agents), dtype=np.float32)
            sim_age = np.zeros((n_years + 1, max_n_agents), dtype=np.float32)
            sim_health[0] = init_health_raw
            sim_age[0] = init_age_raw

            # build长期使用的log预算序列（从第1年复制，其余年份保持不变）
            if log_budgets is not None:
                lb0 = log_budgets[ep, 0]  # [max_n_agents] 或 [max_n_agents,1]
                if lb0.ndim == 2 and lb0.shape[-1] == 1:
                    lb0 = lb0.squeeze(-1)
                sim_log_budgets = np.tile(lb0.reshape(1, -1), (n_years, 1))  # [n_years, max_n_agents]
            else:
                sim_log_budgets = None

            # for每个时间步（年份）进行长期评估（完全由状态转移和归一化生成）
            for t in range(n_years):
                # === 构造当前年份的观测 obs_t ===
                if policy_mean is not None and policy_std is not None:
                    policy_dim = len(policy_mean)
                    # when前年份的原始特征
                    cur_health_raw = sim_health[t]
                    cur_age_raw = sim_age[t]
                    cur_policy_raw = np.stack(
                        [cur_health_raw, cur_age_raw, base_adt_raw, base_len_raw, base_imp_raw],
                        axis=-1
                    )  # [max_n_agents, 5]
                    # 重新归一化
                    cur_policy_norm = (cur_policy_raw - policy_mean[:5]) / (policy_std[:5] + 1e-8)

                    # from第1年的完整obs复制一份，再覆盖前几个policy_obs维度
                    obs_t = base_obs_ep.copy()
                    obs_t[:, :5] = cur_policy_norm
                else:
                    # 退化：直接使用第1年的obs
                    obs_t = base_obs_ep.copy()
                
                # ========== Step 4: 使用BC模型生成预算 ==========
                # according toBC算法类型调用不同的接口
                if bc_algorithm_type == 'marl':
                    # for于MARL模型，需要检查是否需要拼接budget
                    obs_for_bc = obs_t.copy()  # [max_n_agents, state_dim]
                    
                    # checkBC模型是否需要budget在obs中
                    bc_needs_budget_in_obs = getattr(bc_model, 'has_budget_in_obs', True)  # MARL默认需要
                    if bc_needs_budget_in_obs and sim_log_budgets is not None:
                        # concat预算到obs（使用长期模拟的预算序列）
                        budget_t = sim_log_budgets[t]  # [max_n_agents]
                        if budget_t.ndim == 1:
                            budget_t = budget_t[:, np.newaxis]  # [max_n_agents, 1]
                        obs_for_bc = np.concatenate([obs_t, budget_t], axis=-1)  # [max_n_agents, state_dim+1]
                    
                    obs_batch = obs_for_bc.reshape(1, max_n_agents, -1)  # [1, n_agents, dim]
                    try:
                        bc_actions = wrapped_bc.act(
                            obs=obs_batch,
                            agent_mask=agent_mask.reshape(1, -1),
                            training=False
                        )
                        
                        # handle输出格式：DiscreteBCMultiAgent返回[B, N]
                        if isinstance(bc_actions, (tuple, list)):
                            bc_actions = bc_actions[0]
                        bc_actions = np.asarray(bc_actions)
                        # if是2维数组[B, N]，取第一行
                        if bc_actions.ndim == 2:
                            bc_actions = bc_actions[0]  # [N]
                        # ensure是1维数组
                        if bc_actions.ndim == 0:
                            bc_actions = np.array([bc_actions])
                        # ensure长度匹配
                        if len(bc_actions) < max_n_agents:
                            # if返回的动作数量不足，填充为0
                            bc_actions_full = np.zeros(max_n_agents, dtype=int)
                            bc_actions_full[:len(bc_actions)] = bc_actions
                            bc_actions = bc_actions_full
                    except Exception as e:
                        print(f"BC模型调用失败 (MARL): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                else:  # osrl - 单智能体支持批处理
                    # will所有active agents的obs和budget堆叠成batch
                    obs_batch = obs_t[active_idx]  # [n_active, state_dim]
                    if sim_log_budgets is not None:
                        budget_batch = sim_log_budgets[t, active_idx]  # [n_active]
                        if budget_batch.ndim > 1:
                            budget_batch = budget_batch.squeeze(-1)  # [n_active]
                    else:
                        budget_batch = np.zeros(len(active_idx))
                    
                    try:
                        # 批处理调用：传入 [n_active, state_dim] 和 [n_active]
                        bc_actions_batch = wrapped_bc.act(obs_batch, budget_batch, deterministic=True)
                        
                        # handle输出格式
                        if isinstance(bc_actions_batch, (tuple, list)):
                            bc_actions_batch = bc_actions_batch[0]
                        bc_actions_batch = np.asarray(bc_actions_batch)
                        if bc_actions_batch.ndim == 0:
                            bc_actions_batch = np.array([bc_actions_batch])
                        
                        # will结果填充到完整数组
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                        for i, a_idx in enumerate(active_idx):
                            bc_actions[a_idx] = int(bc_actions_batch[i])
                    except Exception as e:
                        print(f"BC模型调用失败 (OSRL批处理): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)

                # handleBC动作输出格式（统一处理，确保是1维数组）
                if isinstance(bc_actions, (tuple, list)):
                    bc_actions = bc_actions[0]
                bc_actions = np.asarray(bc_actions)
                if bc_actions.ndim == 2:
                    bc_actions = bc_actions[0]  # [N]
                if bc_actions.ndim == 0:
                    bc_actions = np.array([bc_actions])
                # ensure是整数类型
                bc_actions = bc_actions.astype(int)
                # ensure长度匹配
                if len(bc_actions) < max_n_agents:
                    bc_actions_full = np.zeros(max_n_agents, dtype=int)
                    bc_actions_full[:len(bc_actions)] = bc_actions
                    bc_actions = bc_actions_full
                elif len(bc_actions) > max_n_agents:
                    bc_actions = bc_actions[:max_n_agents]
                
                # compute该年份本区域的总预算（BC动作的总成本）
                year_budget_limit = 0.0
                for a in active_idx:
                    act = int(bc_actions[a])  # 现在应该是标量
                    year_budget_limit += get_action_cost(act)
                
                # 放宽到 BC 预算的 1.2 倍，并应用预算因子
                year_budget_limit *= 1.2 * float(budget_factor)

                # ========== Step 5: 获取Q值并使用屏蔽后的Q值进行贪婪选择 ==========
                # 首先获取Q值（不管是否超出预算）
                if sim_log_budgets is not None:
                    budget_batch = sim_log_budgets[t, active_idx]  # [n_active]
                else:
                    budget_batch = None
                
                # for于MARL模型，需要拼接budget到obs
                obs_for_q = obs_t[active_idx].copy()  # [n_active, state_dim]
                if algorithm_type == 'marl':
                    target_needs_budget_in_obs = getattr(target_model, 'has_budget_in_obs', True)
                    if target_needs_budget_in_obs and budget_batch is not None:
                        bb = budget_batch
                        if bb.ndim == 1:
                            bb = bb[:, np.newaxis]  # [n_active, 1]
                        obs_for_q = np.concatenate([obs_t[active_idx], bb], axis=-1)  # [n_active, state_dim+1]
                
                # getQ值
                q_values = get_model_q_values(
                    target_model, obs_for_q, budget_batch, active_idx, algorithm_type, device
                )

                # record屏蔽Q值之前的贪婪动作（用于健康等级下的原始动作偏好统计）
                pre_mask_actions = np.full(max_n_agents, -1, dtype=int)
                if q_values is not None:
                    if q_values.shape[0] == len(active_idx):
                        # Q值是[n_active, n_actions]格式
                        for i, a_idx in enumerate(active_idx):
                            if i < q_values.shape[0]:
                                pre_mask_actions[a_idx] = int(np.argmax(q_values[i]))
                    else:
                        # Q值是[n_agents, n_actions]格式
                        for a_idx in active_idx:
                            if a_idx < q_values.shape[0]:
                                pre_mask_actions[a_idx] = int(np.argmax(q_values[a_idx]))
                
                # get健康类别用于屏蔽
                health_categories = []
                for a_idx in active_idx:
                    if health is not None and t < health.shape[1]:
                        raw_h = int(health[ep, t, a_idx])
                        h_cat = project_health_to_categories(raw_h)
                        health_categories.append(h_cat)
                    else:
                        health_categories.append(0)  # default值
                
                # 应用屏蔽
                masked_q_values = mask_invalid_q_values(q_values, health_categories, active_idx)
                
                # use屏蔽后的Q值进行贪婪选择
                if masked_q_values is not None:
                    # useQ值贪婪选择
                    target_actions = np.zeros(max_n_agents, dtype=int)
                    if masked_q_values.shape[0] == len(active_idx):
                        # Q值是[n_active, n_actions]格式
                        for i, a_idx in enumerate(active_idx):
                            # selectQ值最大的动作（屏蔽后的）
                            target_actions[a_idx] = int(np.argmax(masked_q_values[i]))
                    else:
                        # Q值是[n_agents, n_actions]格式
                        for i, a_idx in enumerate(active_idx):
                            if a_idx < masked_q_values.shape[0]:
                                target_actions[a_idx] = int(np.argmax(masked_q_values[a_idx]))
                else:
                    # if没有Q值，回退到原始模型动作选择
                    if algorithm_type == 'marl':
                        # 多智能体模型 - 需要拼接budget
                        obs_for_target = obs_t.copy()  # [max_n_agents, state_dim]
                        
                        # check目标模型是否需要budget在obs中
                        target_needs_budget_in_obs = getattr(target_model, 'has_budget_in_obs', True)  # MARL默认需要
                        if target_needs_budget_in_obs and sim_log_budgets is not None:
                            # concat预算到obs（使用长期模拟预算）
                            budget_t = sim_log_budgets[t]  # [max_n_agents]
                            if budget_t.ndim == 1:
                                budget_t = budget_t[:, np.newaxis]  # [max_n_agents, 1]
                            obs_for_target = np.concatenate([obs_t, budget_t], axis=-1)  # [max_n_agents, state_dim+1]
                        
                        obs_batch = obs_for_target.reshape(1, max_n_agents, -1)  # [1, n_agents, dim]
                        try:
                            target_actions = wrapped_target.act(
                                obs=obs_batch,
                                agent_mask=agent_mask.reshape(1, -1),
                                training=False
                            )
                            
                            # handle输出格式：MARL模型返回[B, N]
                            if isinstance(target_actions, (tuple, list)):
                                target_actions = target_actions[0]
                            target_actions = np.asarray(target_actions)
                            if target_actions.ndim == 2:
                                target_actions = target_actions[0]  # [N]
                            if target_actions.ndim == 0:
                                target_actions = np.array([target_actions])
                            target_actions = target_actions.astype(int)
                            # ensure长度匹配
                            if len(target_actions) < max_n_agents:
                                target_actions_full = np.zeros(max_n_agents, dtype=int)
                                target_actions_full[:len(target_actions)] = target_actions
                                target_actions = target_actions_full
                            elif len(target_actions) > max_n_agents:
                                target_actions = target_actions[:max_n_agents]
                        except Exception as e:
                            print(f"目标模型调用失败 (MARL): {e}")
                            target_actions = np.zeros(max_n_agents, dtype=int)
                    else:  # osrl - 单智能体支持批处理
                        algo_name = extract_algorithm_name(model_path)
                        is_multitask_bc = (algo_name == "multitask_bc" or 
                                          algo_name == "multitask_bc_top20" or 
                                          algo_name == "multitask_bc_top50")
                        
                        if is_multitask_bc:
                            # MultiTaskBC不支持批处理，需要逐个调用
                            target_actions = np.zeros(max_n_agents, dtype=int)
                            for i, a_idx in enumerate(active_idx):
                                obs_single = obs_t[a_idx]  # [state_dim]
                                if sim_log_budgets is not None:
                                    budget_single = float(sim_log_budgets[t, a_idx])
                                else:
                                    budget_single = 0.0

                                try:
                                    action = wrapped_target.act(obs_single, budget_single, deterministic=True)
                                    target_actions[a_idx] = int(action)
                                except Exception as e:
                                    print(f"目标模型调用失败 (MultiTaskBC单个agent {a_idx}): {e}")
                                    target_actions[a_idx] = 0
                        else:
                            # 其他OSRL算法支持批处理
                            # will所有active agents的obs和budget堆叠成batch
                            obs_batch = obs_t[active_idx]  # [n_active, state_dim]
                            if sim_log_budgets is not None:
                                budget_batch = sim_log_budgets[t, active_idx]  # [n_active]
                            else:
                                budget_batch = np.zeros(len(active_idx))
                            
                            try:
                                # 批处理调用
                                target_actions_batch = wrapped_target.act(obs_batch, budget_batch, deterministic=True)
                                
                                # handle输出格式
                                if isinstance(target_actions_batch, (tuple, list)):
                                    target_actions_batch = target_actions_batch[0]
                                target_actions_batch = np.asarray(target_actions_batch)
                                if target_actions_batch.ndim == 0:
                                    target_actions_batch = np.array([target_actions_batch])
                                
                                # will结果填充到完整数组
                                target_actions = np.zeros(max_n_agents, dtype=int)
                                for i, a_idx in enumerate(active_idx):
                                    target_actions[a_idx] = int(target_actions_batch[i])
                            except Exception as e:
                                print(f"目标模型调用失败 (OSRL批处理): {e}")
                                target_actions = np.zeros(max_n_agents, dtype=int)
                
                # ========== Step 5 (续): 检查预算并应用启发式控制器 ==========
                # handle目标动作输出格式（统一处理）
                if isinstance(target_actions, (tuple, list)):
                    target_actions = target_actions[0]
                target_actions = np.asarray(target_actions)
                if target_actions.ndim == 2:
                    target_actions = target_actions[0]  # [N]
                if target_actions.ndim == 0:
                    target_actions = np.array([target_actions])
                target_actions = target_actions.astype(int)
                # ensure长度匹配
                if len(target_actions) < max_n_agents:
                    target_actions_full = np.zeros(max_n_agents, dtype=int)
                    target_actions_full[:len(target_actions)] = target_actions
                    target_actions = target_actions_full
                elif len(target_actions) > max_n_agents:
                    target_actions = target_actions[:max_n_agents]

                # if无法得到屏蔽前动作，回退为当前目标动作
                for a in active_idx:
                    if pre_mask_actions[a] < 0:
                        pre_mask_actions[a] = int(target_actions[a])
                # compute当前动作的总成本
                current_cost = sum([get_action_cost(int(target_actions[a])) for a in active_idx])
                
                # if超出预算，使用启发式控制器（使用屏蔽后的Q值）
                if current_cost > year_budget_limit + 1e-5:
                    # get当前所有Agent的健康状态，传给Controller（使用长期模拟健康）
                    current_health_states = sim_health[t]  # [max_n_agents]
                
                    # 应用启发式预算约束（使用屏蔽后的Q值）
                    final_actions = controller.enforce_budget(
                        actions=target_actions,
                        q_values=masked_q_values,
                        budget_limit=year_budget_limit,
                        active_indices=active_idx,
                        health_states=current_health_states
                    )
                else:
                    final_actions = target_actions.copy()
                
                # ========== Step 6: 评估指标 ==========
                step_cost = 0.0
                step_health_gain = 0.0
                step_health_gain_vs_nothing = 0.0
                step_health_gain_vs_history = 0.0
                
                # record原始动作分布（仅对active agents）
                for a in active_idx:
                    original_action_counts[int(target_actions[a])] += 1
                
                # record最终动作分布（仅对active agents）
                for a in active_idx:
                    final_action_counts[int(final_actions[a])] += 1
                
                for a in active_idx:
                    act_final = int(final_actions[a])
                    act_original = int(pre_mask_actions[a])
                    cost = get_action_cost(act_final)
                    step_cost += cost
                    
                    # 健康改善计算（使用长期模拟的健康状态）
                    if t + 1 <= n_years:
                        # raw健康评分 (0-9)，来自模拟轨迹
                        raw_h = int(sim_health[t, a])
                        # map到 4 级健康类别 (0-3)，用于统计动作偏好
                        h_cat = project_health_to_categories(raw_h)

                        # by 4 级健康类别记录原始动作(屏蔽前)和最终动作（如只关心维修动作，可加条件 act_* > 0）
                        health_original_action_counts[h_cat][act_original] += 1
                        health_final_action_counts[h_cat][act_final] += 1
                        
                        # expected健康改善（这里继续使用原始评分，内部会自行做类别投影）
                        gain = calculate_expected_health_improvement(raw_h, act_final, transition_matrices)
                        
                        # 相对于不做动作的改善
                        gain_nothing = calculate_expected_health_improvement(raw_h, 0, transition_matrices)
                        
                        # 相对于历史动作的改善
                        gain_history = 0.0
                        if actions_history is not None and t < actions_history.shape[1]:
                            hist_act = int(actions_history[ep, t, a])
                            # record原始数据集中，在该 4 级健康类别下历史动作的分布
                            dataset_health_action_counts[h_cat][hist_act] += 1
                            gain_history = calculate_expected_health_improvement(raw_h, hist_act, transition_matrices)
                        
                        step_health_gain += gain
                        step_health_gain_vs_nothing += (gain - gain_nothing)
                        step_health_gain_vs_history += (gain - gain_history)
                        
                        # record相对于nothing和history的改善百分比
                        if gain_nothing != 0:
                            bridge_metrics['gains_vs_nothing_pct'].append((gain - gain_nothing) / abs(gain_nothing) * 100 if gain_nothing != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_nothing_pct'].append(0.0)
                        
                        if gain_history != 0:
                            bridge_metrics['gains_vs_history_pct'].append((gain - gain_history) / abs(gain_history) * 100 if gain_history != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_history_pct'].append(0.0)
                        
                        bridge_metrics['costs'].append(cost)
                        bridge_metrics['gains'].append(gain)
                        bridge_metrics['gains_nothing'].append(gain_nothing)
                        bridge_metrics['gains_history'].append(gain_history)
                        # 保留原始健康评分，便于和其他指标对齐
                        bridge_metrics['initial_health'].append(raw_h)
                
                # 累积指标
                ep_total_cost += step_cost
                ep_total_budget += year_budget_limit
                ep_total_health_gain += step_health_gain
                ep_total_health_gain_vs_nothing += step_health_gain_vs_nothing
                ep_total_health_gain_vs_history += step_health_gain_vs_history
                ep_steps += 1

                # record该年份的预算、花费和桥梁平均健康等级（使用模拟健康）
                avg_health_t = float(np.mean(sim_health[t, active_idx]))
                ep_yearly_budgets.append(float(year_budget_limit))
                ep_yearly_costs.append(float(step_cost))
                ep_yearly_avg_health.append(avg_health_t)
                
                total_metrics['total_cost'] += step_cost
                total_metrics['total_budget'] += year_budget_limit
                total_metrics['total_health_gain'] += step_health_gain
                total_metrics['total_health_gain_vs_nothing'] += step_health_gain_vs_nothing
                total_metrics['total_health_gain_vs_history'] += step_health_gain_vs_history
                total_metrics['steps'] += 1
                
                if step_cost > year_budget_limit + 1e-5:
                    total_metrics['budget_violations'] += 1

                # === 基于状态转移矩阵更新下一年的健康状态和年龄 ===
                if t + 1 <= n_years:
                    for a in active_idx:
                        cur_h = float(sim_health[t, a])
                        act_final = int(final_actions[a])
                        new_h = apply_health_transition(cur_h, act_final, transition_matrices)
                        sim_health[t + 1, a] = new_h
                        # 年龄每年+1
                        sim_age[t + 1, a] = sim_age[t, a] + 1.0
            
            # recordepisode级别指标
            episode_metrics.append({
                'episode': ep,
                'total_cost': float(ep_total_cost),
                'total_budget': float(ep_total_budget),
                'total_health_gain': float(ep_total_health_gain),
                'total_health_gain_vs_nothing': float(ep_total_health_gain_vs_nothing),
                'total_health_gain_vs_history': float(ep_total_health_gain_vs_history),
                'steps': ep_steps,
                # 每一年的预算、成本和平均健康等级
                'yearly_budgets': ep_yearly_budgets,
                'yearly_costs': ep_yearly_costs,
                'yearly_avg_health': ep_yearly_avg_health,
            })
    
    # 聚合结果
    n_steps = total_metrics['steps']
    if n_steps == 0:
        print("警告: 没有评估任何步骤")
        return None
    
    # 控制台打印每年平均健康等级与平均花费（在所有episode上的平均）
    # year_idx 从 0 到 n_years-1，对应第 1..n_years 年
    if episode_metrics and 'yearly_avg_health' in episode_metrics[0]:
        # 找到最长的年份长度（一般等于 n_years）
        max_years = max(len(ep['yearly_avg_health']) for ep in episode_metrics)
        print("\n按年份统计的平均健康等级和平均年度花费（所有episode平均）:")
        for year_idx in range(max_years):
            year_health_values = []
            year_cost_values = []
            for ep_info in episode_metrics:
                if year_idx < len(ep_info['yearly_avg_health']):
                    year_health_values.append(ep_info['yearly_avg_health'][year_idx])
                if 'yearly_costs' in ep_info and year_idx < len(ep_info['yearly_costs']):
                    year_cost_values.append(ep_info['yearly_costs'][year_idx])
            if year_health_values or year_cost_values:
                avg_health_year = float(np.mean(year_health_values)) if year_health_values else 0.0
                avg_cost_year = float(np.mean(year_cost_values)) if year_cost_values else 0.0
                print(f"  Year {year_idx + 1}: 平均健康等级 = {avg_health_year:.4f}, 平均花费 = ${avg_cost_year:.2f}")

    # compute动作分布百分比（整体）
    total_original_actions = sum(original_action_counts.values())
    total_final_actions = sum(final_action_counts.values())
    
    original_action_distribution = {
        action: (count / total_original_actions * 100) if total_original_actions > 0 else 0.0
        for action, count in original_action_counts.items()
    }
    final_action_distribution = {
        action: (count / total_final_actions * 100) if total_final_actions > 0 else 0.0
        for action, count in final_action_counts.items()
    }

    # compute不同健康等级下的原始动作偏好（百分比）
    health_level_original_action_distribution = {}
    for h, action_counts in health_original_action_counts.items():
        total_h_actions = sum(action_counts.values())
        if total_h_actions > 0:
            health_level_original_action_distribution[int(h)] = {
                int(a): (cnt / total_h_actions * 100.0)
                for a, cnt in action_counts.items()
            }
        else:
            health_level_original_action_distribution[int(h)] = {}

    # compute不同健康等级下的最终动作偏好（百分比）
    health_level_final_action_distribution = {}
    for h, action_counts in health_final_action_counts.items():
        total_h_actions = sum(action_counts.values())
        if total_h_actions > 0:
            health_level_final_action_distribution[int(h)] = {
                int(a): (cnt / total_h_actions * 100.0)
                for a, cnt in action_counts.items()
            }
        else:
            health_level_final_action_distribution[int(h)] = {}

    # compute原始数据集中，不同健康等级下“历史动作”的偏好（百分比）
    dataset_health_level_action_distribution = {}
    for h, action_counts in dataset_health_action_counts.items():
        total_h_actions = sum(action_counts.values())
        if total_h_actions > 0:
            dataset_health_level_action_distribution[int(h)] = {
                int(a): (cnt / total_h_actions * 100.0)
                for a, cnt in action_counts.items()
            }
        else:
            dataset_health_level_action_distribution[int(h)] = {}
    
    # compute相对于nothing和history的平均改善百分比
    bridge_avg_health_gain_vs_nothing_pct = float(np.mean(bridge_metrics['gains_vs_nothing_pct'])) if bridge_metrics['gains_vs_nothing_pct'] else 0.0
    bridge_avg_health_gain_vs_history_pct = float(np.mean(bridge_metrics['gains_vs_history_pct'])) if bridge_metrics['gains_vs_history_pct'] else 0.0
    
    results = {
        'model_name': os.path.basename(model_path),
        'algorithm_type': algorithm_type,
        'num_episodes': num_eps,
        'num_steps': n_steps,
        'mean_step_cost': float(total_metrics['total_cost'] / n_steps),
        'mean_step_budget': float(total_metrics['total_budget'] / n_steps),
        'budget_utilization': float(total_metrics['total_cost'] / total_metrics['total_budget']) if total_metrics['total_budget'] > 0 else 0.0,
        'total_health_gain': float(total_metrics['total_health_gain']),
        'mean_health_gain_per_step': float(total_metrics['total_health_gain'] / n_steps),
        'total_health_gain_vs_nothing': float(total_metrics['total_health_gain_vs_nothing']),
        'total_health_gain_vs_history': float(total_metrics['total_health_gain_vs_history']),
        'violations_count': int(total_metrics['budget_violations']),
        'violation_rate': float(total_metrics['budget_violations'] / n_steps) if n_steps > 0 else 0.0,
        'efficiency_gain_per_1k': float((total_metrics['total_health_gain'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'efficiency_gain_per_1k_vs_history': float((total_metrics['total_health_gain_vs_history'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'bridge_avg_cost': float(np.mean(bridge_metrics['costs'])) if bridge_metrics['costs'] else 0.0,
        'bridge_avg_health_gain': float(np.mean(bridge_metrics['gains'])) if bridge_metrics['gains'] else 0.0,
        'bridge_avg_health_gain_vs_nothing_pct': bridge_avg_health_gain_vs_nothing_pct,
        'bridge_avg_health_gain_vs_history_pct': bridge_avg_health_gain_vs_history_pct,
        'original_action_distribution': original_action_distribution,
        'final_action_distribution': final_action_distribution,
        'health_level_original_action_distribution': health_level_original_action_distribution,
        'health_level_final_action_distribution': health_level_final_action_distribution,
        # raw数据集中，不同 4 级健康状态下的历史动作比例
        'dataset_health_level_action_distribution': dataset_health_level_action_distribution,
        'episode_metrics': episode_metrics
    }
    
    print(f"评估完成:")
    print(f"  平均成本: ${results['mean_step_cost']:.2f}")
    print(f"  平均预算: ${results['mean_step_budget']:.2f}")
    print(f"  预算利用率: {results['budget_utilization']:.4f}")
    print(f"  违规次数: {results['violations_count']} ({results['violation_rate']:.4f})")
    print(f"  效率 (gain/$1k): {results['efficiency_gain_per_1k']:.4f}")
    print(f"  效率 vs History (gain/$1k): {results['efficiency_gain_per_1k_vs_history']:.4f}")
   
    print(f"  总健康改善 vs Nothing: {results['total_health_gain_vs_nothing']:.4f}")
    print(f"  总健康改善 vs History: {results['total_health_gain_vs_history']:.4f}")
    print(f"  桥梁平均健康改善 vs Nothing (%): {results['bridge_avg_health_gain_vs_nothing_pct']:.2f}%")
    print(f"  桥梁平均健康改善 vs History (%): {results['bridge_avg_health_gain_vs_history_pct']:.2f}%")
    
    return results

def evaluate_single_model_cdt(
    model_path,
    target_model,
    bc_model,
    test_data_dict,
    device,
    transition_matrices,
    controller,
    action_costs
):
    """
    CDT算法的专用评估函数
    使用BC模型生成预算，然后评估CDT模型在预算约束下的表现
    """
    print(f"\n{'='*60}")
    print(f"评估CDT模型: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    # 解包数据
    data = test_data_dict['data']          # [num_eps, T, n_agents, state_dim]
    health = test_data_dict['health']      # [num_eps, T+1, n_agents]
    actual_n_agents = test_data_dict['actual_n_agents']
    log_budgets = test_data_dict.get('log_budgets', None)
    actions_history = test_data_dict.get('actions', None)
    rewards = test_data_dict.get('rewards', None)
    raw_cost = test_data_dict.get('raw_cost', None)
    
    # handlelog_budgets维度
    if log_budgets is not None:
        if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
            log_budgets = log_budgets.squeeze(-1)  # [num_eps, T, n_agents]
    
    num_eps, T, max_n_agents, state_dim = data.shape
    
    # computereturns_to_go和costs_to_go（CDT需要）
    if rewards is None:
        print("警告: 没有rewards数据，无法计算returns_to_go，使用零值")
        returns_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    else:
        returns_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
        for ep in range(num_eps):
            for a in range(max_n_agents):
                rews = rewards[ep, :, a]  # [T]
                if rews.ndim > 1:
                    rews = rews.squeeze(-1)
                returns_to_go[ep, :, a] = rews[::-1].cumsum()[::-1]  # 反向累加
    
    if raw_cost is None:
        print("警告: 没有raw_cost数据，无法计算costs_to_go，使用零值")
        costs_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
    else:
        costs_to_go = np.zeros((num_eps, T, max_n_agents), dtype=np.float32)
        for ep in range(num_eps):
            for a in range(max_n_agents):
                costs = raw_cost[ep, :, a]  # [T]
                if costs.ndim > 1:
                    costs = costs.squeeze(-1)
                costs_to_go[ep, :, a] = costs[::-1].cumsum()[::-1]  # 反向累加
    
    # createtime_steps
    time_steps = np.zeros((num_eps, T, max_n_agents), dtype=np.int64)
    for ep in range(num_eps):
        for a in range(max_n_agents):
            time_steps[ep, :, a] = np.arange(T)
    
    # evaluate指标
    total_metrics = defaultdict(float)
    bridge_metrics = defaultdict(list)
    episode_metrics = []
    original_action_counts = defaultdict(int)
    final_action_counts = defaultdict(int)
    
    # 包装模型
    bc_algorithm_type = get_algorithm_type(BC_ALGORITHM_NAME) if BC_ALGORITHM_NAME in ALGO_TYPE else 'osrl'
    
    # 确定action_dim
    if actions_history is not None:
        action_dim = int(np.max(actions_history)) + 1
    else:
        action_dim = 4  # default4个动作
    
    with ModelWrapper(target_model, algorithm_type='cdt') as wrapped_target, \
         ModelWrapper(bc_model, algorithm_type=bc_algorithm_type) as wrapped_bc:
        
        # for每个episode（区域）进行评估
        for ep in tqdm(range(num_eps), desc=f"评估CDT {os.path.basename(model_path)}", unit="episode"):
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
            
            # createagent mask
            agent_mask = np.zeros(max_n_agents, dtype=np.float32)
            agent_mask[active_idx] = 1.0
            
            # Episode级别的指标
            ep_total_cost = 0.0
            ep_total_budget = 0.0
            ep_total_health_gain = 0.0
            ep_total_health_gain_vs_nothing = 0.0
            ep_total_health_gain_vs_history = 0.0
            ep_steps = 0
            
            # 为每个agent维护CDT历史序列
            cdt_histories = {}
            for a in active_idx:
                cdt_histories[a] = {
                    'states': [],
                    'actions': [],
                    'returns_to_go': [],
                    'costs_to_go': [],
                    'time_steps': []
                }
            
            # for每个时间步（年份）进行评估
            for t in range(T):
                # get当前状态
                obs_t = data[ep, t]  # [max_n_agents, state_dim]
                
                # ========== Step 1: 使用BC模型生成预算 ==========
                if bc_algorithm_type == 'marl':
                    obs_for_bc = obs_t.copy()
                    bc_needs_budget_in_obs = getattr(bc_model, 'has_budget_in_obs', True)
                    if bc_needs_budget_in_obs and log_budgets is not None:
                        budget_t = log_budgets[ep, t]
                        if budget_t.ndim == 1:
                            budget_t = budget_t[:, np.newaxis]
                        obs_for_bc = np.concatenate([obs_t, budget_t], axis=-1)
                    
                    obs_batch = obs_for_bc.reshape(1, max_n_agents, -1)
                    try:
                        bc_actions = wrapped_bc.act(
                            obs=obs_batch,
                            agent_mask=agent_mask.reshape(1, -1),
                            training=False
                        )
                        if isinstance(bc_actions, (tuple, list)):
                            bc_actions = bc_actions[0]
                        bc_actions = np.asarray(bc_actions)
                        if bc_actions.ndim == 2:
                            bc_actions = bc_actions[0]
                        if bc_actions.ndim == 0:
                            bc_actions = np.array([bc_actions])
                        bc_actions = bc_actions.astype(int)
                        if len(bc_actions) < max_n_agents:
                            bc_actions_full = np.zeros(max_n_agents, dtype=int)
                            bc_actions_full[:len(bc_actions)] = bc_actions
                            bc_actions = bc_actions_full
                        elif len(bc_actions) > max_n_agents:
                            bc_actions = bc_actions[:max_n_agents]
                    except Exception as e:
                        print(f"BC模型调用失败 (MARL): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                else:
                    # OSRL BC模型
                    obs_batch = obs_t[active_idx]
                    if log_budgets is not None:
                        budget_batch = log_budgets[ep, t, active_idx]
                        if budget_batch.ndim > 1:
                            budget_batch = budget_batch.squeeze(-1)
                    else:
                        budget_batch = np.zeros(len(active_idx))
                    
                    try:
                        bc_actions_batch = wrapped_bc.act(obs_batch, budget_batch, deterministic=True)
                        if isinstance(bc_actions_batch, (tuple, list)):
                            bc_actions_batch = bc_actions_batch[0]
                        bc_actions_batch = np.asarray(bc_actions_batch)
                        if bc_actions_batch.ndim == 0:
                            bc_actions_batch = np.array([bc_actions_batch])
                        
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                        for i, a_idx in enumerate(active_idx):
                            bc_actions[a_idx] = int(bc_actions_batch[i])
                    except Exception as e:
                        print(f"BC模型调用失败 (OSRL): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                
                # compute该年份本区域的总预算
                year_budget_limit = 0.0
                for a in active_idx:
                    act = int(bc_actions[a])
                    year_budget_limit += get_action_cost(act)

                # 放宽到 BC 预算的 1.2 倍
                year_budget_limit *= 1.2
                # ========== Step 2: 使用CDT模型获取动作 ==========
                target_actions = np.zeros(max_n_agents, dtype=int)
                
                for a in active_idx:
                    # updateCDT历史序列
                    obs_a = obs_t[a]  # [state_dim]
                    ret_a = returns_to_go[ep, t, a]
                    cost_a = costs_to_go[ep, t, a]
                    time_a = t
                    
                    cdt_histories[a]['states'].append(obs_a)
                    cdt_histories[a]['returns_to_go'].append(ret_a)
                    cdt_histories[a]['costs_to_go'].append(cost_a)
                    cdt_histories[a]['time_steps'].append(time_a)
                    
                    # limit序列长度
                    max_seq_len = 10
                    if len(cdt_histories[a]['states']) > max_seq_len:
                        for key in cdt_histories[a]:
                            cdt_histories[a][key] = cdt_histories[a][key][-max_seq_len:]
                    
                    # build动作序列（历史动作 + 当前零动作）
                    if len(cdt_histories[a]['actions']) == 0:
                        actions_seq = np.zeros((len(cdt_histories[a]['states']), action_dim), dtype=np.float32)
                    else:
                        actions_seq = np.array(cdt_histories[a]['actions'])
                        if actions_seq.shape[0] < len(cdt_histories[a]['states']):
                            padding = np.zeros((len(cdt_histories[a]['states']) - actions_seq.shape[0], action_dim), dtype=np.float32)
                            actions_seq = np.vstack([actions_seq, padding])
                        elif actions_seq.shape[0] > len(cdt_histories[a]['states']):
                            actions_seq = actions_seq[:len(cdt_histories[a]['states'])]
                    
                    # build输入张量
                    states_tensor = th.tensor(np.array(cdt_histories[a]['states']), dtype=th.float32, device=device)
                    actions_tensor = th.tensor(actions_seq, dtype=th.float32, device=device)
                    returns_tensor = th.tensor(np.array(cdt_histories[a]['returns_to_go']), dtype=th.float32, device=device)
                    costs_tensor = th.tensor(np.array(cdt_histories[a]['costs_to_go']), dtype=th.float32, device=device)
                    time_tensor = th.tensor(np.array(cdt_histories[a]['time_steps']), dtype=th.long, device=device)
                    
                    # 调用CDT模型
                    try:
                        with th.no_grad():
                            pred = target_model.act(
                                states=states_tensor,
                                actions=actions_tensor,
                                returns_to_go=returns_tensor,
                                costs_to_go=costs_tensor,
                                time_steps=time_tensor,
                                deterministic=True
                            )
                        
                        # handle输出
                        if isinstance(pred, (tuple, list)):
                            pred = pred[0]
                        if hasattr(pred, 'detach'):
                            pred = pred.detach()
                        if hasattr(pred, 'cpu'):
                            pred = pred.cpu()
                        if hasattr(pred, 'numpy'):
                            pred = pred.numpy()
                        
                        # CDT返回的可能是连续动作值，需要转换为离散动作
                        # if返回的是概率分布，使用argmax
                        # if返回的是连续值，需要根据动作空间进行离散化
                        if isinstance(pred, np.ndarray):
                            if pred.ndim == 0:
                                # scalar，直接转换为整数
                                act = int(np.clip(pred, 0, action_dim - 1))
                            elif pred.ndim == 1:
                                # 一维数组，可能是概率分布或动作值
                                if len(pred) == action_dim:
                                    # probability分布，使用argmax
                                    act = int(np.argmax(pred))
                                else:
                                    # 单个动作值，需要离散化
                                    act = int(np.clip(pred[0] if len(pred) > 0 else 0, 0, action_dim - 1))
                            elif pred.ndim == 2:
                                # 二维数组，取最后一个时间步
                                if pred.shape[0] == 1:
                                    act = int(np.argmax(pred[0]) if pred.shape[1] == action_dim else int(np.clip(pred[0, 0], 0, action_dim - 1)))
                                else:
                                    act = int(np.argmax(pred[-1]) if pred.shape[1] == action_dim else int(np.clip(pred[-1, 0], 0, action_dim - 1)))
                            else:
                                # 多维数组，展平后处理
                                act = int(np.clip(pred.reshape(-1)[0], 0, action_dim - 1))
                        else:
                            # scalar或其他类型
                            act = int(np.clip(float(pred), 0, action_dim - 1))
                        
                        target_actions[a] = act
                        
                        # will选择的动作添加到历史中（one-hot）
                        action_one_hot = np.zeros(action_dim, dtype=np.float32)
                        action_one_hot[act] = 1.0
                        cdt_histories[a]['actions'].append(action_one_hot)
                        
                    except Exception as e:
                        print(f"CDT模型调用失败 (agent {a}): {e}")
                        import traceback
                        traceback.print_exc()
                        target_actions[a] = 0
                
                # 应用动作屏蔽：检查并修正无效动作
                for a in active_idx:
                    if health is not None and t < health.shape[1]:
                        raw_h = int(health[ep, t, a])
                        h_cat = project_health_to_categories(raw_h)
                        act = int(target_actions[a])
                        
                        # 屏蔽无效动作
                        if h_cat == 0:  # poor(0)状态
                            # 屏蔽action2和action3
                            if act == 2 or act == 3:
                                target_actions[a] = 0  # 改为不做动作
                        elif h_cat == 3:  # excellent(3)状态
                            # 屏蔽action1
                            if act == 1:
                                target_actions[a] = 0  # 改为不做动作
                
                # record原始动作分布
                for a in active_idx:
                    original_action_counts[int(target_actions[a])] += 1
                
                # ========== Step 3: 检查预算并应用启发式控制器 ==========
                current_cost = sum([get_action_cost(int(target_actions[a])) for a in active_idx])
                
                if current_cost > year_budget_limit + 1e-5:
                    # getQ值用于优先级排序（CDT可能没有get_q_values，需要处理）
                    q_values = None
                    try:
                        # 尝试获取Q值（如果CDT模型有这个方法）
                        if hasattr(target_model, 'get_q_values'):
                            q_values_list = []
                            for a_idx in active_idx:
                                # buildCDT输入
                                states_t = th.tensor(np.array(cdt_histories[a_idx]['states']), dtype=th.float32, device=device)
                                actions_t = th.tensor(np.array(cdt_histories[a_idx]['actions']), dtype=th.float32, device=device)
                                returns_t = th.tensor(np.array(cdt_histories[a_idx]['returns_to_go']), dtype=th.float32, device=device)
                                costs_t = th.tensor(np.array(cdt_histories[a_idx]['costs_to_go']), dtype=th.float32, device=device)
                                time_t = th.tensor(np.array(cdt_histories[a_idx]['time_steps']), dtype=th.long, device=device)
                                
                                # getQ值（如果CDT支持）
                                q_vals = target_model.get_q_values(states_t, actions_t, returns_t, costs_t, time_t)
                                # 转换为numpy数组
                                if isinstance(q_vals, th.Tensor):
                                    q_vals = q_vals.cpu().numpy()
                                q_values_list.append(q_vals)
                            
                            # willQ值列表转换为数组格式
                            if q_values_list:
                                # 假设所有Q值都有相同的形状
                                if isinstance(q_values_list[0], np.ndarray):
                                    q_values = np.array(q_values_list)  # [n_active, n_actions] 或类似格式
                                else:
                                    q_values = None
                    except:
                        pass
                    
                    # if获取了Q值，应用屏蔽
                    if q_values is not None:
                        # get健康类别用于屏蔽
                        health_categories = []
                        for a_idx in active_idx:
                            if health is not None and t < health.shape[1]:
                                raw_h = int(health[ep, t, a_idx])
                                h_cat = project_health_to_categories(raw_h)
                                health_categories.append(h_cat)
                            else:
                                health_categories.append(0)  # default值
                        
                        # 应用屏蔽
                        q_values = mask_invalid_q_values(q_values, health_categories, active_idx)
                    
                    # get当前所有Agent的健康状态，传给Controller
                    current_health_states = health[ep, t] if health is not None else np.zeros(max_n_agents)

                    # 应用启发式预算约束
                    final_actions = controller.enforce_budget(
                        actions=target_actions,
                        q_values=q_values,
                        budget_limit=year_budget_limit,
                        active_indices=active_idx,
                        health_states=current_health_states
                    )
                else:
                    final_actions = target_actions.copy()
                
                # record最终动作分布
                for a in active_idx:
                    final_action_counts[int(final_actions[a])] += 1
                
                # ========== Step 4: 评估指标 ==========
                step_cost = 0.0
                step_health_gain = 0.0
                step_health_gain_vs_nothing = 0.0
                step_health_gain_vs_history = 0.0
                
                for a in active_idx:
                    act = int(final_actions[a])
                    cost = get_action_cost(act)
                    step_cost += cost
                    
                    # 健康改善计算
                    if health is not None and t + 1 < health.shape[1]:
                        cur_h = int(health[ep, t, a])
                        
                        gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                        gain_nothing = calculate_expected_health_improvement(cur_h, 0, transition_matrices)
                        
                        gain_history = 0.0
                        if actions_history is not None and t < actions_history.shape[1]:
                            hist_act = int(actions_history[ep, t, a])
                            gain_history = calculate_expected_health_improvement(cur_h, hist_act, transition_matrices)
                        
                        step_health_gain += gain
                        step_health_gain_vs_nothing += (gain - gain_nothing)
                        step_health_gain_vs_history += (gain - gain_history)
                        
                        # record相对于nothing和history的改善百分比
                        if gain_nothing != 0:
                            bridge_metrics['gains_vs_nothing_pct'].append((gain - gain_nothing) / abs(gain_nothing) * 100 if gain_nothing != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_nothing_pct'].append(0.0)
                        
                        if gain_history != 0:
                            bridge_metrics['gains_vs_history_pct'].append((gain - gain_history) / abs(gain_history) * 100 if gain_history != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_history_pct'].append(0.0)
                        
                        bridge_metrics['costs'].append(cost)
                        bridge_metrics['gains'].append(gain)
                        bridge_metrics['gains_nothing'].append(gain_nothing)
                        bridge_metrics['gains_history'].append(gain_history)
                        bridge_metrics['initial_health'].append(cur_h)
                
                # 累积指标
                ep_total_cost += step_cost
                ep_total_budget += year_budget_limit
                ep_total_health_gain += step_health_gain
                ep_total_health_gain_vs_nothing += step_health_gain_vs_nothing
                ep_total_health_gain_vs_history += step_health_gain_vs_history
                ep_steps += 1
                
                total_metrics['total_cost'] += step_cost
                total_metrics['total_budget'] += year_budget_limit
                total_metrics['total_health_gain'] += step_health_gain
                total_metrics['total_health_gain_vs_nothing'] += step_health_gain_vs_nothing
                total_metrics['total_health_gain_vs_history'] += step_health_gain_vs_history
                total_metrics['steps'] += 1
                
                if step_cost > year_budget_limit + 1e-5:
                    total_metrics['budget_violations'] += 1
            
            # recordepisode级别指标
            episode_metrics.append({
                'episode': ep,
                'total_cost': float(ep_total_cost),
                'total_budget': float(ep_total_budget),
                'total_health_gain': float(ep_total_health_gain),
                'total_health_gain_vs_nothing': float(ep_total_health_gain_vs_nothing),
                'total_health_gain_vs_history': float(ep_total_health_gain_vs_history),
                'steps': ep_steps
            })
    
    # 聚合结果
    n_steps = total_metrics['steps']
    if n_steps == 0:
        print("警告: 没有评估任何步骤")
        return None
    
    # compute动作分布百分比
    total_original_actions = sum(original_action_counts.values())
    total_final_actions = sum(final_action_counts.values())
    
    original_action_distribution = {
        action: (count / total_original_actions * 100) if total_original_actions > 0 else 0.0
        for action, count in original_action_counts.items()
    }
    final_action_distribution = {
        action: (count / total_final_actions * 100) if total_final_actions > 0 else 0.0
        for action, count in final_action_counts.items()
    }
    
    # compute相对于nothing和history的平均改善百分比
    bridge_avg_health_gain_vs_nothing_pct = float(np.mean(bridge_metrics['gains_vs_nothing_pct'])) if bridge_metrics['gains_vs_nothing_pct'] else 0.0
    bridge_avg_health_gain_vs_history_pct = float(np.mean(bridge_metrics['gains_vs_history_pct'])) if bridge_metrics['gains_vs_history_pct'] else 0.0
    
    results = {
        'model_name': os.path.basename(model_path),
        'algorithm_type': 'cdt',
        'num_episodes': num_eps,
        'num_steps': n_steps,
        'mean_step_cost': float(total_metrics['total_cost'] / n_steps),
        'mean_step_budget': float(total_metrics['total_budget'] / n_steps),
        'budget_utilization': float(total_metrics['total_cost'] / total_metrics['total_budget']) if total_metrics['total_budget'] > 0 else 0.0,
        'total_health_gain': float(total_metrics['total_health_gain']),
        'mean_health_gain_per_step': float(total_metrics['total_health_gain'] / n_steps),
        'total_health_gain_vs_nothing': float(total_metrics['total_health_gain_vs_nothing']),
        'total_health_gain_vs_history': float(total_metrics['total_health_gain_vs_history']),
        'violations_count': int(total_metrics['budget_violations']),
        'violation_rate': float(total_metrics['budget_violations'] / n_steps) if n_steps > 0 else 0.0,
        'efficiency_gain_per_1k': float((total_metrics['total_health_gain'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'efficiency_gain_per_1k_vs_history': float((total_metrics['total_health_gain_vs_history'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'bridge_avg_cost': float(np.mean(bridge_metrics['costs'])) if bridge_metrics['costs'] else 0.0,
        'bridge_avg_health_gain': float(np.mean(bridge_metrics['gains'])) if bridge_metrics['gains'] else 0.0,
        'bridge_avg_health_gain_vs_nothing_pct': bridge_avg_health_gain_vs_nothing_pct,
        'bridge_avg_health_gain_vs_history_pct': bridge_avg_health_gain_vs_history_pct,
        'original_action_distribution': original_action_distribution,
        'final_action_distribution': final_action_distribution,
        'episode_metrics': episode_metrics
    }
    
    print(f"评估完成:")
    print(f"  平均成本: ${results['mean_step_cost']:.2f}")
    print(f"  平均预算: ${results['mean_step_budget']:.2f}")
    print(f"  预算利用率: {results['budget_utilization']:.4f}")
    print(f"  违规次数: {results['violations_count']} ({results['violation_rate']:.4f})")
    print(f"  效率 (gain/$1k): {results['efficiency_gain_per_1k']:.4f}")
    print(f"  效率 vs History (gain/$1k): {results['efficiency_gain_per_1k_vs_history']:.4f}")
    print(f"  总健康改善 vs Nothing: {results['total_health_gain_vs_nothing']:.4f}")
    print(f"  总健康改善 vs History: {results['total_health_gain_vs_history']:.4f}")
    print(f"  桥梁平均健康改善 vs Nothing (%): {results['bridge_avg_health_gain_vs_nothing_pct']:.2f}%")
    print(f"  桥梁平均健康改善 vs History (%): {results['bridge_avg_health_gain_vs_history_pct']:.2f}%")
    
    return results


def evaluate_heuristic_algorithm_before(
    bc_model,
    test_data_dict,
    device,
    transition_matrices,
    action_costs
):
    """
    评估基于health状态的启发式算法
    策略：
    - Poor (health=0): 使用动作2 (major_repair)
    - Fair (health=1): 如果有预算，使用动作2 (major_repair)，否则动作1 (minor_repair)
    - Good (health=2): 如果有剩余预算，使用动作1 (minor_repair)
    - Excellent (health=3): 不做任何动作 (动作0)
    """
    print(f"\n{'='*60}")
    print(f"评估启发式算法（基于health状态）")
    print(f"{'='*60}")
    
    # 解包数据
    data = test_data_dict['data']          # [num_eps, T, n_agents, state_dim]
    health = test_data_dict['health']      # [num_eps, T+1, n_agents]
    actual_n_agents = test_data_dict['actual_n_agents']
    log_budgets = test_data_dict.get('log_budgets', None)
    actions_history = test_data_dict.get('actions', None)
    
    # handlelog_budgets维度
    if log_budgets is not None:
        if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
            log_budgets = log_budgets.squeeze(-1)  # [num_eps, T, n_agents]
    
    num_eps, T, max_n_agents, state_dim = data.shape
    
    if health is None:
        print("错误: 需要health数据来运行启发式算法")
        return None
    
    # evaluate指标
    total_metrics = defaultdict(float)
    bridge_metrics = defaultdict(list)
    episode_metrics = []
    final_action_counts = defaultdict(int)
    
    # 包装BC模型（用于生成预算）
    bc_algorithm_type = get_algorithm_type(BC_ALGORITHM_NAME) if BC_ALGORITHM_NAME in ALGO_TYPE else 'osrl'
    
    with ModelWrapper(bc_model, algorithm_type=bc_algorithm_type) as wrapped_bc:
        
        # for每个episode（区域）进行评估
        for ep in tqdm(range(num_eps), desc="评估启发式算法", unit="episode"):
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
            
            # createagent mask
            agent_mask = np.zeros(max_n_agents, dtype=np.float32)
            agent_mask[active_idx] = 1.0
            
            # Episode级别的指标
            ep_total_cost = 0.0
            ep_total_budget = 0.0
            ep_total_health_gain = 0.0
            ep_total_health_gain_vs_nothing = 0.0
            ep_total_health_gain_vs_history = 0.0
            ep_steps = 0
            
            # for每个时间步（年份）进行评估
            for t in range(T):
                # get当前状态
                obs_t = data[ep, t]  # [max_n_agents, state_dim]
                
                # ========== Step 1: 使用BC模型生成预算 ==========
                if bc_algorithm_type == 'marl':
                    obs_for_bc = obs_t.copy()
                    bc_needs_budget_in_obs = getattr(bc_model, 'has_budget_in_obs', True)
                    if bc_needs_budget_in_obs and log_budgets is not None:
                        budget_t = log_budgets[ep, t]
                        if budget_t.ndim == 1:
                            budget_t = budget_t[:, np.newaxis]
                        obs_for_bc = np.concatenate([obs_t, budget_t], axis=-1)
                    
                    obs_batch = obs_for_bc.reshape(1, max_n_agents, -1)
                    try:
                        bc_actions = wrapped_bc.act(
                            obs=obs_batch,
                            agent_mask=agent_mask.reshape(1, -1),
                            training=False
                        )
                        if isinstance(bc_actions, (tuple, list)):
                            bc_actions = bc_actions[0]
                        bc_actions = np.asarray(bc_actions)
                        if bc_actions.ndim == 2:
                            bc_actions = bc_actions[0]
                        if bc_actions.ndim == 0:
                            bc_actions = np.array([bc_actions])
                        bc_actions = bc_actions.astype(int)
                        if len(bc_actions) < max_n_agents:
                            bc_actions_full = np.zeros(max_n_agents, dtype=int)
                            bc_actions_full[:len(bc_actions)] = bc_actions
                            bc_actions = bc_actions_full
                        elif len(bc_actions) > max_n_agents:
                            bc_actions = bc_actions[:max_n_agents]
                    except Exception as e:
                        print(f"BC模型调用失败 (MARL): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                else:
                    # OSRL BC模型
                    obs_batch = obs_t[active_idx]
                    if log_budgets is not None:
                        budget_batch = log_budgets[ep, t, active_idx]
                        if budget_batch.ndim > 1:
                            budget_batch = budget_batch.squeeze(-1)
                    else:
                        budget_batch = np.zeros(len(active_idx))
                    
                    try:
                        bc_actions_batch = wrapped_bc.act(obs_batch, budget_batch, deterministic=True)
                        if isinstance(bc_actions_batch, (tuple, list)):
                            bc_actions_batch = bc_actions_batch[0]
                        bc_actions_batch = np.asarray(bc_actions_batch)
                        if bc_actions_batch.ndim == 0:
                            bc_actions_batch = np.array([bc_actions_batch])
                        
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                        for i, a_idx in enumerate(active_idx):
                            bc_actions[a_idx] = int(bc_actions_batch[i])
                    except Exception as e:
                        print(f"BC模型调用失败 (OSRL): {e}")
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                
                # compute该年份本区域的总预算
                year_budget_limit = 0.0
                for a in active_idx:
                    act = int(bc_actions[a])
                    year_budget_limit += get_action_cost(act)

                # 放宽到 BC 预算的 1.2 倍
                year_budget_limit *= 1.2
                
                # ========== Step 2: 使用启发式算法决定动作 ==========
                # 收集所有active agents的health信息
                agent_list = []
                for a in active_idx:
                    cur_h = int(health[ep, t, a])
                    agent_list.append({
                        'agent_id': a,
                        'health': cur_h
                    })
                
                # byhealth从小到大排序
                agent_list.sort(key=lambda x: x['health'])
                
                # init所有动作为0
                heuristic_actions = np.zeros(max_n_agents, dtype=int)
                remaining_budget = year_budget_limit
                
                # by优先级分配动作
                for agent_info in agent_list:
                    a = agent_info['agent_id']
                    cur_h = agent_info['health']
                    
                    if cur_h == 0:  # Poor: 优先使用动作2 (major_repair)
                        action = 2
                        cost = get_action_cost(action)
                        if remaining_budget >= cost:
                            heuristic_actions[a] = action
                            remaining_budget -= cost
                        # if预算不足，保持为0
                    elif cur_h == 1:  # Fair: 优先动作2，否则动作1
                        action = 2
                        cost = get_action_cost(action)
                        if remaining_budget >= cost:
                            heuristic_actions[a] = action
                            remaining_budget -= cost
                        else:
                            # 尝试使用动作1 (minor_repair)
                            action = 1
                            cost = get_action_cost(action)
                            if remaining_budget >= cost:
                                heuristic_actions[a] = action
                                remaining_budget -= cost
                    elif cur_h == 2:  # Good: 如果还有预算，使用动作1 (minor_repair)
                        action = 1
                        cost = get_action_cost(action)
                        if remaining_budget >= cost:
                            heuristic_actions[a] = action
                            remaining_budget -= cost
                    # cur_h == 3 (Excellent): 不做任何动作，保持为0
                
                final_actions = heuristic_actions
                
                # ========== Step 3: 评估指标 ==========
                step_cost = 0.0
                step_health_gain = 0.0
                step_health_gain_vs_nothing = 0.0
                step_health_gain_vs_history = 0.0
                
                # record最终动作分布
                for a in active_idx:
                    final_action_counts[int(final_actions[a])] += 1
                
                for a in active_idx:
                    act = int(final_actions[a])
                    cost = get_action_cost(act)
                    step_cost += cost
                    
                    # 健康改善计算
                    if health is not None and t + 1 < health.shape[1]:
                        cur_h = int(health[ep, t, a])
                        
                        # expected健康改善
                        gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                        gain_nothing = calculate_expected_health_improvement(cur_h, 0, transition_matrices)
                        
                        # 相对于历史动作的改善
                        gain_history = 0.0
                        if actions_history is not None and t < actions_history.shape[1]:
                            hist_act = int(actions_history[ep, t, a])
                            gain_history = calculate_expected_health_improvement(cur_h, hist_act, transition_matrices)
                        
                        step_health_gain += gain
                        step_health_gain_vs_nothing += (gain - gain_nothing)
                        step_health_gain_vs_history += (gain - gain_history)
                        
                        # record相对于nothing和history的改善百分比
                        if gain_nothing != 0:
                            bridge_metrics['gains_vs_nothing_pct'].append((gain - gain_nothing) / abs(gain_nothing) * 100 if gain_nothing != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_nothing_pct'].append(0.0)
                        
                        if gain_history != 0:
                            bridge_metrics['gains_vs_history_pct'].append((gain - gain_history) / abs(gain_history) * 100 if gain_history != 0 else 0.0)
                        else:
                            bridge_metrics['gains_vs_history_pct'].append(0.0)
                        
                        bridge_metrics['costs'].append(cost)
                        bridge_metrics['gains'].append(gain)
                        bridge_metrics['gains_nothing'].append(gain_nothing)
                        bridge_metrics['gains_history'].append(gain_history)
                        bridge_metrics['initial_health'].append(cur_h)
                
                # 累积指标
                ep_total_cost += step_cost
                ep_total_budget += year_budget_limit
                ep_total_health_gain += step_health_gain
                ep_total_health_gain_vs_nothing += step_health_gain_vs_nothing
                ep_total_health_gain_vs_history += step_health_gain_vs_history
                ep_steps += 1
                
                total_metrics['total_cost'] += step_cost
                total_metrics['total_budget'] += year_budget_limit
                total_metrics['total_health_gain'] += step_health_gain
                total_metrics['total_health_gain_vs_nothing'] += step_health_gain_vs_nothing
                total_metrics['total_health_gain_vs_history'] += step_health_gain_vs_history
                total_metrics['steps'] += 1
                
                if step_cost > year_budget_limit + 1e-5:
                    total_metrics['budget_violations'] += 1
            
            # recordepisode级别指标
            episode_metrics.append({
                'episode': ep,
                'total_cost': float(ep_total_cost),
                'total_budget': float(ep_total_budget),
                'total_health_gain': float(ep_total_health_gain),
                'total_health_gain_vs_nothing': float(ep_total_health_gain_vs_nothing),
                'total_health_gain_vs_history': float(ep_total_health_gain_vs_history),
                'steps': ep_steps
            })
    
    # 聚合结果
    n_steps = total_metrics['steps']
    if n_steps == 0:
        print("警告: 没有评估任何步骤")
        return None
    
    # compute动作分布百分比
    total_final_actions = sum(final_action_counts.values())
    
    final_action_distribution = {
        action: (count / total_final_actions * 100) if total_final_actions > 0 else 0.0
        for action, count in final_action_counts.items()
    }
    
    # compute相对于nothing和history的平均改善百分比
    bridge_avg_health_gain_vs_nothing_pct = float(np.mean(bridge_metrics['gains_vs_nothing_pct'])) if bridge_metrics['gains_vs_nothing_pct'] else 0.0
    bridge_avg_health_gain_vs_history_pct = float(np.mean(bridge_metrics['gains_vs_history_pct'])) if bridge_metrics['gains_vs_history_pct'] else 0.0
    
    results = {
        'model_name': 'heuristic_health_based',
        'algorithm_type': 'heuristic',
        'num_episodes': num_eps,
        'num_steps': n_steps,
        'mean_step_cost': float(total_metrics['total_cost'] / n_steps),
        'mean_step_budget': float(total_metrics['total_budget'] / n_steps),
        'budget_utilization': float(total_metrics['total_cost'] / total_metrics['total_budget']) if total_metrics['total_budget'] > 0 else 0.0,
        'total_health_gain': float(total_metrics['total_health_gain']),
        'mean_health_gain_per_step': float(total_metrics['total_health_gain'] / n_steps),
        'total_health_gain_vs_nothing': float(total_metrics['total_health_gain_vs_nothing']),
        'total_health_gain_vs_history': float(total_metrics['total_health_gain_vs_history']),
        'violations_count': int(total_metrics['budget_violations']),
        'violation_rate': float(total_metrics['budget_violations'] / n_steps) if n_steps > 0 else 0.0,
        'efficiency_gain_per_1k': float((total_metrics['total_health_gain'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'efficiency_gain_per_1k_vs_history': float((total_metrics['total_health_gain_vs_history'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'bridge_avg_cost': float(np.mean(bridge_metrics['costs'])) if bridge_metrics['costs'] else 0.0,
        'bridge_avg_health_gain': float(np.mean(bridge_metrics['gains'])) if bridge_metrics['gains'] else 0.0,
        'bridge_avg_health_gain_vs_nothing_pct': bridge_avg_health_gain_vs_nothing_pct,
        'bridge_avg_health_gain_vs_history_pct': bridge_avg_health_gain_vs_history_pct,
        'final_action_distribution': final_action_distribution,
        'episode_metrics': episode_metrics
    }
    
    print(f"评估完成:")
    print(f"  平均成本: ${results['mean_step_cost']:.2f}")
    print(f"  平均预算: ${results['mean_step_budget']:.2f}")
    print(f"  预算利用率: {results['budget_utilization']:.4f}")
    print(f"  违规次数: {results['violations_count']} ({results['violation_rate']:.4f})")
    print(f"  效率 (gain/$1k): {results['efficiency_gain_per_1k']:.4f}")
    print(f"  效率 vs History (gain/$1k): {results['efficiency_gain_per_1k_vs_history']:.4f}")
    print(f"  总健康改善 vs Nothing: {results['total_health_gain_vs_nothing']:.4f}")
    print(f"  总健康改善 vs History: {results['total_health_gain_vs_history']:.4f}")
    print(f"  桥梁平均健康改善 vs Nothing (%): {results['bridge_avg_health_gain_vs_nothing_pct']:.2f}%")
    print(f"  桥梁平均健康改善 vs History (%): {results['bridge_avg_health_gain_vs_history_pct']:.2f}%")
    print(f"\n动作分布:")
    print(f"  总动作数: {total_final_actions}")
    for action in sorted(final_action_distribution.keys()):
        count = final_action_counts[action]
        percentage = final_action_distribution[action]
        print(f"  Action {action}: {count} ({percentage:.2f}%)")
    
    return results

# 请确保在文件顶部的配置区域更新 ACTION_COSTS
# ACTION_COSTS = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}

def evaluate_heuristic_algorithm(
    bc_model,
    test_data_dict,
    device,
    transition_matrices,
    action_costs,
    budget_factor=1.0
):
    """
    评估基于健康状态映射的启发式算法（考虑预算）
    
    成本参考:
    Action 1 (Minor): $1148.81
    Action 2 (Major): $2317.70
    Action 3 (Replc): $3004.33
    
    策略逻辑（带预算约束）:
    - critical/poor -> Action 1
    - fair -> Action 2
    - good -> Action 0
    若预算不足，则该桥梁回退为 Action 0。
    """
    print(f"\n{'='*60}")
    print(f"评估启发式算法（Health Mapping + Budget Constraint）")
    print(f"{'='*60}")
    
    # 解包数据
    data = test_data_dict['data']          
    health = test_data_dict['health']      
    actual_n_agents = test_data_dict['actual_n_agents']
    log_budgets = test_data_dict.get('log_budgets', None)
    actions_history = test_data_dict.get('actions', None)
    
    # handlelog_budgets维度
    if log_budgets is not None:
        if log_budgets.ndim == 4 and log_budgets.shape[-1] == 1:
            log_budgets = log_budgets.squeeze(-1)
    
    num_eps, T, max_n_agents, state_dim = data.shape
    
    if health is None:
        print("错误: 需要health数据来运行启发式算法")
        return None
    
    # evaluate指标
    total_metrics = defaultdict(float)
    bridge_metrics = defaultdict(list)
    episode_metrics = []
    final_action_counts = defaultdict(int)
    
    # 包装BC模型（用于生成预算）
    bc_algorithm_type = get_algorithm_type(BC_ALGORITHM_NAME) if BC_ALGORITHM_NAME in ALGO_TYPE else 'osrl'
    
    # ensure使用正确的成本 (硬编码以防外部传入旧字典)
    current_action_costs = {0: 0, 1: 1148.81, 2: 2317.70, 3: 3004.33}
    
    with ModelWrapper(bc_model, algorithm_type=bc_algorithm_type) as wrapped_bc:
        
        # for每个episode（区域）进行评估
        for ep in tqdm(range(num_eps), desc="评估启发式算法", unit="episode"):
            ep_actual_n = int(actual_n_agents[ep]) if hasattr(actual_n_agents, '__len__') else int(actual_n_agents)
            active_idx = list(range(ep_actual_n))
            
            # createagent mask
            agent_mask = np.zeros(max_n_agents, dtype=np.float32)
            agent_mask[active_idx] = 1.0
            
            ep_total_cost = 0.0
            ep_total_budget = 0.0
            ep_total_health_gain = 0.0
            ep_total_health_gain_vs_nothing = 0.0
            ep_total_health_gain_vs_history = 0.0
            ep_steps = 0
            
            for t in range(T):
                obs_t = data[ep, t]
                
                # ========== Step 1: 生成预算 (逻辑保持不变) ==========
                if bc_algorithm_type == 'marl':
                    obs_for_bc = obs_t.copy()
                    bc_needs_budget_in_obs = getattr(bc_model, 'has_budget_in_obs', True)
                    if bc_needs_budget_in_obs and log_budgets is not None:
                        budget_t = log_budgets[ep, t]
                        if budget_t.ndim == 1:
                            budget_t = budget_t[:, np.newaxis]
                        obs_for_bc = np.concatenate([obs_t, budget_t], axis=-1)
                    
                    obs_batch = obs_for_bc.reshape(1, max_n_agents, -1)
                    try:
                        bc_actions = wrapped_bc.act(obs_batch, agent_mask.reshape(1, -1), training=False)
                        if isinstance(bc_actions, (tuple, list)): bc_actions = bc_actions[0]
                        bc_actions = np.asarray(bc_actions)
                        if bc_actions.ndim == 2: bc_actions = bc_actions[0]
                        if bc_actions.ndim == 0: bc_actions = np.array([bc_actions])
                        bc_actions = bc_actions.astype(int)
                        if len(bc_actions) < max_n_agents:
                            bc_actions_full = np.zeros(max_n_agents, dtype=int)
                            bc_actions_full[:len(bc_actions)] = bc_actions
                            bc_actions = bc_actions_full
                        elif len(bc_actions) > max_n_agents:
                            bc_actions = bc_actions[:max_n_agents]
                    except Exception:
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                else:
                    obs_batch = obs_t[active_idx]
                    if log_budgets is not None:
                        budget_batch = log_budgets[ep, t, active_idx]
                        if budget_batch.ndim > 1: budget_batch = budget_batch.squeeze(-1)
                    else:
                        budget_batch = np.zeros(len(active_idx))
                    try:
                        bc_actions_batch = wrapped_bc.act(obs_batch, budget_batch, deterministic=True)
                        if isinstance(bc_actions_batch, (tuple, list)): bc_actions_batch = bc_actions_batch[0]
                        bc_actions_batch = np.asarray(bc_actions_batch)
                        if bc_actions_batch.ndim == 0: bc_actions_batch = np.array([bc_actions_batch])
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                        for i, a_idx in enumerate(active_idx):
                            bc_actions[a_idx] = int(bc_actions_batch[i])
                    except Exception:
                        bc_actions = np.zeros(max_n_agents, dtype=int)
                
                # compute预算限制 (使用新的成本计算)
                year_budget_limit = 0.0
                for a in active_idx:
                    act = int(bc_actions[a])
                    year_budget_limit += current_action_costs.get(act, 0.0)

                # 放宽系数，并应用预算因子
                year_budget_limit *= 1.2 * float(budget_factor)
                
                # ========== Step 2: Health mapping with budget ==========
                heuristic_actions = np.zeros(max_n_agents, dtype=int)
                remaining_budget = year_budget_limit
                cost_action_1 = current_action_costs[1]
                cost_action_2 = current_action_costs[2]

                # first处理 critical/poor
                for a in active_idx:
                    cur_h = int(health[ep, t, a])
                    h_cat = project_health_to_categories(cur_h)
                    if h_cat <= 1:
                        if remaining_budget >= cost_action_1:
                            heuristic_actions[a] = 1
                            remaining_budget -= cost_action_1

                # then处理 fair
                for a in active_idx:
                    cur_h = int(health[ep, t, a])
                    h_cat = project_health_to_categories(cur_h)
                    if h_cat == 2:
                        if remaining_budget >= cost_action_2:
                            heuristic_actions[a] = 2
                            remaining_budget -= cost_action_2

                final_actions = heuristic_actions
                
                # ========== Step 3: 指标计算 ==========
                step_cost = 0.0
                step_health_gain = 0.0
                step_health_gain_vs_nothing = 0.0
                step_health_gain_vs_history = 0.0
                
                for a in active_idx:
                    final_action_counts[int(final_actions[a])] += 1
                
                for a in active_idx:
                    act = int(final_actions[a])
                    cost = current_action_costs.get(act, 0.0)
                    step_cost += cost
                    
                    if health is not None and t + 1 < health.shape[1]:
                        cur_h = int(health[ep, t, a])
                        
                        gain = calculate_expected_health_improvement(cur_h, act, transition_matrices)
                        gain_nothing = calculate_expected_health_improvement(cur_h, 0, transition_matrices)
                        
                        gain_history = 0.0
                        if actions_history is not None and t < actions_history.shape[1]:
                            hist_act = int(actions_history[ep, t, a])
                            gain_history = calculate_expected_health_improvement(cur_h, hist_act, transition_matrices)
                        
                        step_health_gain += gain
                        step_health_gain_vs_nothing += (gain - gain_nothing)
                        step_health_gain_vs_history += (gain - gain_history)
                        
                        if gain_nothing != 0:
                            bridge_metrics['gains_vs_nothing_pct'].append((gain - gain_nothing) / abs(gain_nothing) * 100)
                        else:
                            bridge_metrics['gains_vs_nothing_pct'].append(0.0)
                        
                        if gain_history != 0:
                            bridge_metrics['gains_vs_history_pct'].append((gain - gain_history) / abs(gain_history) * 100)
                        else:
                            bridge_metrics['gains_vs_history_pct'].append(0.0)
                        
                        bridge_metrics['costs'].append(cost)
                        bridge_metrics['gains'].append(gain)
                
                ep_total_cost += step_cost
                ep_total_budget += year_budget_limit
                ep_total_health_gain += step_health_gain
                ep_total_health_gain_vs_nothing += step_health_gain_vs_nothing
                ep_total_health_gain_vs_history += step_health_gain_vs_history
                ep_steps += 1
                
                total_metrics['total_cost'] += step_cost
                total_metrics['total_budget'] += year_budget_limit
                total_metrics['total_health_gain'] += step_health_gain
                total_metrics['total_health_gain_vs_nothing'] += step_health_gain_vs_nothing
                total_metrics['total_health_gain_vs_history'] += step_health_gain_vs_history
                total_metrics['steps'] += 1
                
                if step_cost > year_budget_limit + 1e-5:
                    total_metrics['budget_violations'] += 1
            
            episode_metrics.append({
                'episode': ep,
                'total_cost': float(ep_total_cost),
                'total_budget': float(ep_total_budget),
                'total_health_gain': float(ep_total_health_gain),
                'total_health_gain_vs_nothing': float(ep_total_health_gain_vs_nothing),
                'total_health_gain_vs_history': float(ep_total_health_gain_vs_history),
                'steps': ep_steps
            })
    
    n_steps = total_metrics['steps']
    if n_steps == 0: return None
    
    total_final_actions = sum(final_action_counts.values())
    final_action_distribution = {
        action: (count / total_final_actions * 100) if total_final_actions > 0 else 0.0
        for action, count in final_action_counts.items()
    }
    
    bridge_avg_health_gain_vs_nothing_pct = float(np.mean(bridge_metrics['gains_vs_nothing_pct'])) if bridge_metrics['gains_vs_nothing_pct'] else 0.0
    bridge_avg_health_gain_vs_history_pct = float(np.mean(bridge_metrics['gains_vs_history_pct'])) if bridge_metrics['gains_vs_history_pct'] else 0.0
    
    results = {
        'model_name': 'heuristic_pragmatic_v2',
        'algorithm_type': 'heuristic',
        'num_episodes': num_eps,
        'num_steps': n_steps,
        'mean_step_cost': float(total_metrics['total_cost'] / n_steps),
        'mean_step_budget': float(total_metrics['total_budget'] / n_steps),
        'budget_utilization': float(total_metrics['total_cost'] / total_metrics['total_budget']) if total_metrics['total_budget'] > 0 else 0.0,
        'total_health_gain': float(total_metrics['total_health_gain']),
        'mean_health_gain_per_step': float(total_metrics['total_health_gain'] / n_steps),
        'total_health_gain_vs_nothing': float(total_metrics['total_health_gain_vs_nothing']),
        'total_health_gain_vs_history': float(total_metrics['total_health_gain_vs_history']),
        'violations_count': int(total_metrics['budget_violations']),
        'violation_rate': float(total_metrics['budget_violations'] / n_steps) if n_steps > 0 else 0.0,
        'efficiency_gain_per_1k': float((total_metrics['total_health_gain'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'efficiency_gain_per_1k_vs_history': float((total_metrics['total_health_gain_vs_history'] * 1000) / total_metrics['total_cost']) if total_metrics['total_cost'] > 0 else 0.0,
        'bridge_avg_cost': float(np.mean(bridge_metrics['costs'])) if bridge_metrics['costs'] else 0.0,
        'bridge_avg_health_gain': float(np.mean(bridge_metrics['gains'])) if bridge_metrics['gains'] else 0.0,
        'bridge_avg_health_gain_vs_nothing_pct': bridge_avg_health_gain_vs_nothing_pct,
        'bridge_avg_health_gain_vs_history_pct': bridge_avg_health_gain_vs_history_pct,
        'final_action_distribution': final_action_distribution,
        'episode_metrics': episode_metrics
    }
    
    print(f"评估完成:")
    print(f"  平均成本: ${results['mean_step_cost']:.2f}")
    print(f"  平均预算: ${results['mean_step_budget']:.2f}")
    print(f"  效率 (gain/$1k): {results['efficiency_gain_per_1k']:.4f}")
    print(f"  总健康改善 vs History: {results['total_health_gain_vs_history']:.4f}")
    print(f"\n动作分布:")
    print(f"  总动作数: {total_final_actions}")
    for action in sorted(final_action_distribution.keys()):
        count = final_action_counts[action]
        percentage = final_action_distribution[action]
        print(f"  Action {action}: {count} ({percentage:.2f}%)")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用BC算法生成预算约束评估模型')
    
    # 必需参数
    parser.add_argument('--test_buffer', type=str, default='marl/data_benchmark/episodes/test_buffer.pt', help='测试数据buffer路径')
    
    # optional参数
    parser.add_argument('--config', type=str, default='paper/benchmark/flows/config.yaml', help='config file path')
    parser.add_argument('--device_id', type=int, default=3, help='GPU设备ID')
    parser.add_argument('--output_dir', type=str, default='paper/benchmark/analysis/100years/results', help='output directory')
    parser.add_argument('--bc_algorithm', type=str, default=BC_ALGORITHM_NAME, help='BC算法名称')
    parser.add_argument('--target_algorithms', type=str, nargs='+', default=None, help='要评估的算法列表（覆盖默认列表）')
    parser.add_argument('--budget_factor', type=float, default=1.0,
                        help='预算因子：同时作用于全局预算计算和观测中的预算特征（log_budgets）')
    parser.add_argument('--n_years', type=int, default=100,
                        help='长期模拟的总年份（后续年份的健康状态和观测由状态转移矩阵+归一化规则生成）')
    parser.add_argument('--max_episodes', type=int, default=20,
                        help='只使用前若干个episode进行评估，例如20表示只评估前20个episode')
    
    args = parser.parse_args()
    
    # 设置
    device = th.device(f"cuda:{args.device_id}" if th.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"使用设备: {device}")
    
    # 1. 加载配置和环境信息
    print("\n[1/6] 加载配置和环境信息...")
    config = load_config(args.config)
    env_info = load_env_info(config['data']['env_info_file'])
    transition_matrices = build_transition_matrices()
    action_costs = ACTION_COSTS
    
    # 2. 查找并加载BC模型
    print(f"\n[2/6] 查找并加载BC模型 ({args.bc_algorithm})...")
    bc_model_path = find_latest_model(MODEL_DIR, args.bc_algorithm)
    if not bc_model_path:
        print(f"错误: 无法找到BC模型 ({args.bc_algorithm})")
        sys.exit(1)
    
    bc_model = load_model_safe(bc_model_path, device)
    if bc_model is None:
        print(f"错误: 无法加载BC模型")
        sys.exit(1)
    print(f"BC模型加载成功: {bc_model_path}")
    
    # 3. 加载测试数据
    print(f"\n[3/6] 加载测试数据: {args.test_buffer}")
    test_buffer = load_buffer(args.test_buffer, device)
    # ensurebuffer在正确设备上
    if hasattr(test_buffer, 'to'):
        test_buffer.to(device)
    if test_buffer is not None:
        data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents = extract_data_from_buffer(test_buffer)
    else:
        print("错误: 无法加载测试数据")
        sys.exit(1)
    #data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents = extract_data_from_buffer(test_buffer)
    
    # extract_data_from_buffer返回的是numpy数组，不需要.cpu().numpy()
    test_data_dict = {
        'data': np.asarray(data),                    # [num_eps, T, n_agents, state_dim]
        'actions': np.asarray(actions),              # [num_eps, T, n_agents]
        'episode_budgets': np.asarray(episode_budgets) if episode_budgets is not None else None,  # [num_eps] 或标量
        'health': np.asarray(health) if health is not None else None,  # [num_eps, T+1, n_agents]
        'raw_cost': np.asarray(raw_cost) if raw_cost is not None else None,  # [num_eps, T, n_agents]
        'rewards': np.asarray(rewards) if rewards is not None else None,  # [num_eps, T, n_agents]
        'log_budgets': np.asarray(log_budgets) if log_budgets is not None else None,  # [num_eps, T, n_agents]
        'actual_n_agents': np.asarray(actual_n_agents)  # [num_eps]
    }

    # if指定了max_episodes，只保留前max_episodes个episode
    if args.max_episodes is not None:
        max_eps = int(args.max_episodes)
        print(f"\n仅使用前 {max_eps} 个episode进行评估")
        for key in ['data', 'actions', 'episode_budgets', 'health', 'raw_cost', 'rewards', 'log_budgets', 'actual_n_agents']:
            arr = test_data_dict.get(key, None)
            if arr is None:
                continue
            # scalar或一维标量数组不需要截断
            if not hasattr(arr, 'shape') or arr.shape[0] <= max_eps:
                continue
            test_data_dict[key] = arr[:max_eps]

    # === 预算因子：对log_budgets和episode_budgets做反归一化缩放再写回 ===
    if args.budget_factor != 1.0 and test_data_dict['log_budgets'] is not None:
        print(f"\n应用预算因子 budget_factor = {args.budget_factor} 到log_budgets和episode_budgets...")
        log_budgets_arr = test_data_dict['log_budgets']

        # shape兼容处理
        original_shape = log_budgets_arr.shape
        squeezed = False
        if log_budgets_arr.ndim == 4 and log_budgets_arr.shape[-1] == 1:
            log_budgets_arr = log_budgets_arr.squeeze(-1)  # [num_eps, T, n_agents]
            squeezed = True

        # normalize参数
        norm_params = env_info.get('normalization_params', {})
        if 'log_budgets' in norm_params:
            mean = norm_params['log_budgets']['mean']
            std = norm_params['log_budgets']['std']
        else:
            print("警告: env_info中未找到log_budgets归一化参数，使用默认mean=0,std=1")
            mean, std = 0.0, 1.0

        # 反归一化 -> 线性空间预算 -> 乘以预算因子 -> 重新log & 归一化
        raw_log_budgets = log_budgets_arr * std + mean
        raw_budgets = np.expm1(raw_log_budgets)          # 回到原始预算尺度
        scaled_budgets = raw_budgets * float(args.budget_factor)
        scaled_log_budgets = np.log1p(scaled_budgets)
        new_log_budgets = (scaled_log_budgets - mean) / (std + 1e-8)

        # restore形状
        if squeezed:
            new_log_budgets = new_log_budgets[..., np.newaxis]

        test_data_dict['log_budgets'] = new_log_budgets.astype(np.float32)

        # 同步缩放episode级别预算（如果存在）
        if test_data_dict['episode_budgets'] is not None:
            test_data_dict['episode_budgets'] = (
                np.asarray(test_data_dict['episode_budgets'], dtype=np.float32) *
                float(args.budget_factor)
            )

        print("预算因子应用完成。")
    
    print(f"测试数据形状: {test_data_dict['data'].shape}")
    print(f"Episode数量: {len(test_data_dict['data'])}")
    if test_data_dict['actions'] is not None:
        print(f"动作数据形状: {test_data_dict['actions'].shape}")
    if test_data_dict['log_budgets'] is not None:
        print(f"Log预算数据形状: {test_data_dict['log_budgets'].shape}")
    
    all_results = []

    if "heuristic" in TARGET_ALGORITHMS:
        print(f"\n评估启发式算法...")
        heuristic_result = evaluate_heuristic_algorithm(
            bc_model=bc_model,
            test_data_dict=test_data_dict,
            device=device,
            transition_matrices=transition_matrices,
            action_costs=action_costs,
            budget_factor=args.budget_factor
        )
        if heuristic_result:
            heuristic_result['algorithm_name'] = 'heuristic_health_based'
            all_results.append(heuristic_result)

    else:
        # 4. 初始化启发式预算控制器
        print(f"\n[4/6] 初始化启发式预算控制器...")
        controller = HeuristicBudgetController(action_costs)
        
        # 5. 查找目标模型
        print(f"\n[5/6] 查找目标模型...")
        target_algorithms = args.target_algorithms if args.target_algorithms else TARGET_ALGORITHMS
        target_models_info = []
        
        for algo_name in target_algorithms:
            model_path = find_latest_model(MODEL_DIR, algo_name)
            if model_path:
                target_models_info.append({
                    'algorithm_name': algo_name,
                    'model_path': model_path
                })
                print(f"  找到: {algo_name} -> {model_path}")
        
        if not target_models_info:
            print("错误: 未找到任何目标模型")
            sys.exit(1)
        
        # 6. 评估循环
        print(f"\n[6/6] 开始评估...")
    
        
        for model_info in target_models_info:
            algo_name = model_info['algorithm_name']
            model_path = model_info['model_path']
            
            print(f"\n评估算法: {algo_name}")
            
            result = evaluate_single_model(
                model_path=model_path,
                bc_model=bc_model,
                test_data_dict=test_data_dict,
                device=device,
                transition_matrices=transition_matrices,
                controller=controller,
                action_costs=action_costs,
                budget_factor=args.budget_factor,
                n_years=args.n_years,
                env_info=env_info
            )
            
            if result:
                result['algorithm_name'] = algo_name
                all_results.append(result)
        # 7. 保存结果（按算法名称依次保存，每个算法一个文件）
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_files = []
        for r in all_results:
            algo_name = r['algorithm_name']
            path = os.path.join(args.output_dir, f"{algo_name}_budget_constrained_eval_{dt_str}.json")
            output_files.append(path)
    
    print(f"\n保存结果...")
    # 转换为可序列化格式
    def convert_to_serializable(obj):
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
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    # by算法分别保存：启发式单文件；多算法时每算法一文件
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # according to预算因子自动添加前缀（或中缀）用于区分不同实验
    if abs(args.budget_factor - 1.0) > 1e-8:
        bf_tag = f"bf{args.budget_factor}".replace(".", "_")
        bf_suffix = f"_{bf_tag}"
    else:
        bf_suffix = ""

    if "heuristic" in TARGET_ALGORITHMS:
        serializable_results = convert_to_serializable(all_results)
        output_file = os.path.join(
            args.output_dir,
            f"heuristic_health_based_budget_constrained_eval{bf_suffix}_{dt_str}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        output_files = [output_file]
    else:
        # 重新构建带有预算因子标记的输出文件名
        new_output_files = []
        for r in all_results:
            algo_name = r['algorithm_name']
            path = os.path.join(
                args.output_dir,
                f"{algo_name}_budget_constrained_eval{bf_suffix}_{dt_str}.json"
            )
            serializable_results = convert_to_serializable([r])
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            new_output_files.append(path)
        output_files = new_output_files
    
    print(f"\n评估完成！结果已保存到以下文件：")
    for p in output_files:
        print(f"  - {p}")
    print(f"共评估了 {len(all_results)} 个模型，保存了 {len(output_files)} 个文件")


if __name__ == "__main__":
    main()