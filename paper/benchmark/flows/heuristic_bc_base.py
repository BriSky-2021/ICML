import numpy as np
import torch


def project_health_to_categories(health_value):
    """
    Map raw health values to 4 categories.
    0-2 -> 0 (critical)
    3-4 -> 1 (poor)
    5-6 -> 2 (fair)
    7+  -> 3 (good)
    """
    if health_value <= 2:
        return 0
    if health_value <= 4:
        return 1
    if health_value <= 6:
        return 2
    return 3


def denorm_log_budgets(log_budgets, log_budget_norm_params):
    if log_budget_norm_params is None:
        return np.expm1(log_budgets)
    mean = log_budget_norm_params["mean"]
    std = log_budget_norm_params["std"]
    log_raw = log_budgets * std + mean
    return np.expm1(log_raw)


def generate_heuristic_actions(
    data,
    health,
    log_budgets,
    actual_n_agents,
    action_costs,
    log_budget_norm_params=None,
    budget_scale=1.2,
    seed=0,
    verbose=True,
):
    if health is None:
        raise ValueError("health is required to generate heuristic actions")

    data = np.asarray(data)
    health = np.asarray(health)

    num_eps, T, max_n_agents, _ = data.shape
    actions_h = np.zeros((num_eps, T, max_n_agents), dtype=int)

    for ep in range(num_eps):
        active_n = int(actual_n_agents[ep]) if actual_n_agents is not None else max_n_agents
        active_idx = list(range(active_n))

        for t in range(T):
            for a in active_idx:
                raw_h = int(health[ep, t, a])
                h_cat = project_health_to_categories(raw_h)
                if h_cat <= 1:
                    actions_h[ep, t, a] = 1
                elif h_cat == 2:
                    actions_h[ep, t, a] = 2
                else:
                    actions_h[ep, t, a] = 0

    if verbose:
        total_counts = np.zeros(len(action_costs), dtype=np.int64)
        total_active = 0
        for ep in range(num_eps):
            active_n = int(actual_n_agents[ep]) if actual_n_agents is not None else max_n_agents
            active_idx = list(range(active_n))
            acts = actions_h[ep, :, active_idx].reshape(-1)
            total_active += acts.size
            for a in range(len(action_costs)):
                total_counts[a] += int((acts == a).sum())

        print("[Heuristic] action counts:", {i: int(c) for i, c in enumerate(total_counts)})
        if total_active > 0:
            ratios = {i: float(c) / float(total_active) for i, c in enumerate(total_counts)}
            print("[Heuristic] action ratios:", ratios)

    return actions_h


def append_heuristic_actions_to_dataset(dataset, actions_heuristic, data):
    actions_heuristic = np.asarray(actions_heuristic)
    num_eps, T, n_agents, _ = data.shape
    action_list = []

    for ep in range(num_eps):
        for t in range(T):
            for agent in range(n_agents):
                action_list.append(int(actions_heuristic[ep, t, agent]))

    dataset["actions_heuristic"] = np.asarray(action_list)
    return dataset


def compute_cql_heuristic_loss(q_all, actions_heuristic):
    logsumexp_q = torch.logsumexp(q_all, dim=1)
    q_heur = q_all.gather(1, actions_heuristic.unsqueeze(1)).squeeze(1)
    return (logsumexp_q - q_heur).mean()
