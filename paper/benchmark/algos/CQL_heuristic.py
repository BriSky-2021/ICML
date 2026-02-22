import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

def compute_cql_heuristic_loss(q_all, actions_heuristic):
    """
    Use cross-entropy loss to directly encourage selecting heuristic actions.
    More direct than CQL-style loss; maximizes softmax(Q)[a_heur].
    """
    log_probs = F.log_softmax(q_all, dim=1)
    loss = F.nll_loss(log_probs, actions_heuristic, reduction='mean')
    return loss

class MLP(nn.Module):
    """
    Generic MLP for Q-network.
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CQLHeuristicDiscrete(nn.Module):
    """
    CQL with heuristic imitation loss for discrete actions.

    Algorithm:
    1. Standard DQN-style Bellman update (MSE loss).
    2. CQL Regularization: logsumexp(Q) - Q(s, a_data)
    3. Heuristic loss: logsumexp(Q) - Q(s, a_heuristic)
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list = [128, 128],
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_weight: float = 0.2,
        heuristic_weight: float = 1.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.cql_weight = cql_weight
        self.heuristic_weight = heuristic_weight
        self.device = device

        # Q-network
        self.q_net = MLP(state_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLP(state_dim, n_actions, hidden_sizes).to(device)

        # Hard copy
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def act(self, state, budget=None, deterministic=True, training=False):
        """
        Policy is argmax Q(s,a).
        Handles budget concatenation similar to OneStepRL logic.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Budget handling logic (Compatible with OneStepRL)
        if budget is not None:
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            
            current_state_dim = state.shape[-1]
            if current_state_dim < self.state_dim:
                if budget.dim() == 0:
                    budget = budget.unsqueeze(0).unsqueeze(-1)
                elif budget.dim() == 1:
                    budget = budget.unsqueeze(-1)
                elif budget.dim() == 2:
                    if budget.shape[1] != 1:
                        budget = budget[:, 0:1]
                
                if state.shape[0] != budget.shape[0]:
                    if budget.shape[0] == 1:
                        budget = budget.expand(state.shape[0], -1)
                    elif state.shape[0] == 1:
                        state = state.expand(budget.shape[0], -1)
                
                state = torch.cat([state, budget], dim=-1)
        
        with torch.no_grad():
            q_vals = self.q_net(state)
        
        if deterministic:
            actions = torch.argmax(q_vals, dim=-1)
        else:
            probs = F.softmax(q_vals, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
        return actions.cpu().numpy() if actions.shape[0] > 1 else actions.item()

    def get_action_prob(self, state, budget=None):
        """
        Returns action probabilities based on Softmax(Q).
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Budget handling logic
        if budget is not None:
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            
            current_state_dim = state.shape[-1]
            if current_state_dim < self.state_dim:
                if budget.dim() == 0:
                    budget = budget.unsqueeze(0).unsqueeze(-1)
                elif budget.dim() == 1:
                    budget = budget.unsqueeze(-1)
                elif budget.dim() == 2:
                    if budget.shape[1] != 1:
                        budget = budget[:, 0:1]
                
                if state.shape[0] != budget.shape[0]:
                    if budget.shape[0] == 1:
                        budget = budget.expand(state.shape[0], -1)
                    elif state.shape[0] == 1:
                        state = state.expand(budget.shape[0], -1)
                
                state = torch.cat([state, budget], dim=-1)
        
        q_vals = self.q_net(state)
        probs = F.softmax(q_vals, dim=-1)
        return probs

    def compute_loss(self, batch):
        """
        Computes CQL Loss + Bellman Loss.
        batch: dict with keys 'obs', 'actions', 'actions_heuristic',
               'rewards', 'next_obs', 'dones', and optionally 'budget', 'next_budget'
        """
        obs = to_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        acts = to_tensor(batch['actions'], dtype=torch.long, device=self.device)
        if 'actions_heuristic' not in batch:
            raise ValueError("actions_heuristic is required for heuristic CQL")
        acts_heur = to_tensor(batch['actions_heuristic'], dtype=torch.long, device=self.device)
        rews = to_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_obs = to_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones = to_tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # Handle budget if present in batch but not in obs (similar to CPQ logic)
        if 'budget' in batch and obs.shape[-1] < self.state_dim:
            bud = to_tensor(batch['budget'], dtype=torch.float32, device=self.device)
            if bud.ndim == 1: bud = bud.unsqueeze(-1)
            obs = torch.cat([obs, bud], dim=-1)
            
        if 'next_budget' in batch and next_obs.shape[-1] < self.state_dim:
            next_bud = to_tensor(batch['next_budget'], dtype=torch.float32, device=self.device)
            if next_bud.ndim == 1: next_bud = next_bud.unsqueeze(-1)
            next_obs = torch.cat([next_obs, next_bud], dim=-1)

        # --- 1. Bellman Loss (DQN) ---
        # Q(s, :)
        q_all = self.q_net(obs)
        # Q(s, a)
        q_pred = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Target: r + gamma * max_a' Q_target(s', a')
            q_next_all = self.q_net_target(next_obs)
            q_next_max, _ = q_next_all.max(dim=1)
            target_q = rews + self.gamma * (1 - dones) * q_next_max

        loss_bellman = F.mse_loss(q_pred, target_q)

        # --- 2. CQL Loss ---
        # Loss = alpha * (log(sum(exp(Q(s,a)))) - Q(s, a_data))
        # log_sum_exp over all actions
        logsumexp_q = torch.logsumexp(q_all, dim=1)
        # Q value of the data action
        q_data = q_pred 
        
        loss_cql = (logsumexp_q - q_data).mean()

        if acts_heur.ndim > 1:
            acts_heur = acts_heur.squeeze(-1)
        loss_heur = compute_cql_heuristic_loss(q_all, acts_heur)

        total_loss = loss_bellman + self.cql_weight * loss_cql + self.heuristic_weight * loss_heur

        return total_loss, loss_bellman, loss_cql, loss_heur

    def update(self, batch, optimizer):
        """
        Updates Q network.
        """
        total_loss, loss_bellman, loss_cql, loss_heur = self.compute_loss(batch)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        optimizer.step()

        # Target soft update
        self._soft_update(self.q_net_target, self.q_net, self.tau)

        return (
            total_loss.item(),
            loss_bellman.item(),
            loss_cql.item(),
            loss_heur.item(),
        )

    def get_q_values(self, state, budget=None):
        """
        Get Q-values for the given state (and optional budget).
        
        Args:
            state: [batch, state_dim] or [batch, n_agents, state_dim] or [state_dim]
            budget: scalar, [batch], [batch, 1], or None. 
                   If None, assumes state already includes budget or uses state as-is.
        
        Returns:
            q_values: [batch, n_actions] or [batch, n_agents, n_actions] or [n_actions]
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Handle budget if provided separately
            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
                
                # Normalize state dimensions
                original_shape = state.shape
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                elif state.dim() == 3:
                    # [batch, n_agents, state_dim] -> flatten to [batch * n_agents, state_dim]
                    batch_size, n_agents, state_dim = state.shape
                    state = state.view(batch_size * n_agents, state_dim)
                    need_reshape = True
                else:
                    need_reshape = False
                
                # Normalize budget dimensions
                current_state_dim = state.shape[-1]
                if current_state_dim < self.state_dim:
                    if budget.dim() == 0:
                        budget = budget.unsqueeze(0).unsqueeze(-1)
                    elif budget.dim() == 1:
                        budget = budget.unsqueeze(-1)
                    elif budget.dim() == 2:
                        if budget.shape[1] != 1:
                            budget = budget[:, 0:1]
                    
                    # Handle broadcasting
                    if state.shape[0] != budget.shape[0]:
                        if budget.shape[0] == 1:
                            budget = budget.expand(state.shape[0], -1)
                        elif state.shape[0] == 1:
                            state = state.expand(budget.shape[0], -1)
                    
                    state = torch.cat([state, budget], dim=-1)
                
                # Reshape back if needed
                if 'need_reshape' in locals() and need_reshape:
                    state = state.view(batch_size, n_agents, -1)
            else:
                # Assume state already contains budget or budget is not needed
                if state.dim() == 1:
                    state = state.unsqueeze(0)
            
            # Forward pass
            q_vals = self.q_net(state)
            
            # Reshape output if input was 3D
            if 'original_shape' in locals() and len(original_shape) == 3:
                q_vals = q_vals.view(original_shape[0], original_shape[1], -1)
            elif 'original_shape' in locals() and len(original_shape) == 1:
                q_vals = q_vals.squeeze(0)
            
            return q_vals