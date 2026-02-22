import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

class MLPQCriticDiscrete(nn.Module):
    """
    Q-network for discrete action space.
    Input: state + budget. Output: Q-values for all actions.
    """
    def __init__(self, input_dim, n_actions, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CPQHeuristicDiscrete(nn.Module):
    """
    CPQ with Heuristic Imitation Loss for Discrete Action Spaces.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        budget_dim: int = 1,
        hidden_sizes: list = [128, 128],
        gamma: float = 0.99,
        tau: float = 0.005,
        cost_limit: float = 10,
        episode_len: int = 300,
        heuristic_weight: float = 0.3,
        device: str = "cpu"
    ):
        super().__init__()
        self.input_dim = state_dim + budget_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.heuristic_weight = heuristic_weight
        self.device = device
        
        self.update_cnt = 0

        # Q network (Reward)
        self.q_net = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        
        # Qc network (Cost)
        self.qc_net = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        self.qc_net_target = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)

        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.qc_net_target.load_state_dict(self.qc_net.state_dict())

        self.qc_thres = cost_limit * (1 - gamma**episode_len) / (1 - gamma) / episode_len

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def act(self, state, budget, deterministic=True, training=False):
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            
            if state.dim() == 1: state = state.unsqueeze(0)
            if budget.dim() == 0: budget = budget.unsqueeze(0)
            if budget.dim() == 1: budget = budget.unsqueeze(1)
            
            x = torch.cat([state, budget], dim=-1)
            q_vals = self.q_net(x)
            
            if deterministic:
                qc_vals = self.qc_net(x)
                mask = (qc_vals <= self.qc_thres)
                
                if mask.sum() == 0:
                    mask = torch.ones_like(mask, dtype=torch.bool)
                
                q_vals_masked = q_vals.clone()
                q_vals_masked[~mask] = -float('inf')
                
                actions = torch.argmax(q_vals_masked, dim=-1)
            else:
                probs = F.softmax(q_vals, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
            return actions.cpu().numpy() if actions.shape[0] > 1 else actions.item()

    def compute_loss(self, batch):
        obs = to_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        bud = to_tensor(batch['budget'], dtype=torch.float32, device=self.device)
        acts = to_tensor(batch['actions'], dtype=torch.long, device=self.device)
        rews = to_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_obs = to_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        next_bud = to_tensor(batch['next_budget'], dtype=torch.float32, device=self.device)
        costs = to_tensor(batch['costs'], dtype=torch.float32, device=self.device)
        dones = to_tensor(batch['dones'], dtype=torch.float32, device=self.device)

        if 'actions_heuristic' not in batch:
            raise ValueError("actions_heuristic is required for heuristic CPQ")
        acts_heur = to_tensor(batch['actions_heuristic'], dtype=torch.long, device=self.device)

        if bud.ndim == 1: bud = bud.unsqueeze(-1)
        if next_bud.ndim == 1: next_bud = next_bud.unsqueeze(-1)

        # 1. Compute current Q(s, a) and Qc(s, a)
        x = torch.cat([obs, bud], dim=-1)
        q_all = self.q_net(x)
        q_pred = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)
        qc_pred = self.qc_net(x).gather(1, acts.unsqueeze(1)).squeeze(1)

        # 2. Compute target (Double DQN)
        with torch.no_grad():
            x_next = torch.cat([next_obs, next_bud], dim=-1)
            
            q_next_online = self.q_net(x_next)
            qc_next_online = self.qc_net(x_next)
            
            mask = (qc_next_online <= self.qc_thres)
            all_unsafe = (mask.sum(dim=1) == 0)
            mask[all_unsafe] = True 
            
            q_next_online_masked = q_next_online.clone()
            q_next_online_masked[~mask] = -float('inf') 
            next_actions = torch.argmax(q_next_online_masked, dim=1, keepdim=True)

            q_next_target_val = self.q_net_target(x_next).gather(1, next_actions).squeeze(1)
            qc_next_target_val = self.qc_net_target(x_next).gather(1, next_actions).squeeze(1)

            target_q = rews + self.gamma * (1 - dones) * q_next_target_val
            target_qc = costs + self.gamma * (1 - dones) * qc_next_target_val

        # 3. Compute MSE loss
        loss_q = F.mse_loss(q_pred, target_q)
        loss_qc = F.mse_loss(qc_pred, target_qc)

        # 4. Heuristic Imitation Loss (on Q network)
        if acts_heur.ndim > 1:
            acts_heur = acts_heur.squeeze(-1)
        log_probs = F.log_softmax(q_all, dim=1)
        loss_heur = F.nll_loss(log_probs, acts_heur, reduction='mean')

        return loss_q, loss_qc, loss_heur

    def update(self, batch, optimizer_q, optimizer_qc):
        self.update_cnt += 1
        loss_q, loss_qc, loss_heur = self.compute_loss(batch)
        
        # Update Q network (with heuristic loss)
        optimizer_q.zero_grad()
        total_q_loss = loss_q + self.heuristic_weight * loss_heur
        total_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        optimizer_q.step()
        
        # Update Qc network
        optimizer_qc.zero_grad()
        loss_qc.backward()
        torch.nn.utils.clip_grad_norm_(self.qc_net.parameters(), max_norm=10.0)
        optimizer_qc.step()
        
        # Soft update target networks
        self._soft_update(self.q_net_target, self.q_net, self.tau)
        self._soft_update(self.qc_net_target, self.qc_net, self.tau)
        
        return loss_q.item(), loss_qc.item(), loss_heur.item()

    def get_action_prob(self, state, budget):
        """
        Get action probability distribution (softmax over Q-values).
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(budget):
            budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if budget.dim() == 0:
            budget = budget.unsqueeze(0)
        if budget.dim() == 1:
            budget = budget.unsqueeze(1)
        
        x = torch.cat([state, budget], dim=-1)
        q_vals = self.q_net(x)
        probs = F.softmax(q_vals, dim=-1)
        
        return probs

    def get_q_values(self, state, budget=None):
        """
        Get Q-values for the given state (and optional budget).
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
                
                if state.dim() == 2 and budget.dim() == 1:
                    budget = budget.unsqueeze(1)
                
                if state.dim() == 3:
                    if budget.dim() == 2:
                        budget = budget.unsqueeze(1).expand(-1, state.shape[1], -1)
                
                x = torch.cat([state, budget], dim=-1)
            else:
                x = state

            q_vals = self.q_net(x)
            
            return q_vals
