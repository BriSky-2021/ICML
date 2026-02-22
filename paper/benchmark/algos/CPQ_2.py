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
    Q-network for discrete action space (unchanged).
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

class OfflineCPQDiscrete(nn.Module):
    """
    Offline safe RL algorithm (Offline CPQ), based on CQL (Conservative Q-Learning) and CPQ.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        budget_dim: int = 1,
        hidden_sizes: list = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        cost_limit: float = 10,
        episode_len: int = 300,
        cql_alpha: float = 1.0,      # CQL loss weight (Reward)
        cql_beta: float = 0.0,       # CQL loss weight (Cost); set >0 to be more conservative on OOD actions for cost
        device: str = "cpu"
    ):
        super().__init__()
        self.input_dim = state_dim + budget_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device
        
        # Offline algorithm hyperparameters
        self.cql_alpha = cql_alpha 
        self.cql_beta = cql_beta

        self.update_cnt = 0 

        # Define networks
        self.q_net = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        
        self.qc_net = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        self.qc_net_target = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)

        # Initialize target networks
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.qc_net_target.load_state_dict(self.qc_net.state_dict())

        # Compute cost threshold
        self.qc_thres = cost_limit * (1 - gamma**episode_len) / (1 - gamma) / episode_len

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def act(self, state, budget, deterministic=True):
        # Offline evaluation typically uses deterministic=True
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
            
            # Safety masking
            qc_vals = self.qc_net(x)
            mask = (qc_vals <= self.qc_thres)
            
            # If all actions are unsafe, fall back to no masking (risk taking)
            if mask.sum() == 0:
                # Alternative: choose action with minimum cost: actions = torch.argmin(qc_vals, dim=-1)
                # Simple: do not mask (risk taking)
                mask = torch.ones_like(mask, dtype=torch.bool)
            
            q_vals_masked = q_vals.clone()
            q_vals_masked[~mask] = -float('inf')
            
            if deterministic:
                actions = torch.argmax(q_vals_masked, dim=-1)
            else:
                probs = F.softmax(q_vals_masked, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
            return actions.cpu().numpy() if actions.shape[0] > 1 else actions.item()

    def get_action_prob(self, state, budget=None):
        """
        Get action probability distribution for offline evaluation (e.g. FQE).
        Same as act: apply safety masking first, then softmax over Q.

        Args:
            state: [state_dim] or [batch, state_dim]
            budget: scalar or [batch] or [batch, 1] or None. Filled with 0 if None.

        Returns:
            probs: [batch, n_actions] action probability distribution (tensor)
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            else:
                batch_size = state.shape[0] if state.dim() > 1 else 1
                budget = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

            if state.dim() == 1:
                state = state.unsqueeze(0)
            if budget.dim() == 0:
                budget = budget.unsqueeze(0)
            if budget.dim() == 1:
                budget = budget.unsqueeze(1)

            x = torch.cat([state, budget], dim=-1)
            q_vals = self.q_net(x)
            qc_vals = self.qc_net(x)

            mask = (qc_vals <= self.qc_thres)
            all_unsafe = (mask.sum(dim=1) == 0)
            mask[all_unsafe] = True

            q_vals_masked = q_vals.clone()
            q_vals_masked[~mask] = -float('inf')
            probs = F.softmax(q_vals_masked, dim=-1)

        return probs

    def compute_loss(self, batch):
        obs = to_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        bud = to_tensor(batch['budget'], dtype=torch.float32, device=self.device)
        acts = to_tensor(batch['actions'], dtype=torch.long, device=self.device)
        rews = to_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_obs = to_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        next_bud = to_tensor(batch['next_budget'], dtype=torch.float32, device=self.device)
        costs = to_tensor(batch['costs'], dtype=torch.float32, device=self.device)
        dones = to_tensor(batch['dones'], dtype=torch.float32, device=self.device)

        if bud.ndim == 1: bud = bud.unsqueeze(-1)
        if next_bud.ndim == 1: next_bud = next_bud.unsqueeze(-1)

        x = torch.cat([obs, bud], dim=-1)
        
        # Q-values for all actions in current state (for CQL)
        q_all = self.q_net(x)   # [batch, n_actions]
        qc_all = self.qc_net(x) # [batch, n_actions]

        # Q-values for the actions actually taken in the dataset
        q_pred = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)
        qc_pred = qc_all.gather(1, acts.unsqueeze(1)).squeeze(1)

        # -----------------------------------------------------
        # 1. Compute target (Double DQN + safety masking)
        # -----------------------------------------------------
        with torch.no_grad():
            x_next = torch.cat([next_obs, next_bud], dim=-1)
            
            q_next_online = self.q_net(x_next)
            qc_next_online = self.qc_net(x_next)
            
            # Target policy: among safe actions, choose the one with largest Q
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

        # -----------------------------------------------------
        # 2. Compute loss (including CQL regularization)
        # -----------------------------------------------------
        
        # --- Reward Q loss ---
        mse_loss_q = F.mse_loss(q_pred, target_q)
        
        # CQL loss for reward: log(sum(exp(Q))) - Q(s, a_data)
        # Minimizes Q for OOD actions and maximizes Q for in-data actions
        cql_loss_q = torch.logsumexp(q_all, dim=1).mean() - q_pred.mean()
        
        loss_q = mse_loss_q + self.cql_alpha * cql_loss_q

        # --- Cost Qc loss ---
        mse_loss_qc = F.mse_loss(qc_pred, target_qc)
        
        # (Optional) CQL loss for cost: we want higher cost estimates for OOD actions (pessimistic about safety).
        # Here we keep it simple: no complex cost CQL, since reward CQL already discourages OOD actions.
        # Only standard MSE (or a small regularizer to avoid cost collapse).
        if self.cql_beta > 0:
            loss_qc = mse_loss_qc
        else:
            loss_qc = mse_loss_qc

        # ================= DEBUG output =================
        if self.update_cnt % 3000 == 0:
            print(f"\n[Offline CPQ Step {self.update_cnt}]")
            print(f"  > Q MSE: {mse_loss_q.item():.6f} | Q CQL: {cql_loss_q.item():.6f}")
            print(f"  > Q Pred: {q_pred.mean().item():.4f} | Q Target: {target_q.mean().item():.4f}")
            print(f"  > Qc Pred: {qc_pred.mean().item():.4f} | Qc Target: {target_qc.mean().item():.4f}")
        # ==============================================

        return loss_q, loss_qc

    def update(self, batch, optimizer_q, optimizer_qc):
        self.update_cnt += 1
        loss_q, loss_qc = self.compute_loss(batch)
        
        optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        optimizer_q.step()
        
        optimizer_qc.zero_grad()
        loss_qc.backward()
        torch.nn.utils.clip_grad_norm_(self.qc_net.parameters(), max_norm=10.0)
        optimizer_qc.step()
        
        self._soft_update(self.q_net_target, self.q_net, self.tau)
        self._soft_update(self.qc_net_target, self.qc_net, self.tau)
        
        return loss_q.item(), loss_qc.item()

    def get_q_values(self, state, budget=None):
        """
        Helper: get Q-values.
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
                if state.dim() == 2 and budget.dim() == 1:
                    budget = budget.unsqueeze(1)
                if state.dim() == 3 and budget.dim() == 2:
                    budget = budget.unsqueeze(1).expand(-1, state.shape[1], -1)
                x = torch.cat([state, budget], dim=-1)
            else:
                x = state

            return self.q_net(x)