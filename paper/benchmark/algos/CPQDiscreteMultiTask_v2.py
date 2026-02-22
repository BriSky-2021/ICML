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

class CPQDiscreteMultiTask(nn.Module):
    """
    Multi-task discrete-action CPQ algorithm.
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
        
        # Counter for controlling print frequency
        self.update_cnt = 0 

        # Define networks
        # Q network (Reward)
        self.q_net = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLPQCriticDiscrete(self.input_dim, n_actions, hidden_sizes).to(device)
        
        # Qc network (Cost)
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

        if bud.ndim == 1: bud = bud.unsqueeze(-1)
        if next_bud.ndim == 1: next_bud = next_bud.unsqueeze(-1)

        # 1. Compute current Q(s, a) and Qc(s, a)
        x = torch.cat([obs, bud], dim=-1)
        q_pred = self.q_net(x).gather(1, acts.unsqueeze(1)).squeeze(1)
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

        # ================= DEBUG output =================
        # Print every 1000 updates to avoid flooding
        if self.update_cnt % 1000 == 0:
            print(f"\n[DEBUG Step {self.update_cnt}]")
            print(f"  > Q Loss: {loss_q.item():.6f} | Qc Loss: {loss_qc.item():.6f}")
            print(f"  > Reward Mean: {rews.mean().item():.4f} | Cost Mean: {costs.mean().item():.4f}")
            print(f"  > Q Pred Mean: {q_pred.mean().item():.4f} | Q Target Mean: {target_q.mean().item():.4f}")
            print(f"  > Qc Pred Mean: {qc_pred.mean().item():.4f} | Qc Target Mean: {target_qc.mean().item():.4f}")
            
            # Check for abnormal values
            if q_pred.mean().abs() > 1000:
                print("  !!! WARNING: Q Values are exploding (Check Reward Scale or LR) !!!")
        # =================================================

        return loss_q, loss_qc

    def update(self, batch, optimizer_q, optimizer_qc):
        self.update_cnt += 1
        loss_q, loss_qc = self.compute_loss(batch)
        
        # Update Q network
        optimizer_q.zero_grad()
        loss_q.backward()
        # Print gradient norm (for debug)
        if self.update_cnt % 1000 == 0:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            print(f"  > Q Grad Norm: {q_grad_norm.item():.4f}")
        else:
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
        
        return loss_q.item(), loss_qc.item()

    def get_action_prob(self, state, budget):
        """
        Get action probability distribution (softmax over Q-values).
        
        Args:
            state: [state_dim] or [batch, state_dim]
            budget: scalar or [batch, 1] or [batch]
        
        Returns:
            probs: [batch, action_dim] - action probability distribution
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
        q_vals = self.q_net(x)  # [batch, n_actions]
        
        # Use softmax to convert Q-values to probability distribution
        probs = F.softmax(q_vals, dim=-1)  # [batch, n_actions]
        
        return probs


    # Add this method inside the CPQDiscreteMultiTask class
    def get_q_values(self, state, budget=None):
        """
        Get Q-values for the given state (and optional budget).
        
        Args:
            state: [batch, state_dim] or [batch, n_agents, state_dim]
            budget: [batch, 1] or None. If None, assumes state includes budget
                   or uses a default/zero budget if applicable.
        
        Returns:
            q_values: [batch, n_actions] or [batch, n_agents, n_actions]
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Handle budget if provided separately
            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
                
                # Normalize dimensions
                if state.dim() == 2 and budget.dim() == 1:
                    budget = budget.unsqueeze(1)
                
                # If state is [B, N, D], expand budget to [B, N, 1]
                if state.dim() == 3:
                    if budget.dim() == 2: # [B, 1]
                        budget = budget.unsqueeze(1).expand(-1, state.shape[1], -1)
                
                x = torch.cat([state, budget], dim=-1)
            else:
                # Assume state already contains budget or budget is not needed/handled externally
                x = state

            # Forward pass
            q_vals = self.q_net(x)
            
            return q_vals