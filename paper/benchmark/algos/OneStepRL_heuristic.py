import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

class MLP(nn.Module):
    """
    Generic MLP for Q-network and Policy.
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

class OneStepRLHeuristicDiscrete(nn.Module):
    """
    One-Step RL with Heuristic-based BC for Discrete Action Spaces.
    
    Algorithm:
    1. Train a Behavior Cloning (BC) model directly on heuristic actions (not data actions).
    2. Train Q_beta(s,a) using Expected SARSA with pi_beta (learned from heuristic).
    3. Evaluation Policy: pi(s) = argmax_a Q_beta(s,a)
    
    Note: BC network learns heuristic policy directly, which helps stabilize Q-learning.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list = [128, 128],
        gamma: float = 0.99,
        tau: float = 0.005,
        heuristic_weight: float = 0.3,
        device: str = "cpu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.heuristic_weight = heuristic_weight
        self.device = device
        self.update_cnt = 0  # For debug output

        # Behavior Cloning Model (estimates pi_beta)
        self.bc_net = MLP(state_dim, n_actions, hidden_sizes).to(device)

        # Q-network (estimates Q^pi_beta)
        self.q_net = MLP(state_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLP(state_dim, n_actions, hidden_sizes).to(device)
        
        # Initialize Q network with smaller weights to prevent explosion
        for param in self.q_net.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)  # Smaller gain for stability
        for param in self.q_net_target.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)

        # Hard copy
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def act(self, state, budget=None, deterministic=True, training=False):
        """
        In One-Step RL, the policy is simply the argmax of the learned Q_beta.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
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

    def compute_loss(self, batch):
        """
        Computes losses for BC (using heuristic actions only) and Q-learning.
        batch: dict with keys 'obs', 'actions', 'actions_heuristic', 'rewards', 'next_obs', 'dones'
        """
        obs = to_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        acts = to_tensor(batch['actions'], dtype=torch.long, device=self.device)
        rews = to_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_obs = to_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones = to_tensor(batch['dones'], dtype=torch.float32, device=self.device)
        
        if 'actions_heuristic' not in batch:
            raise ValueError("actions_heuristic is required for heuristic OneStepRL")
        acts_heur = to_tensor(batch['actions_heuristic'], dtype=torch.long, device=self.device)

        # --- 1. BC Loss: Directly use heuristic actions (replacing original BC loss) ---
        bc_logits = self.bc_net(obs)
        if acts_heur.ndim > 1:
            acts_heur = acts_heur.squeeze(-1)
        # Use heuristic actions as the target for BC
        loss_bc = F.cross_entropy(bc_logits, acts_heur)

        # --- 2. Q-Learning Loss (Expected SARSA) ---
        # Note: Q-learning still uses original actions from data for TD target
        q_pred = self.q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        # Clip Q predictions to prevent explosion
        q_pred = torch.clamp(q_pred, min=-100.0, max=100.0)

        with torch.no_grad():
            bc_next_logits = self.bc_net(next_obs)
            bc_next_probs = F.softmax(bc_next_logits, dim=-1)
            q_next_all = self.q_net_target(next_obs)
            # Clip Q values before computing expected value
            q_next_all = torch.clamp(q_next_all, min=-100.0, max=100.0)
            q_next_expected = (bc_next_probs * q_next_all).sum(dim=1)
            target_q = rews + self.gamma * (1 - dones) * q_next_expected
            # Clip target to prevent explosion
            target_q = torch.clamp(target_q, min=-100.0, max=100.0)

        loss_q = F.mse_loss(q_pred, target_q)

        # Debug output every 1000 updates
        self.update_cnt += 1
        if self.update_cnt % 1000 == 0:
            print(f"\n[OneStepRL-Heuristic Debug @ step {self.update_cnt}]")
            print(f"  > Reward Mean: {rews.mean().item():.4f} | Reward Std: {rews.std().item():.4f}")
            print(f"  > Q Pred Mean: {q_pred.mean().item():.4f} | Q Pred Std: {q_pred.std().item():.4f}")
            print(f"  > Q Target Mean: {target_q.mean().item():.4f} | Q Target Std: {target_q.std().item():.4f}")
            print(f"  > Q Next Expected Mean: {q_next_expected.mean().item():.4f}")
            print(f"  > BC Probs Entropy: {(-bc_next_probs * torch.log(bc_next_probs + 1e-8)).sum(dim=1).mean().item():.4f}")

        # Return loss_bc (which is now heuristic loss) and loss_q
        # Keep loss_heur for logging compatibility, but it's the same as loss_bc now
        return loss_bc, loss_q, loss_bc

    def update(self, batch, optimizer_bc, optimizer_q):
        """
        Updates both BC and Q networks.
        BC network is trained directly on heuristic actions (no original BC loss).
        """
        loss_bc, loss_q, loss_heur = self.compute_loss(batch)

        # Update Behavior Cloning (directly on heuristic actions)
        optimizer_bc.zero_grad()
        loss_bc.backward()
        optimizer_bc.step()

        # Update Q-function
        optimizer_q.zero_grad()
        loss_q.backward()
        # More aggressive gradient clipping for Q network
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        optimizer_q.step()

        # Target soft update
        self._soft_update(self.q_net_target, self.q_net, self.tau)

        return loss_bc.item(), loss_q.item(), loss_heur.item()

    def get_action_prob(self, state, budget=None):
        """
        Get action probability distribution (using BC network pi_beta).
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
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
        
        bc_logits = self.bc_net(state)
        probs = F.softmax(bc_logits, dim=-1)
        
        return probs

    def get_q_values(self, state, budget=None):
        """
        Get Q-values (Q_beta) for the given state.
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)

            if budget is not None:
                if not torch.is_tensor(budget):
                    budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
                
                current_state_dim = state.shape[-1]
                if current_state_dim < self.state_dim:
                    if budget.dim() == 0:
                        budget = budget.unsqueeze(0).unsqueeze(-1)
                    elif budget.dim() == 1:
                        budget = budget.unsqueeze(-1)
                    
                    if state.shape[0] != budget.shape[0]:
                         if budget.shape[0] == 1:
                            budget = budget.expand(state.shape[0], -1)
                    
                    state = torch.cat([state, budget], dim=-1)

            q_vals = self.q_net(state)
            
            return q_vals
