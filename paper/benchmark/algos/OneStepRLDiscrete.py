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

class OneStepRLDiscrete(nn.Module):
    """
    One-Step RL (Brandfonbrener et al., 2021) for Discrete Action Spaces.
    
    Algorithm:
    1. Train a Behavior Cloning (BC) model to estimate pi_beta(a|s).
    2. Train Q_beta(s,a) using Expected SARSA with pi_beta:
       Target = r + gamma * sum_a' ( pi_beta(a'|s') * Q_target(s', a') )
    3. Evaluation Policy: pi(s) = argmax_a Q_beta(s,a)
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list = [128, 128],
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Behavior Cloning Model (estimates pi_beta)
        self.bc_net = MLP(state_dim, n_actions, hidden_sizes).to(device)

        # Q-network (estimates Q^pi_beta)
        self.q_net = MLP(state_dim, n_actions, hidden_sizes).to(device)
        self.q_net_target = MLP(state_dim, n_actions, hidden_sizes).to(device)

        # Hard copy
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def act(self, state, budget=None, deterministic=True, training=False):
        """
        In One-Step RL, the policy is simply the argmax of the learned Q_beta.
        
        Args:
            state: [state_dim] or [batch, state_dim]
            budget: scalar, [batch] or None; if provided and state does not include budget, it is concatenated
            deterministic: bool
            training: bool (unused, for interface compatibility)
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # If budget provided and state dim < expected, concat budget
        if budget is not None:
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            
            current_state_dim = state.shape[-1]
            if current_state_dim < self.state_dim:
                # Handle budget dimensions
                if budget.dim() == 0:
                    budget = budget.unsqueeze(0).unsqueeze(-1)
                elif budget.dim() == 1:
                    budget = budget.unsqueeze(-1)
                elif budget.dim() == 2:
                    if budget.shape[1] != 1:
                        budget = budget[:, 0:1]
                
                # Ensure batch dimensions match
                if state.shape[0] != budget.shape[0]:
                    if budget.shape[0] == 1:
                        budget = budget.expand(state.shape[0], -1)
                    elif state.shape[0] == 1:
                        state = state.expand(budget.shape[0], -1)
                
                # Concat budget
                state = torch.cat([state, budget], dim=-1)
        
        # One-step improvement: Greedy wrt Q_beta
        with torch.no_grad():
            q_vals = self.q_net(state)
        
        if deterministic:
            actions = torch.argmax(q_vals, dim=-1)
        else:
            # Optional: Boltzmann exploration during evaluation if needed
            probs = F.softmax(q_vals, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
        return actions.cpu().numpy() if actions.shape[0] > 1 else actions.item()

    def compute_loss(self, batch):
        """
        Computes losses for both the BC component and the Q-learning component.
        batch: dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        """
        obs = to_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        acts = to_tensor(batch['actions'], dtype=torch.long, device=self.device)
        rews = to_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_obs = to_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        dones = to_tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # --- 1. Behavior Cloning Loss ---
        # Predict logits for actions
        bc_logits = self.bc_net(obs)
        loss_bc = F.cross_entropy(bc_logits, acts)

        # --- 2. Q-Learning Loss (Expected SARSA) ---
        # Current Q(s, a)
        q_pred = self.q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Estimate pi_beta(a'|s') using the BC network
            # Note: We use the current BC net, not a target BC net, as per standard implementations
            bc_next_logits = self.bc_net(next_obs)
            bc_next_probs = F.softmax(bc_next_logits, dim=-1) # [batch, n_actions]

            # Get Q_target(s', :) for all actions
            q_next_all = self.q_net_target(next_obs) # [batch, n_actions]

            # Expected Value over next actions: E_{a' ~ pi_beta} [Q(s', a')]
            # Sum(prob(a') * Q(s', a'))
            q_next_expected = (bc_next_probs * q_next_all).sum(dim=1)

            target_q = rews + self.gamma * (1 - dones) * q_next_expected

        loss_q = F.mse_loss(q_pred, target_q)

        return loss_bc, loss_q

    def update(self, batch, optimizer_bc, optimizer_q):
        """
        Updates both BC and Q networks.
        """
        loss_bc, loss_q = self.compute_loss(batch)

        # Update Behavior Cloning
        optimizer_bc.zero_grad()
        loss_bc.backward()
        # Optional: clip grad norm
        # torch.nn.utils.clip_grad_norm_(self.bc_net.parameters(), max_norm=10)
        optimizer_bc.step()

        # Update Q-function
        optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        optimizer_q.step()

        # Target soft update
        self._soft_update(self.q_net_target, self.q_net, self.tau)

        return loss_bc.item(), loss_q.item()

    def get_action_prob(self, state, budget=None):
        """
        Get action probability distribution (using BC network pi_beta).
        
        Args:
            state: [state_dim] or [batch, state_dim]
                If model was trained with budget, state should be [state_dim+1] or [batch, state_dim+1].
            budget: scalar, [batch] or None.
                If provided and state does not include budget, budget is concatenated to state.
        
        Returns:
            probs: [batch, action_dim] - action probability distribution
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Handle dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        
        # If budget provided and state dim < expected, concat budget
        if budget is not None:
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
            
            current_state_dim = state.shape[-1]
            
            if current_state_dim < self.state_dim:
                # Handle budget dimensions
                if budget.dim() == 0:
                    budget = budget.unsqueeze(0).unsqueeze(-1)  # scalar -> [1, 1]
                elif budget.dim() == 1:
                    budget = budget.unsqueeze(-1)  # [batch] -> [batch, 1]
                elif budget.dim() == 2:
                    if budget.shape[1] != 1:
                        budget = budget[:, 0:1]  # [batch, N] -> [batch, 1]
                else:
                    budget = budget.view(-1, 1)
                
                # Ensure batch dimensions match
                if state.shape[0] != budget.shape[0]:
                    if budget.shape[0] == 1:
                        budget = budget.expand(state.shape[0], -1)
                    elif state.shape[0] == 1:
                        state = state.expand(budget.shape[0], -1)
                    else:
                        raise ValueError(f"state and budget batch dim mismatch: state={state.shape[0]}, budget={budget.shape[0]}")
                
                # Concat budget to state
                state = torch.cat([state, budget], dim=-1)
        
        # Use BC network for policy distribution
        bc_logits = self.bc_net(state)  # [batch, n_actions]
        probs = F.softmax(bc_logits, dim=-1)  # [batch, n_actions]
        
        return probs

    def get_q_values(self, state, budget=None):
        """
        Get Q-values (Q_beta) for the given state.
        
        Args:
            state: [batch, state_dim]
            budget: Optional budget to concatenate
            
        Returns:
            q_values: [batch, n_actions]
        """
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Handle dimension if single sample
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Handle Budget Concatenation logic (copied from act method)
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

            # Forward pass
            q_vals = self.q_net(state)
            
            return q_vals