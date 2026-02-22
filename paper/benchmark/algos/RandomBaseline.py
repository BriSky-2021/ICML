import torch as th
import numpy as np
import random

class RandomBaseline:
    """
    Random baseline: sample actions uniformly for baseline comparison.
    """
    
    def __init__(self, action_dim, device='cpu', seed=None):
        """
        Initialize random baseline.
        
        Args:
            action_dim: action space size
            device: device
            seed: random seed
        """
        self.action_dim = action_dim
        self.device = device
        
        if seed is not None:
            self.set_seed(seed)
    
    def set_seed(self, seed):
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
    
    def act(self, obs, budget=None, legal_actions=None, deterministic=False,training=False):
        """
        Sample a random action.
        
        Args:
            obs: observation [batch_size, obs_dim] or [obs_dim]
            budget: budget (optional)
            legal_actions: legal action mask [batch_size, action_dim] or [action_dim]
            deterministic: whether deterministic (ignored for random policy)
        
        Returns:
            actions: selected action(s)
        """
        if isinstance(obs, th.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = obs
        
        # Handle batch dimension
        if obs_np.ndim == 1:
            batch_size = 1
            single_obs = True
        else:
            batch_size = obs_np.shape[0]
            single_obs = False
        
        actions = []
        
        for i in range(batch_size):
            if legal_actions is not None:
                # Use legal action mask
                if isinstance(legal_actions, th.Tensor):
                    legal_mask = legal_actions.cpu().numpy()
                else:
                    legal_mask = legal_actions
                
                if legal_mask.ndim == 1:
                    current_legal = legal_mask
                else:
                    current_legal = legal_mask[i]
                
                # Find all legal actions
                legal_action_indices = np.where(current_legal)[0]
                
                if len(legal_action_indices) > 0:
                    # Random choice among legal actions
                    action = np.random.choice(legal_action_indices)
                else:
                    # No legal action: use action 0 (no-op)
                    action = 0
            else:
                # No legal constraint: random over full action space
                action = np.random.randint(0, self.action_dim)
            
            actions.append(action)
        
        actions = np.array(actions)
        
        if single_obs:
            return actions[0]
        else:
            return actions
    
    def predict(self, obs, budget=None, legal_actions=None):
        """
        Predict action (compatible with evaluation interfaces).
        """
        return self.act(obs, budget, legal_actions, deterministic=True)
    
    def get_action(self, obs, budget=None, legal_actions=None):
        """
        Get action (compatible with evaluation interfaces).
        """
        return self.act(obs, budget, legal_actions)

class RandomBaselineOSRL(RandomBaseline):
    """
    Random baseline for OSRL evaluation interface.
    
    Default: uniform random over all actions (ignores budget).
    With use_budget_constraint=True: random within affordable actions only.
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', seed=None, use_budget_constraint=False):
        super().__init__(action_dim, device, seed)
        self.state_dim = state_dim
        self.use_budget_constraint = use_budget_constraint
    
    def act(self, obs, budget, deterministic=False):
        """
        act for OSRL interface.
        
        Args:
            obs: observation [obs_dim] or [batch_size, obs_dim]
            budget: budget scalar or [batch_size]
            deterministic: whether deterministic
        
        Returns:
            action: selected action(s)
        """
        if isinstance(obs, th.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.array(obs)
        
        if obs_np.ndim == 1:
            batch_size = 1
            single_obs = True
        else:
            batch_size = obs_np.shape[0]
            single_obs = False
        
        actions = []
        
        # Action costs used when budget constraint is enabled
        action_costs = [0, 71.56, 1643.31, 2433.53]
        
        for i in range(batch_size):
            if not self.use_budget_constraint:
                # Uniform random
                action = int(np.random.randint(0, self.action_dim))
            else:
                # Random within affordable actions
                if isinstance(budget, (list, np.ndarray)):
                    current_budget = budget[i] if not single_obs else budget[0]
                else:
                    current_budget = budget
                affordable_actions = [idx for idx, cost in enumerate(action_costs) if current_budget >= cost]
                if len(affordable_actions) > 0:
                    action = int(np.random.choice(affordable_actions))
                else:
                    action = 0
            actions.append(action)
        
        actions = np.array(actions)
        
        if single_obs:
            return actions[0]
        else:
            return actions

class RandomBaselineMARL(RandomBaseline):
    """
    Random baseline for MARL evaluation interface.
    """
    
    def __init__(self, obs_dim, max_n_agents, action_dim, device='cpu', seed=None, use_legal_constraint=True):
        super().__init__(action_dim, device, seed)
        self.obs_dim = obs_dim
        self.max_n_agents = max_n_agents
        self.use_legal_constraint = use_legal_constraint
    
    def act(self, obs, agent_mask=None, legal_actions=None, deterministic=False,training=False):
        """
        Same interface as other MARL algorithms:
        - Inputs:
            obs: [B, N, obs_dim]
            agent_mask: [B, N] (optional)
            legal_actions: [B, N, action_dim] (optional)
        - Returns:
            actions: [B, N], invalid agents set to -1
        """
        # To numpy
        if isinstance(obs, th.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs)
        B, N, _ = obs_np.shape

        # Normalize agent_mask -> [B, N] or None
        mask = None
        if agent_mask is not None:
            if isinstance(agent_mask, th.Tensor):
                mask = agent_mask.detach().cpu().numpy()
            else:
                mask = np.asarray(agent_mask)
            mask = mask.astype(bool)

        # Normalize legal_actions -> [B, N, action_dim] or None (controlled by flag)
        legal = None
        if self.use_legal_constraint and (legal_actions is not None):
            if isinstance(legal_actions, th.Tensor):
                legal = legal_actions.detach().cpu().numpy()
            else:
                legal = np.asarray(legal_actions)
            legal = legal.astype(bool)

        actions = np.full((B, N), -1, dtype=np.int64)

        for b in range(B):
            for n in range(N):
                # Skip invalid agents
                if mask is not None and not mask[b, n]:
                    actions[b, n] = -1
                    continue

                if legal is not None:
                    legal_indices = np.where(legal[b, n])[0]
                    if len(legal_indices) > 0:
                        actions[b, n] = int(np.random.choice(legal_indices))
                    else:
                        actions[b, n] = 0
                else:
                    actions[b, n] = int(np.random.randint(0, self.action_dim))

        return actions