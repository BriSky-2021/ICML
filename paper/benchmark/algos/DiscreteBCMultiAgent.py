import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepRNN(nn.Module):
    def __init__(self, input_dim, linear_layer_dim, recurrent_layer_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, linear_layer_dim)
        self.rnn = nn.GRU(linear_layer_dim, recurrent_layer_dim, batch_first=True)
        self.fc2 = nn.Linear(recurrent_layer_dim, output_dim)

    def forward(self, x, hxs=None):
        # x: [batch, T, input_dim]
        x = F.relu(self.fc1(x))
        out, hxs = self.rnn(x, hxs)
        logits = self.fc2(out)
        return logits, hxs

    def initial_state(self, batch, device):
        # [num_layers, batch, hidden_size]
        return torch.zeros(1, batch, self.rnn.hidden_size, device=device)

class DiscreteBCMultiAgent:
    def __init__(
        self,
        obs_dim,
        max_n_agents,
        action_dim,
        linear_layer_dim=64,
        recurrent_layer_dim=64,
        learning_rate=1e-3,
        add_agent_id_to_obs=True,
        device="cpu"
    ):
        self.obs_dim = obs_dim
        self.max_n_agents = max_n_agents
        self.action_dim = action_dim
        self.device = device
        self.add_agent_id_to_obs = add_agent_id_to_obs

        input_dim = obs_dim + (max_n_agents if add_agent_id_to_obs else 0)
        self.policy_network = DeepRNN(input_dim, linear_layer_dim, recurrent_layer_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def concat_agent_id(self, obs):
        # obs: [B, T, N, obs_dim]
        B, T, N, O = obs.shape
        agent_onehot = torch.eye(self.max_n_agents, device=obs.device)[None, None, :, :]  # [1,1,N,N]
        agent_onehot = agent_onehot[:, :, :N, :]  # [1,1,N,max_n_agents]
        agent_onehot = agent_onehot.expand(B, T, N, self.max_n_agents)
        obs = torch.cat([obs, agent_onehot], dim=-1)  # [B, T, N, obs_dim+max_n_agents]
        return obs

    def train_step(self, batch):
        """
        batch: dict, keys:
            'observations': [B, T, N, obs_dim]
            'actions':      [B, T, N]
            'agent_mask':   [B, N] (1/0)
        """
        obs = batch['observations']     # [B, T, N, obs_dim]
        actions = batch['actions']      # [B, T, N]
        agent_mask = batch['agent_mask']# [B, N]

        B, T, N, O = obs.shape

        # Expand agent_mask to [B, T, N]
        agent_mask_bt = agent_mask.unsqueeze(1).expand(B, T, N)

        # Concat agent_id
        if self.add_agent_id_to_obs:
            obs = self.concat_agent_id(obs)

        # Reshape to [B*N, T, ...]
        obs = obs.reshape(B * N, T, -1)
        actions = actions.reshape(B * N, T)
        agent_mask_bt = agent_mask_bt.reshape(B * N, T)

        # Keep only valid agents
        valid_idx = agent_mask_bt[:, 0] > 0
        obs = obs[valid_idx]
        actions = actions[valid_idx]

        # RNN state
        hxs = self.policy_network.initial_state(obs.shape[0], self.device)

        # Forward pass
        logits, _ = self.policy_network(obs, hxs)    # [B*N_valid, T, action_dim]
        logits = logits.reshape(-1, self.action_dim) # [B*N_valid*T, action_dim]
        actions = actions.reshape(-1)                # [B*N_valid*T]

        # Loss
        loss = F.cross_entropy(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'policy_loss': loss.item()}

    def act(self, obs, agent_mask=None, legal_actions=None,training=False):
        """
        obs: [B, N, obs_dim]
        agent_mask: [B, N] or None
        legal_actions: [B, N, action_dim] or None
        Returns: actions [B, N] (invalid agents as -1)
        """
        B, N, O = obs.shape
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(1)  # [B, 1, N, obs_dim]
        if self.add_agent_id_to_obs:
            obs = self.concat_agent_id(obs)
        obs = obs.reshape(B * N, 1, -1)
        hxs = self.policy_network.initial_state(B * N, self.device)
        logits, _ = self.policy_network(obs, hxs)
        logits = logits.squeeze(1)  # [B*N, action_dim]

        if legal_actions is not None:
            legal_actions = torch.tensor(legal_actions, dtype=torch.bool, device=self.device).reshape(-1, self.action_dim)
            logits[~legal_actions] = -1e8

        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1).reshape(B, N)
        actions = actions.cpu().numpy()
        if agent_mask is not None:
            actions[agent_mask == 0] = -1
        return actions

    def get_action_prob(self, obs, agent_mask=None, legal_actions=None):
        """
        Get action probability distribution.
        
        Args:
            obs: [B, N, obs_dim] or [B, obs_dim] (single agent) or numpy array
            agent_mask: [B, N] or None
            legal_actions: [B, N, action_dim] or None
        
        Returns:
            probs: [B, N, action_dim] or [B, action_dim] (single agent) - action probability distribution
        """
        # Convert to numpy for unified handling
        if torch.is_tensor(obs):
            obs_np = obs.cpu().numpy()
            is_tensor = True
        else:
            obs_np = np.array(obs)
            is_tensor = False
        
        # Handle dimensions: ensure [B, N, obs_dim] format
        if obs_np.ndim == 2:
            # Check if [B, obs_dim] or [B, N, obs_dim] (N=1 may be squeezed)
            if obs_np.shape[1] == self.obs_dim:
                # [B, obs_dim] -> [B, 1, obs_dim]
                obs_np = obs_np[:, np.newaxis, :]
            elif obs_np.shape[1] > self.obs_dim:
                # May include extra dims (e.g. budget); take first obs_dim
                obs_np = obs_np[:, np.newaxis, :self.obs_dim]
            else:
                raise ValueError(f"obs shape mismatch: expected at least {self.obs_dim} dims, got {obs_np.shape[1]}")
        elif obs_np.ndim == 3:
            # Already [B, N, obs_dim]; check last dim
            if obs_np.shape[2] > self.obs_dim:
                obs_np = obs_np[:, :, :self.obs_dim]
            elif obs_np.shape[2] < self.obs_dim:
                raise ValueError(f"obs shape mismatch: expected obs_dim={self.obs_dim}, got {obs_np.shape[2]}")
        elif obs_np.ndim == 1:
            # [obs_dim] -> [1, 1, obs_dim]
            if obs_np.shape[0] > self.obs_dim:
                obs_np = obs_np[:self.obs_dim]
            obs_np = obs_np[np.newaxis, np.newaxis, :]
        else:
            raise ValueError(f"Unsupported obs ndim: {obs_np.ndim}, expected 1, 2 or 3")
        
        B, N, O = obs_np.shape
        
        # Ensure O equals obs_dim (obs_dim may already include budget from training)
        if O != self.obs_dim:
            if O > self.obs_dim:
                obs_np = obs_np[:, :, :self.obs_dim]
                O = self.obs_dim
            else:
                raise ValueError(f"obs shape mismatch: expected obs_dim={self.obs_dim}, got O={O}")
        
        # Convert to tensor
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        
        # Add time dim: [B, N, obs_dim] -> [B, 1, N, obs_dim]
        obs = obs.unsqueeze(1)  # [B, 1, N, obs_dim]
        
        # Concat agent_id if required by model
        if self.add_agent_id_to_obs:
            obs = self.concat_agent_id(obs)  # [B, 1, N, obs_dim+max_n_agents]
        
        # Reshape: [B, 1, N, ...] -> [B*N, 1, ...]
        obs = obs.reshape(B * N, 1, -1)
        
        # Get initial state and forward
        hxs = self.policy_network.initial_state(B * N, self.device)
        logits, _ = self.policy_network(obs, hxs)
        logits = logits.squeeze(1)  # [B*N, action_dim]
        
        if legal_actions is not None:
            if torch.is_tensor(legal_actions):
                legal_actions_t = legal_actions
            else:
                legal_actions_t = torch.tensor(legal_actions, dtype=torch.bool, device=self.device)
            legal_actions_t = legal_actions_t.reshape(-1, self.action_dim)
            logits[~legal_actions_t] = -1e8
        
        probs = F.softmax(logits, dim=-1)  # [B*N, action_dim]
        probs = probs.reshape(B, N, self.action_dim)  # [B, N, action_dim]
        
        # If agent_mask given, zero out probabilities for invalid agents
        if agent_mask is not None:
            if torch.is_tensor(agent_mask):
                agent_mask_t = agent_mask
            else:
                agent_mask_t = torch.tensor(agent_mask, dtype=torch.float32, device=self.device)
            
            if agent_mask_t.dim() == 1:
                agent_mask_t = agent_mask_t.unsqueeze(0)  # [B] -> [B, 1]
            agent_mask_t = agent_mask_t.unsqueeze(-1)  # [B, N] -> [B, N, 1]
            probs = probs * agent_mask_t  # Zero out invalid agents
            # Renormalize
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # If single agent ([B, 1, action_dim]), squeeze N dim
        if N == 1:
            probs = probs.squeeze(1)  # [B, 1, action_dim] -> [B, action_dim]
        
        return probs