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
        q = self.fc2(out)
        return q, hxs

    def initial_state(self, batch, device):
        return torch.zeros(1, batch, self.rnn.hidden_size, device=device)

class IQLCQLMultiAgent:
    def __init__(
        self,
        obs_dim,
        max_n_agents,
        action_dim,
        linear_layer_dim=256,
        recurrent_layer_dim=256,
        cql_weight=1.0,
        discount=0.99,
        learning_rate=1e-4,
        device="cpu"
    ):
        self.obs_dim = obs_dim
        self.max_n_agents = max_n_agents
        self.action_dim = action_dim
        self.device = device
        input_dim = obs_dim + max_n_agents    # concat one-hot agent id
        self.q_network = DeepRNN(input_dim, linear_layer_dim, recurrent_layer_dim, action_dim).to(device)
        self.target_q_network = DeepRNN(input_dim, linear_layer_dim, recurrent_layer_dim, action_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.discount = discount
        self.cql_weight = cql_weight

    def concat_agent_id(self, obs):
        # obs: [B, T, N, obs_dim]
        B, T, N, O = obs.shape
        agent_onehot = torch.eye(self.max_n_agents, device=obs.device)[None,None,:,:] # [1,1,N,N]
        agent_onehot = agent_onehot[:, :, :N, :]  # [1,1,N,max_n_agents]
        agent_onehot = agent_onehot.expand(B, T, N, self.max_n_agents)
        obs = torch.cat([obs, agent_onehot], dim=-1)  # [B, T, N, obs_dim+max_n_agents]
        return obs

    def train_step(self, batch):
        """
        batch: dict
            'observations': [B, T+1, N, obs_dim]
            'actions': [B, T, N]
            'rewards': [B, T, N]
            'dones': [B, T]
            'legal_actions': [B, T, N, action_dim]
            'agent_mask': [B, N] (1/0)
        """
        obs = batch['observations'][:, :-1]      # [B, T, N, obs_dim]
        next_obs = batch['observations'][:, 1:]  # [B, T, N, obs_dim]
        T = obs.shape[1]

        actions = batch['actions'][:, :T]        # [B, T, N]
        rewards = batch['rewards'][:, :T]        # [B, T, N]
        dones = batch['dones'][:, :T]            # [B, T]
        legal_actions = batch['legal_actions'][:, :T]   # [B, T, N, action_dim]
        agent_mask = batch['agent_mask']         # [B, N]

        B, T, N, O = obs.shape

        # Expand agent_mask to [B, T, N]
        agent_mask_bt = agent_mask.unsqueeze(1).expand(B, T, N)  # [B, T, N]

        # Concat agent id
        obs = self.concat_agent_id(obs)      # [B, T, N, obs_dim+max_n_agents]
        next_obs = self.concat_agent_id(next_obs)

        # reshape to [B*N, T, ...]
        obs = obs.reshape(B*N, T, -1)
        next_obs = next_obs.reshape(B*N, T, -1)
        actions = actions.reshape(B*N, T)
        rewards = rewards.reshape(B*N, T)
        legal_actions = legal_actions.reshape(B*N, T, self.action_dim)
        dones = dones.unsqueeze(2).expand(B, T, N).reshape(B*N, T)
        agent_mask_bt = agent_mask_bt.reshape(B*N, T)  # [B*N, T]

        # Keep only valid agents
        valid_idx = agent_mask_bt[:, 0] > 0
        obs = obs[valid_idx]
        next_obs = next_obs[valid_idx]
        actions = actions[valid_idx]
        rewards = rewards[valid_idx]
        legal_actions = legal_actions[valid_idx]
        dones = dones[valid_idx]
        agent_mask_bt = agent_mask_bt[valid_idx]

        # RNN state
        hxs = self.q_network.initial_state(obs.shape[0], self.device)
        target_hxs = self.target_q_network.initial_state(obs.shape[0], self.device)

        # Q network
        q_seq, _ = self.q_network(obs, hxs)              # [B*N_valid, T, action_dim]
        q_seq_a = q_seq.gather(-1, actions.unsqueeze(-1)).squeeze(-1)   # [B*N_valid, T]
        # Target Q
        with torch.no_grad():
            next_q_seq, _ = self.target_q_network(next_obs, target_hxs)
            next_q_seq[~legal_actions] = -1e8  # Mask illegal actions
            max_next_q = next_q_seq.max(-1)[0]  # [B*N_valid, T]

        target = rewards + self.discount * (1-dones) * max_next_q
        td_loss = F.mse_loss(q_seq_a, target)

        # CQL loss
        logsumexp = torch.logsumexp(q_seq, dim=-1)   # [B*N_valid, T]
        cql_loss = (logsumexp - q_seq_a).mean()

        loss = td_loss + self.cql_weight * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target
        tau = 0.005
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        #print(f"Debug: loss: {loss.item()}, td_loss: {td_loss.item()}, cql_loss: {cql_loss.item()}")

        return {'loss': loss.item(), 'td_loss': td_loss.item(), 'cql_loss': cql_loss.item()}

    def act(self, obs, agent_mask, legal_actions=None,training=True):
        """
        obs: [B, N, obs_dim]
        agent_mask: [B, N]
        legal_actions: [B, N, action_dim] or None
        """
        B, N, O = obs.shape
        device = self.device
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        obs = obs.unsqueeze(1)  # [B, 1, N, obs_dim]
        obs = self.concat_agent_id(obs)  # [B, 1, N, obs_dim+max_n_agents]
        obs = obs.reshape(B * N, 1, -1)
        hxs = self.q_network.initial_state(B * N, device)
        q, _ = self.q_network(obs, hxs)
        q = q.squeeze(1)  # [B*N, action_dim]
        if legal_actions is not None:
            legal_actions = torch.tensor(legal_actions, dtype=torch.bool, device=device).reshape(-1, self.action_dim)
            q[~legal_actions] = -1e8
        act = torch.argmax(q, dim=-1).reshape(B, N)
        # Zero out invalid agents with agent_mask
        act = act.cpu().numpy()
        act[agent_mask == 0] = -1   # Invalid agents set to -1
        return act

    def get_action_prob(self, obs, agent_mask=None, legal_actions=None):
        """
        Get action probability distribution (softmax over Q-values).
        
        Args:
            obs: [B, N, obs_dim]
            agent_mask: [B, N] or None
            legal_actions: [B, N, action_dim] or None
        
        Returns:
            probs: [B, N, action_dim] - action probability distribution
        """
        B, N, O = obs.shape
        device = self.device
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        obs = obs.unsqueeze(1)  # [B, 1, N, obs_dim]
        obs = self.concat_agent_id(obs)  # [B, 1, N, obs_dim+max_n_agents]
        obs = obs.reshape(B * N, 1, -1)
        hxs = self.q_network.initial_state(B * N, device)
        q, _ = self.q_network(obs, hxs)
        q = q.squeeze(1)  # [B*N, action_dim]
        
        if legal_actions is not None:
            legal_actions = torch.tensor(legal_actions, dtype=torch.bool, device=device).reshape(-1, self.action_dim)
            q[~legal_actions] = -1e8
        
        # Use softmax to convert Q to probability (Boltzmann)
        probs = F.softmax(q, dim=-1)  # [B*N, action_dim]
        probs = probs.reshape(B, N, self.action_dim)  # [B, N, action_dim]
        
        # If agent_mask given, zero out invalid agents
        if agent_mask is not None:
            agent_mask_t = torch.tensor(agent_mask, dtype=torch.float32, device=device)
            if agent_mask_t.dim() == 1:
                agent_mask_t = agent_mask_t.unsqueeze(0)
            agent_mask_t = agent_mask_t.unsqueeze(-1)  # [B, N] -> [B, N, 1]
            probs = probs * agent_mask_t
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Single agent: squeeze N dim
        if N == 1:
            probs = probs.squeeze(1)  # [B, 1, action_dim] -> [B, action_dim]
        
        return probs

    def get_q_values(self, obs):
        """
        Get Q-values for the given observation.
        
        Args:
            obs: [B, N, obs_dim] (Tensor or numpy)
            
        Returns:
            q_values: [B, N, n_actions] (Tensor)
        """
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            B, N, _ = obs.shape
            
            # Add time dimension: [B, N, D] -> [B, 1, N, D]
            obs_in = obs.unsqueeze(1)
            
            # Add agent IDs
            obs_in = self.concat_agent_id(obs_in) # [B, 1, N, D+N]
            
            # Flatten for RNN: [B*N, 1, D+N]
            obs_in = obs_in.reshape(B * N, 1, -1)
            
            # Initialize hidden states
            hxs = self.q_network.initial_state(B * N, self.device)
            
            # Forward pass
            q_vals, _ = self.q_network(obs_in, hxs) # [B*N, 1, n_actions]
            
            # Reshape back: [B, N, n_actions]
            q_vals = q_vals.squeeze(1).reshape(B, N, self.action_dim)
            
            return q_vals