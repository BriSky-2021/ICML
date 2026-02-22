import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque
import random

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

class SimplifiedQMixer(nn.Module):
    """Simplified QMixer that adapts to observation dimension."""
    def __init__(self, n_agents, embed_dim=32, hyper_dim=64, obs_dim=4):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim
        
        # State dim: 4 statistics + active agent ratio
        self.state_dim = obs_dim * 4 + 1
        
        # Simple hypernetwork
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, n_agents * embed_dim)
        )
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(self.state_dim, hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, embed_dim)
        )
        self.hyper_b_1 = nn.Linear(self.state_dim, embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def _extract_state_features(self, obs, agent_mask):
        """Extract statistical features from observations as global state."""
        B, T, N, O = obs.shape
        
        # Adapt obs_dim dynamically
        actual_obs_dim = O
        if actual_obs_dim != self.obs_dim:
            self.obs_dim = actual_obs_dim
            self.state_dim = actual_obs_dim * 4 + 1
            
            # Re-initialize network layers
            device = next(self.parameters()).device
            hyper_dim = 64
            
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hyper_dim),
                nn.ReLU(),
                nn.Linear(hyper_dim, self.n_agents * self.embed_dim)
            ).to(device)
            
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim, hyper_dim),
                nn.ReLU(),
                nn.Linear(hyper_dim, self.embed_dim)
            ).to(device)
            
            self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim).to(device)
            
            self.hyper_b_2 = nn.Sequential(
                nn.Linear(self.state_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, 1)
            ).to(device)
        
        # Create output tensor
        state_features = torch.zeros(B, T, self.state_dim, device=obs.device, dtype=obs.dtype)
        
        for b in range(B):
            active_agents = int(agent_mask[b].sum().item())
            
            if active_agents > 0:
                active_obs = obs[b, :, :active_agents, :]
                
                for t in range(T):
                    obs_t = active_obs[t]
                    
                    mean_obs = obs_t.mean(dim=0)
                    
                    if active_agents > 1:
                        std_obs = obs_t.std(dim=0)
                    else:
                        std_obs = torch.zeros_like(mean_obs)
                    
                    min_obs = obs_t.min(dim=0)[0]
                    max_obs = obs_t.max(dim=0)[0]
                    
                    active_ratio = torch.tensor(active_agents / N, device=obs.device, dtype=obs.dtype)
                    
                    features = torch.cat([mean_obs, std_obs, min_obs, max_obs, active_ratio.unsqueeze(0)])
                    state_features[b, t] = features
        
        return state_features

    def forward(self, agent_qs, obs, agent_mask):
        """
        agent_qs: [B, T, n_agents]
        obs: [B, T, n_agents, obs_dim] - raw observations
        agent_mask: [B, n_agents] - agent mask
        Returns: [B, T, 1] Q_tot
        """
        B, T, N = agent_qs.shape
        
        # Extract simplified state features
        states = self._extract_state_features(obs, agent_mask)  # [B, T, state_dim]
        
        agent_qs_flat = agent_qs.reshape(B * T, N, 1)
        states_flat = states.reshape(B * T, self.state_dim)
        
        actual_n_agents = N
        
        # Generate weight matrices
        w1_full = torch.abs(self.hyper_w_1(states_flat)).reshape(B*T, self.n_agents, self.embed_dim)
        w1 = w1_full[:, :actual_n_agents, :]  # [B*T, N, embed_dim]
        
        b1 = self.hyper_b_1(states_flat).reshape(B*T, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(agent_qs_flat.transpose(1, 2), w1) + b1)  # [B*T, 1, embed_dim]

        w2 = torch.abs(self.hyper_w_2(states_flat)).reshape(B*T, self.embed_dim, 1)
        b2 = self.hyper_b_2(states_flat).reshape(B*T, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # [B*T, 1, 1]
        q_tot = q_tot.reshape(B, T, 1)
        return q_tot

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, batch):
        self.buffer.append(batch)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class QMIXCQLMultiAgent:
    def __init__(
        self,
        obs_dim,
        max_n_agents,
        action_dim,
        linear_layer_dim=64,
        recurrent_layer_dim=64,
        mixer_embed_dim=32,
        mixer_hyper_dim=64,
        cql_weight=3.0,  # Moderate CQL weight
        discount=0.95,
        learning_rate=1e-4,
        add_agent_id_to_obs=False,
        state_dim=None,
        tau=0.005,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        grad_clip_norm=5.0,
        buffer_capacity=100000,
        cql_temperature=0.5,  # Moderate temperature, balance conservatism and learning
        cql_min_q_weight=0.2,
        device="cpu"
    ):
        self.obs_dim = obs_dim
        self.max_n_agents = max_n_agents
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.cql_weight = cql_weight
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.grad_clip_norm = grad_clip_norm
        self.cql_temperature = cql_temperature
        self.cql_min_q_weight = cql_min_q_weight
        self.update_count = 0

        # Diversity params - moderate
        self.diversity_weight = 0.05  # Lower diversity weight
        self.min_entropy_threshold = 0.4  # Apply diversity only when entropy is too low
        self.exploration_temperature = 0.1  # Lower exploration temp, favor best action

        # obs + one-hot agent id
        input_dim = obs_dim + (max_n_agents if add_agent_id_to_obs else 0)
        self.q_network = DeepRNN(input_dim, linear_layer_dim, recurrent_layer_dim, action_dim).to(device)
        self.target_q_network = DeepRNN(input_dim, linear_layer_dim, recurrent_layer_dim, action_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Use simplified Mixer
        self.mixer = SimplifiedQMixer(max_n_agents, mixer_embed_dim, mixer_hyper_dim, obs_dim).to(device)
        self.target_mixer = SimplifiedQMixer(max_n_agents, mixer_embed_dim, mixer_hyper_dim, obs_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Reasonable initialization
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=0.5)  # Moderate gain
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.GRU, torch.nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_normal_(param, gain=0.5)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0)
        
        self.q_network.apply(init_weights)
        self.mixer.apply(init_weights)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Algorithm identifier
        self.algorithm_type = 'marl'
        self.needs_budget_input = False
        self.has_budget_in_obs = True

    def concat_agent_id(self, obs):
        # obs: [B, T, N, obs_dim]
        B, T, N, O = obs.shape
        agent_onehot = torch.eye(self.max_n_agents, device=obs.device)[None, None, :, :]
        agent_onehot = agent_onehot[:, :, :N, :]
        agent_onehot = agent_onehot.expand(B, T, N, self.max_n_agents)
        obs = torch.cat([obs, agent_onehot], dim=-1)
        return obs

    def compute_adaptive_cql_loss(self, q_seq, q_seq_a, legal_actions_flat, actions_valid):
        """
        Adaptive CQL loss: adjust constraint strength by current learning state.
        """
        n_valid, T, action_dim = q_seq.shape
        
        # 1. Standard CQL loss
        q_cql = q_seq.clone()
        
        if legal_actions_flat.any():
            min_legal_q = q_cql[legal_actions_flat].min().detach()
            q_cql[~legal_actions_flat] = min_legal_q - 10.0
        
        temperature = self.cql_temperature
        q_scaled = q_cql / temperature
        
        # Numerically stable logsumexp
        max_q = q_scaled.max(dim=-1, keepdim=True)[0].detach()
        stable_q = q_scaled - max_q
        logsumexp_q = max_q.squeeze(-1) + torch.logsumexp(stable_q, dim=-1)
        
        dataset_q = q_seq_a / temperature
        cql_loss_basic = (logsumexp_q - dataset_q).mean()
        
        # 2. Adaptive diversity constraint
        q_probs = F.softmax(q_seq / 0.5, dim=-1)
        action_entropy = -(q_probs * torch.log(q_probs + 1e-8)).sum(dim=-1).mean()
        
        # Apply diversity only when entropy is too low
        if action_entropy < self.min_entropy_threshold:
            diversity_loss = -self.diversity_weight * (action_entropy - self.min_entropy_threshold)
        else:
            diversity_loss = torch.tensor(0.0, device=q_seq.device)
        
        # 3. Q distribution constraint - encourage meaningful spread
        q_values_normalized = q_seq - q_seq.mean(dim=-1, keepdim=True)
        q_spread = q_values_normalized.std(dim=-1).mean()
        
        # Encourage moderate Q spread (not too flat nor too extreme)
        target_spread = 0.01
        spread_loss = 0.1 * (q_spread - target_spread).abs()
        
        # 4. Weakened min-Q regularization
        actions_onehot = torch.zeros_like(q_seq)
        actions_onehot.scatter_(-1, actions_valid.unsqueeze(-1), 1.0)
        q_non_dataset_masked = q_seq * (1 - actions_onehot)
        min_q_loss = q_non_dataset_masked.mean() * self.cql_min_q_weight
        
        total_cql_loss = cql_loss_basic + min_q_loss + diversity_loss + spread_loss
        total_cql_loss = torch.clamp(total_cql_loss, min=0.001, max=5.0)
        
        return total_cql_loss, {
            'cql_basic': cql_loss_basic.item(),
            'min_q_loss': min_q_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'action_entropy': action_entropy.item(),
            'q_spread': q_spread.item(),
            'spread_loss': spread_loss.item()
        }

    def compute_reward_aggregation(self, rewards_full, agent_mask, method='weighted_mean'):
        """Improved reward aggregation."""
        B, T, N = rewards_full.shape
        agent_mask_expanded = agent_mask.unsqueeze(1).expand(B, T, N)
        valid_rewards = rewards_full * agent_mask_expanded.float()
        
        if method == 'weighted_mean':
            num_valid = torch.clamp(agent_mask_expanded.float().sum(dim=2, keepdim=True), min=1.0)
            rewards_agg = valid_rewards.sum(dim=2, keepdim=True) / num_valid
        elif method == 'sum':
            rewards_agg = valid_rewards.sum(dim=2, keepdim=True)
        else:
            num_valid = torch.clamp(agent_mask_expanded.float().sum(dim=2, keepdim=True), min=1.0)
            rewards_agg = valid_rewards.sum(dim=2, keepdim=True) / num_valid
        
        return rewards_agg

    def train_step(self, batch):
        """Improved training step."""
        obs_all = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        legal_actions = batch['legal_actions']
        agent_mask = batch['agent_mask']
        
        obs = obs_all[:, :-1]
        next_obs = obs_all[:, 1:]
        B, T, N, O = obs.shape

        if agent_mask.shape != (B, N):
            raise ValueError(f"Incorrect agent_mask shape! Expected {(B, N)}, but got {agent_mask.shape}")

        agent_mask_bt = agent_mask.unsqueeze(1).expand(B, T, N)

        if self.add_agent_id_to_obs:
            obs = self.concat_agent_id(obs)
            next_obs = self.concat_agent_id(next_obs)

        obs_flat = obs.reshape(B*N, T, -1)
        next_obs_flat = next_obs.reshape(B*N, T, -1)
        actions_flat = actions.reshape(B*N, T)
        agent_mask_bt_flat = agent_mask_bt.reshape(B*N, T)

        valid_idx = agent_mask_bt_flat[:, 0] > 0
        obs_valid = obs_flat[valid_idx]
        next_obs_valid = next_obs_flat[valid_idx]
        actions_valid = actions_flat[valid_idx]
        n_valid = obs_valid.shape[0]

        if n_valid == 0:
            return {'loss': 0.0, 'td_loss': 0.0, 'cql_loss': 0.0, 'skipped': True}

        hxs = self.q_network.initial_state(n_valid, self.device)
        target_hxs = self.target_q_network.initial_state(n_valid, self.device)

        q_seq, _ = self.q_network(obs_valid, hxs)
        q_seq_a = q_seq.gather(-1, actions_valid.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_seq, _ = self.target_q_network(next_obs_valid, target_hxs)
            legal_actions_flat = legal_actions.permute(0,2,1,3).reshape(B*N, T, self.action_dim)[valid_idx]
            next_q_seq[~legal_actions_flat] = -1e8
            max_next_q = next_q_seq.max(-1)[0]

        # Q-mixer part
        q_seq_a_full = torch.zeros(B*N, T, device=self.device)
        q_seq_a_full[valid_idx] = q_seq_a
        q_seq_a_full = q_seq_a_full.reshape(B, N, T).permute(0, 2, 1)
        q_tot = self.mixer(q_seq_a_full, obs, agent_mask)

        max_next_q_full = torch.zeros(B*N, T, device=self.device)
        max_next_q_full[valid_idx] = max_next_q
        max_next_q_full = max_next_q_full.reshape(B, N, T).permute(0, 2, 1)
        target_q_tot = self.target_mixer(max_next_q_full, next_obs, agent_mask)

        # Reward aggregation
        rewards_agg = self.compute_reward_aggregation(rewards, agent_mask, method='weighted_mean')
        
        dones_expanded = dones.unsqueeze(-1).expand(B, T, 1)
        target = rewards_agg + self.discount * (1 - dones_expanded) * target_q_tot
        
        # TD loss
        td_loss = F.smooth_l1_loss(q_tot, target.detach())

        # Adaptive CQL loss
        legal_actions_valid = legal_actions.permute(0,2,1,3).reshape(B*N, T, self.action_dim)[valid_idx]
        
        # Gradually adjust CQL weight
        if self.update_count < 300:
            effective_cql_weight = self.cql_weight * 0.2
        elif self.update_count < 800:
            effective_cql_weight = self.cql_weight * 0.6
        elif self.update_count < 1500:
            effective_cql_weight = self.cql_weight * 0.8
        else:
            effective_cql_weight = self.cql_weight
        
        cql_loss, cql_details = self.compute_adaptive_cql_loss(
            q_seq, q_seq_a, legal_actions_valid, actions_valid
        )

        # Total loss
        loss = td_loss + effective_cql_weight * cql_loss

        # Debug info
        if self.update_count % 100 == 0:
            with torch.no_grad():
                q_values = q_seq.mean(dim=1)
                action_probs = F.softmax(q_values / self.exploration_temperature, dim=-1)
                
                print(f"\n=== Update {self.update_count} Debug Info ===")
                print(f"Active agents: {n_valid}/{B*N}")
                print(f"Effective CQL weight: {effective_cql_weight:.3f}")
                print(f"Action distribution (model): {action_probs.mean(dim=0).cpu().numpy()}")
                print(f"Q-values range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
                print(f"Q-values mean: {q_values.mean(dim=0).cpu().numpy()}")
                print(f"Q-values std: {q_values.std(dim=0).cpu().numpy()}")
                print(f"Q-values spread: {cql_details['q_spread']:.6f}")
                print(f"TD loss: {td_loss.item():.4f}")
                print(f"CQL basic: {cql_details['cql_basic']:.4f}")
                print(f"Action entropy: {cql_details['action_entropy']:.4f}")
                print(f"Diversity loss: {cql_details['diversity_loss']:.4f}")
                print(f"Total CQL: {cql_loss.item():.4f}")
                print(f"Total loss: {loss.item():.4f}")
                
                # Dataset action distribution
                dataset_actions = actions_valid.reshape(-1)
                unique, counts = torch.unique(dataset_actions, return_counts=True)
                action_dist = torch.zeros(self.action_dim, device=self.device)
                action_dist[unique] = counts.float()
                action_dist = action_dist / action_dist.sum()
                print(f"Dataset action dist: {action_dist.cpu().numpy()}")
                print("=" * 50)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected. Skipping update.")
            return {'loss': 0.0, 'td_loss': td_loss.item(), 'cql_loss': cql_loss.item(), 'skipped': True}

        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            self.grad_clip_norm
        )
        
        self.optimizer.step()

        # Target network update
        self.update_count += 1
        if self.update_count % 5 == 0:
            with torch.no_grad():
                for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'loss': loss.item(),
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q_mean': q_tot.mean().item(),
            'grad_norm': grad_norm.item(),
            'effective_cql_weight': effective_cql_weight,
            'action_entropy': cql_details['action_entropy'],
            'q_spread': cql_details['q_spread'],
            'skipped': False
        }

    def _strategic_action_selection(self, obs, agent_mask, legal_actions=None, training=True):
        """
        Strategic action selection: avoid over-concentration while learning meaningful policy.
        """
        B, N, O = obs.shape
        
        # 1. Get individual Q-values
        obs_for_q = obs.clone()
        if self.add_agent_id_to_obs:
            obs_for_q = self.concat_agent_id(obs_for_q.unsqueeze(1)).squeeze(1)
        
        obs_flat = obs_for_q.reshape(B * N, 1, -1)
        hxs = self.q_network.initial_state(B * N, self.device)
        q_individual, _ = self.q_network(obs_flat, hxs)
        q_individual = q_individual.squeeze(1).reshape(B, N, self.action_dim)
        
        # 2. Apply legal_actions mask
        if legal_actions is not None:
            legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.bool, device=self.device)
            q_individual[~legal_actions_tensor] = -1e8
        
        # 3. Action selection strategy
        if training:
            # Training: Boltzmann exploration with temperature decreasing over time
            progress = min(1.0, self.update_count / 2000.0)
            temperature = self.exploration_temperature * (2.0 - progress)  # from ~0.2 down to ~0.1
            
            action_logits = q_individual / temperature
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample with bias towards better actions
            actions = torch.multinomial(action_probs.view(-1, self.action_dim), 1).view(B, N)
            
        else:
            # Eval: lower temperature, favor best action
            temperature = 0.05
            action_probs = F.softmax(q_individual / temperature, dim=-1)
            
            # Sample from top-k actions instead of pure greedy
            top_k = min(3, self.action_dim)
            _, top_indices = torch.topk(q_individual, top_k, dim=-1)
            
            # Mask to allow only top-k actions
            top_k_mask = torch.zeros_like(q_individual, dtype=torch.bool)
            top_k_mask.scatter_(-1, top_indices, True)
            
            masked_probs = action_probs.clone()
            masked_probs[~top_k_mask] = 0
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            actions = torch.multinomial(masked_probs.view(-1, self.action_dim), 1).view(B, N)
        
        return actions

    def act(self, obs, agent_mask, legal_actions=None, training=True):
        """Improved action selection."""
        if training is None:
            training = self.training
            
        B, N, O = obs.shape
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if agent_mask is not None:
            agent_mask_tensor = torch.tensor(agent_mask, dtype=torch.float32, device=self.device)
        else:
            agent_mask_tensor = torch.ones(B, N, device=self.device)
        
        with torch.no_grad():
            # Reduce probability of fully random actions
            use_random = training and np.random.random() < (self.epsilon * 0.1)
            
            if use_random:
                # Even "random" should be somewhat strategic
                if legal_actions is not None:
                    legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.bool, device=self.device)
                    actions = torch.zeros(B, N, dtype=torch.long, device=self.device)
                    for b in range(B):
                        for n in range(N):
                            if agent_mask_tensor[b, n] > 0:
                                legal_idx = torch.where(legal_actions_tensor[b, n])[0]
                                if len(legal_idx) > 0:
                                    actions[b, n] = legal_idx[torch.randint(len(legal_idx), (1,))]
                                else:
                                    actions[b, n] = 0
                else:
                    actions = torch.randint(0, self.action_dim, (B, N), device=self.device)
                    actions = actions * agent_mask_tensor.long()
            else:
                # Use strategic action selection
                actions = self._strategic_action_selection(obs_tensor, agent_mask_tensor, legal_actions, training)
            
            # Apply agent_mask
            actions = actions.cpu().numpy()
            if agent_mask is not None:
                actions[agent_mask == 0] = -1
        
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return actions

    def save_model(self, path):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)

    def load_model(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.update_count = checkpoint.get('update_count', 0)

    def add_experience(self, batch):
        """Add experience to replay buffer."""
        self.replay_buffer.push(batch)

    def sample_batch(self, batch_size):
        """Sample from replay buffer."""
        return self.replay_buffer.sample(batch_size)

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
        
        if agent_mask is not None:
            agent_mask_tensor = torch.tensor(agent_mask, dtype=torch.float32, device=device)
        else:
            agent_mask_tensor = torch.ones(B, N, device=device)
        
        # Handle obs format: need [B, T, N, obs_dim]
        obs = obs.unsqueeze(1)  # [B, 1, N, obs_dim]
        
        # Concat agent_id
        if self.add_agent_id_to_obs:
            obs = self.concat_agent_id(obs)  # [B, 1, N, obs_dim+max_n_agents]
        
        # Reshape to [B*N, T, ...]
        obs = obs.reshape(B * N, 1, -1)
        
        # Get Q-values
        hxs = self.q_network.initial_state(B * N, device)
        q, _ = self.q_network(obs, hxs)
        q = q.squeeze(1)  # [B*N, action_dim]
        
        if legal_actions is not None:
            legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.bool, device=device)
            legal_actions_flat = legal_actions_tensor.reshape(-1, self.action_dim)
            q[~legal_actions_flat] = -1e8
        
        # Use softmax to convert Q to probability distribution
        probs = F.softmax(q, dim=-1)  # [B*N, action_dim]
        probs = probs.reshape(B, N, self.action_dim)  # [B, N, action_dim]
        
        # Apply agent_mask
        agent_mask_tensor = agent_mask_tensor.unsqueeze(-1)  # [B, N] -> [B, N, 1]
        probs = probs * agent_mask_tensor
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Single agent: squeeze N dim
        if N == 1:
            probs = probs.squeeze(1)  # [B, 1, action_dim] -> [B, action_dim]
        
        return probs

    def get_q_values(self, obs):
        """
        Get individual Q-values (before mixing) for the given observation.
        
        Args:
            obs: [B, N, obs_dim]
            
        Returns:
            q_values: [B, N, n_actions]
        """
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
            B, N, _ = obs.shape
            
            # Add time dimension: [B, N, D] -> [B, 1, N, D]
            obs_in = obs.unsqueeze(1)
            
            # Add agent IDs if configured
            if self.add_agent_id_to_obs:
                obs_in = self.concat_agent_id(obs_in)
            
            # Flatten for RNN: [B*N, 1, Input_Dim]
            obs_in = obs_in.reshape(B * N, 1, -1)
            
            # Initialize hidden states
            hxs = self.q_network.initial_state(B * N, self.device)
            
            # Forward pass
            q_vals, _ = self.q_network(obs_in, hxs) # [B*N, 1, n_actions]
            
            # Reshape back: [B, N, n_actions]
            q_vals = q_vals.squeeze(1).reshape(B, N, self.action_dim)
            
            return q_vals