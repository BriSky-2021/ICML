import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .common.net import DiagGaussianActor, TransformerBlock, mlp

class CDT(nn.Module):
    """
    Constrained Decision Transformer (CDT).
    Only the main model is kept; training and logging are removed.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
        device="cpu"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.device = device
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)

        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.cost_prefix = cost_prefix
        if self.cost_prefix:
            self.prefix_emb = nn.Linear(1, embedding_dim)
            dt_seq_len += 1

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=dt_seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        if self.stochastic:
            self.log_temperature = torch.tensor(np.log(init_temperature), device=self.device)
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states: torch.Tensor,        # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,       # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor, # [batch_size, seq_len]
        costs_to_go: torch.Tensor,   # [batch_size, seq_len]
        time_steps: torch.Tensor,    # [batch_size, seq_len]
        padding_mask: torch.Tensor = None,  # [batch_size, seq_len]
        episode_cost: torch.Tensor = None,  # [batch_size, ]
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]
        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())
        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len, self.embedding_dim)

        if padding_mask is not None:
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(batch_size, -1)
        if self.cost_prefix:
            episode_cost = episode_cost.unsqueeze(-1).unsqueeze(-1)
            episode_cost = episode_cost.to(states.dtype)
            episode_cost_emb = self.prefix_emb(episode_cost)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        out = self.out_norm(out)
        if self.cost_prefix:
            out = out[:, 1:]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        out = out.permute(0, 2, 1, 3)
        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]
        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        action_preds = self.action_head(state_feat)
        cost_preds = self.cost_pred_head(action_feature)
        cost_preds = F.log_softmax(cost_preds, dim=-1)
        state_preds = self.state_pred_head(action_feature)
        return action_preds, cost_preds, state_preds

    def act(
        self,
        states: torch.Tensor,        # [batch, seq_len, state_dim] or [seq_len, state_dim]
        actions: torch.Tensor,       # [batch, seq_len, action_dim] or [seq_len, action_dim]
        returns_to_go: torch.Tensor, # [batch, seq_len] or [seq_len]
        costs_to_go: torch.Tensor,   # [batch, seq_len] or [seq_len]
        time_steps: torch.Tensor,    # [batch, seq_len] or [seq_len]
        episode_cost: torch.Tensor = None,
        deterministic: bool = True
    ):
        """
        Inference interface; supports single trajectory (no batch) and batched trajectories.
        """
        # Check whether there is a batch dimension
        single_input = False
        if states.dim() == 2:
            # [seq_len, state_dim] -> [1, seq_len, state_dim]
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            costs_to_go = costs_to_go.unsqueeze(0)
            time_steps = time_steps.unsqueeze(0)
            if episode_cost is not None and episode_cost.dim() == 0:
                episode_cost = episode_cost.unsqueeze(0)
            single_input = True
        elif states.dim() == 3:
            # [batch, seq_len, state_dim], leave as is
            pass
        else:
            raise ValueError("states must be [seq_len, state_dim] or [batch, seq_len, state_dim]")

        action_preds, _, _ = self.forward(
            states, actions, returns_to_go, costs_to_go, time_steps, None, episode_cost
        )
        if self.stochastic:
            action = action_preds.mean if deterministic else action_preds.rsample()
        else:
            action = action_preds
        # Take the last step
        action = action[:, -1]
        if not self.stochastic:
            action = action.clamp(-self.max_action, self.max_action)
        if single_input:
            return action.squeeze(0).detach().cpu().numpy()
        else:
            return action.detach().cpu().numpy()