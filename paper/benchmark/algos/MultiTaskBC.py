import torch
import torch.nn as nn
import numpy as np
# Simple Logger placeholder
class Logger:
    def store(self, **kwargs):
        pass

class MLPActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim, a_hidden_sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in a_hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))  # discrete action output
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.network(obs)  # [batch, action_dim]
        return logits

class MultiTaskBC(nn.Module):
    """
    Multi-Task Behavior Cloning for Discrete Actions with Dynamic Budget
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 a_hidden_sizes: list = [128, 128],
                 device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_hidden_sizes = a_hidden_sizes
        self.device = device
        # +1 for concatenated budget
        self.actor = MLPActorDiscrete(self.state_dim + 1, self.action_dim, self.a_hidden_sizes).to(self.device)

    def actor_loss(self, observations, budgets, actions):
        """
        :param observations: torch.Tensor [batch, state_dim]
        :param budgets: torch.Tensor [batch]  # budget per sample
        :param actions: torch.Tensor [batch]  # LongTensor, action class indices
        """
        obs_with_budget = torch.cat([observations, budgets.unsqueeze(-1)], dim=1)
        logits = self.actor(obs_with_budget)
        loss_actor = nn.CrossEntropyLoss()(logits, actions)
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}
        return loss_actor, stats_actor

    def setup_optimizers(self, actor_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def act(self, obs, budget, deterministic=True):
        """
        :param obs: numpy array or torch.Tensor [state_dim] or [1, state_dim]
        :param budget: float
        :param deterministic: bool
        :return: int (action class index)
        """
        # enforce float32
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(dtype=torch.float32, device=self.device)
        
        # ensure obs is 1D [state_dim]
        if obs.dim() > 1:
            obs = obs.squeeze()

        obs_with_budget = torch.cat([obs, torch.tensor([budget], dtype=torch.float32, device=self.device)])
        logits = self.actor(obs_with_budget.unsqueeze(0))  # [1, action_dim]
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.multinomial(probs, num_samples=1).item()
        return action

    def get_action_prob(self, obs, budget=None):
        """
        Get action probability distribution for offline evaluation (e.g. FQE).
        Same as act: concat(obs, budget) -> actor -> softmax.

        Args:
            obs: [state_dim] or [batch, state_dim], numpy or tensor
            budget: scalar or [batch] or [batch, 1] or None. Filled with 0 if None.

        Returns:
            probs: [batch, action_dim] action probability distribution (tensor)
        """
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(dtype=torch.float32, device=self.device)

        if budget is not None:
            if not torch.is_tensor(budget):
                budget = torch.tensor(budget, dtype=torch.float32, device=self.device)
        else:
            batch_size = obs.shape[0] if obs.dim() > 1 else 1
            budget = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if budget.dim() == 0:
            budget = budget.unsqueeze(0)
        if budget.dim() == 1:
            budget = budget.unsqueeze(1)

        obs_with_budget = torch.cat([obs, budget], dim=-1)
        logits = self.actor(obs_with_budget)
        probs = torch.softmax(logits, dim=-1)
        return probs

class MultiTaskBCTrainer:
    """
    Multi-Task Behavior Cloning Trainer for Discrete Actions with Dynamic Budget.
    No environment interface; for offline dataset training only.
    """
    def __init__(
            self,
            model: MultiTaskBC,
            logger: Logger = Logger(),
            actor_lr: float = 1e-4):

        self.model = model
        self.logger = logger
        self.model.setup_optimizers(actor_lr)

    def train_one_step(self, observations, budgets, actions):
        """
        :param observations: torch.Tensor [batch, state_dim]
        :param budgets: torch.Tensor [batch]  # budget per sample
        :param actions: torch.Tensor [batch]  # LongTensor, action class indices
        """
        loss_actor, stats_actor = self.model.actor_loss(observations, budgets, actions)
        self.logger.store(**stats_actor)

    # No evaluation/rollout or environment interaction

