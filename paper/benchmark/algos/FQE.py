# algos/FQE.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class FQE(nn.Module):
    def __init__(self, state_dim, action_dim, device='cpu', hidden_sizes=[256, 256], lr=5e-5, verbose=False):
        super(FQE, self).__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Q-Network for evaluation
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.q_net = nn.Sequential(*layers).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=lr)
        self.verbose = verbose  # added

    def train_step(self, batch, target_policy_model, gamma=0.99, is_continuous_action=False):
        """
        One FQE training step.
        """
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # Ensure rewards and dones have correct shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # Get real model's act (bypass ModelWrapper)
        real_model = target_policy_model.model if hasattr(target_policy_model, 'model') else target_policy_model
        import inspect
        act_signature = inspect.signature(real_model.act)
        act_params = list(act_signature.parameters.keys())
        needs_budget_param = 'budget' in act_params and 'budgets' in batch and batch['budgets'] is not None
        
        # Detect model type
        is_marl_model = 'agent_mask' in act_params
        is_cdt_model = 'states' in act_params and 'actions' in act_params and 'returns_to_go' in act_params
        
        # If budget present, handle per model
        if 'budgets' in batch and batch['budgets'] is not None:
            budgets = batch['budgets']
            next_budgets = batch.get('next_budgets', None)
            
            if budgets.dim() == 1: 
                budgets = budgets.unsqueeze(-1)
            if next_budgets is not None and next_budgets.dim() == 1:
                next_budgets = next_budgets.unsqueeze(-1)
            
            if not needs_budget_param:
                obs = th.cat([obs, budgets], dim=-1)
                if next_budgets is not None:
                    next_obs = th.cat([next_obs, next_budgets], dim=-1)
        else:
            budgets = None
            next_budgets = None

        # 1. Compute target Q
        with th.no_grad():
            if hasattr(target_policy_model, 'act_prob'): 
                # Discrete: expected V = sum(prob * Q)
                next_probs = self._get_policy_probs(target_policy_model, next_obs, next_budgets, needs_budget_param)
                next_qs = self.target_q_net(next_obs if not needs_budget_param else th.cat([next_obs, next_budgets], dim=-1))
                next_v = (next_probs * next_qs).sum(dim=-1, keepdim=True)
            elif is_cdt_model:
                raise RuntimeError(
                    "CDT is not supported for FQE evaluation. CDT requires sequence info (states, actions, returns_to_go, costs_to_go, time_steps), "
                    "which FQE transition data does not provide. Use another evaluation method."
                )
            else:
                # Call model as in evaluate_unified_v4.py
                next_action = None
                last_exception = None
                exceptions = []
                
                # Strategy 1: MARL interface with positional args
                if is_marl_model:
                    try:
                        batch_size = next_obs.shape[0]
                        obs_dim = next_obs.shape[1]
                        
                        # Build MARL format: next_obs [B, obs_dim] -> [B, 1, obs_dim] (one agent per transition)
                        if next_obs.dim() == 2:
                            next_obs_marl = next_obs.unsqueeze(1)  # [B, 1, obs_dim]
                        else:
                            next_obs_marl = next_obs
                        
                        # Convert to numpy (model may expect numpy)
                        next_obs_marl_np = next_obs_marl.cpu().numpy() if th.is_tensor(next_obs_marl) else next_obs_marl
                        
                        # Create agent_mask: [B, 1] (one agent, all active)
                        agent_mask_np = np.ones((batch_size, 1), dtype=np.float32)
                        
                        # Call with positional args (DiscreteBCMultiAgent: act(obs, agent_mask, legal_actions, training))
                        next_action = target_policy_model.act(
                            next_obs_marl_np,      # obs [B, 1, obs_dim]
                            agent_mask_np,         # agent_mask [B, 1]
                            None,                   # legal_actions
                            False                   # training
                        )
                        
                        # Handle return value
                        if isinstance(next_action, (tuple, list)):
                            next_action = next_action[0]
                        
                        next_action = np.asarray(next_action)
                        # MARL returns [B, N]; we need [B]. With N=1: [B, 1] -> [B]
                        if next_action.ndim == 2:
                            if next_action.shape[1] == 1:
                                next_action = next_action[:, 0]  # [B, 1] -> [B]
                            else:
                                if next_action.shape[0] == 1:
                                    next_action = next_action[0]  # [1, B] -> [B]
                                else:
                                    raise ValueError(f"MARL action shape invalid: {next_action.shape}, expected [B, 1] or [1, B]")
                        
                        # Convert to tensor
                        next_action = th.tensor(next_action, device=self.device).long()
                        
                    except Exception as e:
                        exceptions.append(("MARL (positional)", str(e)))
                        last_exception = e
                
                # Strategy 2: OSRL with budget (positional)
                if next_action is None and needs_budget_param and next_budgets is not None and not is_marl_model:
                    try:
                        if next_budgets.dim() == 2 and next_budgets.shape[1] == 1:
                            budget_for_act = next_budgets.squeeze(-1)
                        else:
                            budget_for_act = next_budgets
                        
                        # Convert to numpy if needed
                        obs_np = next_obs.cpu().numpy() if th.is_tensor(next_obs) else next_obs
                        budget_np = budget_for_act.cpu().numpy() if th.is_tensor(budget_for_act) else budget_for_act
                        
                        # Call as in evaluate_unified_v4.py
                        if obs_np.ndim == 2:
                            obs_in = obs_np  # [B, obs_dim]
                        else:
                            obs_in = obs_np.reshape(1, -1)
                        
                        # Handle scalar budget
                        if budget_np.ndim == 0:
                            budget_in = float(budget_np)
                        elif budget_np.ndim == 1 and len(budget_np) == 1:
                            budget_in = float(budget_np[0])
                        else:
                            budget_in = budget_np
                        
                        # For batch: call per sample or batch
                        if obs_in.shape[0] == 1:
                            next_action = target_policy_model.act(obs_in[0], budget_in)
                        else:
                            # Batch: loop
                            actions_list = []
                            for i in range(obs_in.shape[0]):
                                single_obs = obs_in[i]
                                single_budget = budget_in[i] if hasattr(budget_in, '__len__') and len(budget_in) > i else budget_in
                                action = target_policy_model.act(single_obs, single_budget)
                                actions_list.append(action)
                            next_action = np.array(actions_list)
                        
                        # Convert to tensor
                        if not th.is_tensor(next_action):
                            next_action = th.tensor(next_action, device=self.device)
                        
                    except Exception as e:
                        exceptions.append(("OSRL with budget", str(e)))
                        last_exception = e
                
                # Strategy 3: OSRL without budget (positional)
                if next_action is None and not is_marl_model:
                    try:
                        obs_np = next_obs.cpu().numpy() if th.is_tensor(next_obs) else next_obs
                        
                        if obs_np.ndim == 2:
                            obs_in = obs_np  # [B, obs_dim]
                        else:
                            obs_in = obs_np.reshape(1, -1)
                        
                        # For batch: call per sample or batch
                        if obs_in.shape[0] == 1:
                            next_action = target_policy_model.act(obs_in[0])
                        else:
                            # Batch: loop
                            actions_list = []
                            for i in range(obs_in.shape[0]):
                                action = target_policy_model.act(obs_in[i])
                                actions_list.append(action)
                            next_action = np.array(actions_list)
                        
                        # Convert to tensor
                        if not th.is_tensor(next_action):
                            next_action = th.tensor(next_action, device=self.device)
                        
                    except Exception as e:
                        exceptions.append(("OSRL without budget", str(e)))
                        last_exception = e
                
                # If all attempts failed, raise
                if next_action is None:
                    error_msg = (
                        f"Failed to call policy model act.\n"
                        f"Model type: {type(real_model)}\n"
                        f"act signature: {act_signature}\n"
                        f"Detected: is_marl={is_marl_model}, is_cdt={is_cdt_model}, "
                        f"needs_budget={needs_budget_param}\n"
                        f"Tried interfaces:\n"
                    )
                    for interface_name, error in exceptions:
                        error_msg += f"  - {interface_name}: {error}\n"
                    error_msg += f"Last exception: {last_exception}"
                    raise RuntimeError(error_msg) from last_exception
                
                # Ensure next_action has correct shape and type
                if not th.is_tensor(next_action):
                    next_action = th.tensor(next_action, device=self.device)
                
                if isinstance(next_action, np.ndarray):
                    next_action = th.from_numpy(next_action).to(self.device)
                
                if next_action.dtype != th.long:
                    next_action = next_action.long()
                
                if next_action.dim() == 0:
                    next_action = next_action.unsqueeze(0)
                elif next_action.dim() > 1:
                    next_action = next_action.squeeze()
                    if next_action.dim() == 0:
                        next_action = next_action.unsqueeze(0)
                
                batch_size = next_obs.shape[0]
                if next_action.shape[0] != batch_size:
                    if next_action.numel() == batch_size:
                        next_action = next_action.reshape(batch_size)
                    else:
                        raise RuntimeError(
                            f"Action shape mismatch: next_action.shape={next_action.shape}, "
                            f"batch_size={batch_size}"
                        )
                
                # Compute next_v
                fqe_next_obs = next_obs if not needs_budget_param else th.cat([next_obs, next_budgets], dim=-1)
                next_qs = self.target_q_net(fqe_next_obs)
                
                if next_action.dim() == 1:
                    next_action = next_action.unsqueeze(-1)
                
                next_action = next_action.clamp(0, self.action_dim - 1)
                next_v = next_qs.gather(1, next_action)

            target_q = rewards + gamma * (1 - dones) * next_v

        # 2. Compute current Q
        fqe_obs = obs if not needs_budget_param else th.cat([obs, budgets], dim=-1)
        current_qs = self.q_net(fqe_obs)
        if actions.dim() == 1: 
            actions = actions.unsqueeze(-1)
        
        actions = actions.long().clamp(0, self.action_dim - 1)
        current_q = current_qs.gather(1, actions)

        # 3. Update
        if current_q.shape != target_q.shape:
            if target_q.dim() == 2 and target_q.shape[1] != 1:
                target_q = target_q[:, 0:1]
            elif target_q.dim() == 1:
                target_q = target_q.unsqueeze(-1)
            if current_q.dim() == 2 and current_q.shape[1] != 1:
                current_q = current_q[:, 0:1]
            elif current_q.dim() == 1:
                current_q = current_q.unsqueeze(-1)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self, tau=0.001):
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def _get_policy_probs(self, model, obs, budgets=None, needs_budget_param=False):
        """
        Helper: get action probabilities from different model types.
        """
        # Get real model (bypass ModelWrapper)
        real_model = model.model if hasattr(model, 'model') else model
        
        import inspect
        real_model_class = type(real_model).__name__
        is_marl_model = 'MultiAgent' in real_model_class or 'MARL' in real_model_class
        
        # Check if model was trained with budget in obs
        model_has_budget_in_obs = getattr(real_model, 'has_budget_in_obs', False)
        
        # Ensure obs is tensor
        if not th.is_tensor(obs):   
            obs = th.tensor(obs, dtype=th.float32, device=self.device)
        
        # For MARL: if expected obs_dim > current obs dim, concat budget
        if is_marl_model and hasattr(real_model, 'obs_dim'):
            expected_obs_dim = real_model.obs_dim
            
            # Current obs dim
            if obs.dim() == 2:
                current_obs_dim = obs.shape[1]
            elif obs.dim() == 1:
                current_obs_dim = obs.shape[0]
                obs = obs.unsqueeze(0)  # [obs_dim] -> [1, obs_dim]
            else:
                current_obs_dim = obs.shape[-1]
            
            # If obs dim < expected, concat budget
            if current_obs_dim < expected_obs_dim:
                if budgets is not None:
                    if not th.is_tensor(budgets):
                        budgets = th.tensor(budgets, dtype=th.float32, device=self.device)
                    
                    # Handle budget dims
                    if budgets.dim() == 0:
                        budget_in = budgets.unsqueeze(0).unsqueeze(-1)
                    elif budgets.dim() == 1:
                        budget_in = budgets.unsqueeze(-1)  # [B] -> [B, 1]
                    elif budgets.dim() == 2:
                        if budgets.shape[1] == 1:
                            budget_in = budgets  # [B, 1]
                        else:
                            budget_in = budgets[:, 0:1]  # [B, N] -> [B, 1] take first
                    else:
                        budget_in = budgets
                    
                    # Ensure batch dims match
                    if obs.shape[0] != budget_in.shape[0]:
                        if budget_in.shape[0] == 1:
                            budget_in = budget_in.expand(obs.shape[0], -1)
                        elif obs.shape[0] == 1:
                            obs = obs.expand(budget_in.shape[0], -1)
                        else:
                            raise ValueError(f"obs and budgets batch dim mismatch: obs={obs.shape[0]}, budgets={budget_in.shape[0]}")
                else:
                    batch_size = obs.shape[0]
                    budget_in = th.zeros((batch_size, 1), dtype=th.float32, device=self.device)
                
                # Concat budget to obs
                obs = th.cat([obs, budget_in], dim=-1)
        
        # Method 1: call get_action_prob
        if hasattr(real_model, 'get_action_prob'):
            try:
                sig = inspect.signature(real_model.get_action_prob)
                params = list(sig.parameters.keys())
                
                if 'agent_mask' in params:
                    # MARL: need agent_mask; obs should be [B, obs_dim(+budget)]
                    if obs.dim() == 2:
                        obs_marl = obs.unsqueeze(1)  # [B, obs_dim] -> [B, 1, obs_dim]
                    elif obs.dim() == 1:
                        obs_marl = obs.unsqueeze(0).unsqueeze(0)
                    else:
                        obs_marl = obs
                    
                    # Convert to numpy (get_action_prob may expect numpy)
                    obs_marl_np = obs_marl.cpu().numpy()
                    
                    batch_size = obs_marl_np.shape[0] if obs_marl_np.ndim >= 2 else 1
                    n_agents = obs_marl_np.shape[1] if obs_marl_np.ndim >= 3 else 1
                    
                    # Create agent_mask
                    agent_mask_np = np.ones((batch_size, n_agents), dtype=np.float32)
                    
                    # Call get_action_prob
                    probs = real_model.get_action_prob(
                        obs_marl_np, 
                        agent_mask=agent_mask_np, 
                        legal_actions=None
                    )
                    
                    # Convert to tensor if needed
                    if not th.is_tensor(probs):
                        probs = th.tensor(probs, dtype=th.float32, device=self.device)
                    
                    # If return is [B, N, action_dim], take first agent
                    if probs.dim() == 3:
                        if probs.shape[1] == 1:
                            probs = probs[:, 0, :]  # [B, 1, action_dim] -> [B, action_dim]
                        else:
                            probs = probs[:, 0, :]  # [B, N, action_dim] -> [B, action_dim]
                    
                    return probs
                    
                elif 'budget' in params:
                    # OSRL: budget passed as arg; obs should not include budget
                    if th.is_tensor(obs):
                        obs_np = obs.cpu().numpy()
                    else:
                        obs_np = obs
                    
                    if budgets is not None:
                        if not th.is_tensor(budgets):
                            budgets = th.tensor(budgets, dtype=th.float32, device=self.device)
                        if budgets.dim() == 2 and budgets.shape[1] == 1:
                            budget_in = budgets.squeeze(-1)
                        elif budgets.dim() == 1:
                            budget_in = budgets
                        else:
                            budget_in = budgets
                        budget_in_np = budget_in.cpu().numpy() if th.is_tensor(budget_in) else budget_in
                    else:
                        batch_size = obs_np.shape[0] if obs_np.ndim >= 2 else 1
                        budget_in_np = np.zeros(batch_size, dtype=np.float32)
                    
                    probs = real_model.get_action_prob(obs_np, budget_in_np)
                    if not th.is_tensor(probs):
                        probs = th.tensor(probs, dtype=th.float32, device=self.device)
                    return probs
                    
                else:
                    # Only obs needed; convert to numpy if needed
                    if th.is_tensor(obs):
                        obs_np = obs.cpu().numpy()
                    else:
                        obs_np = obs
                    
                    probs = real_model.get_action_prob(obs_np)
                    if not th.is_tensor(probs):
                        probs = th.tensor(probs, dtype=th.float32, device=self.device)
                    return probs
                    
            except Exception as e:
                print(f"Warning: get_action_prob failed: {e}, trying other methods")
                import traceback
                traceback.print_exc()
        
        # Last fallback: uniform distribution
        print(f"Warning: Model {type(real_model)} does not support get_action_prob, using uniform distribution")
        if obs.dim() == 1:
            batch_size = 1
        else:
            batch_size = obs.shape[0]
        return th.ones((batch_size, self.action_dim), device=self.device) / self.action_dim