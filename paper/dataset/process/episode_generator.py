import pandas as pd
import numpy as np
import torch as th
import json
import pickle
import os
from episode_buffer import EpisodeBatch, ReplayBuffer

# ==== Config ====
CONFIG = {
    "train_ratio": 0.8,
    "save_global_state": True,
    "random_seed": 42,
    "enable_sliding_window": True,
    "window_size": 15,
    "window_stride": 5,
    "min_window_size": 10,
    "train_buffer_file": "train_buffer.pt",
    "test_buffer_file": "test_buffer.pt",
    "train_env_file": "train_env_info.json",
    "test_env_file": "test_env_info.json",
    "train_episodes_file": "episodes_data_train.pkl",
    "test_episodes_file": "episodes_data_test.pkl",
}


class NormalizationParams:
    """
    Store normalization params (mean, std) and support normalize/denormalize.
    """
    def __init__(self):
        self.params = {}

    def update(self, name, mean, std):
        self.params[name] = {"mean": mean, "std": std}

    def normalize(self, name, x):
        mean = self.params[name]["mean"]
        std = self.params[name]["std"]
        # Support array x with broadcasting
        return (x - mean) / (std + 1e-8)

    def denormalize(self, name, x):
        mean = self.params[name]["mean"]
        std = self.params[name]["std"]
        return x * (std + 1e-8) + mean

    def get(self, name):
        return self.params[name]

    def to_dict(self):
        # Convert to plain Python for json/pickle
        return {k: {"mean": float(np.array(v["mean"]).flatten()[0]), "std": float(np.array(v["std"]).flatten()[0])}
                if np.isscalar(v["mean"]) or np.array(v["mean"]).ndim == 0
                else {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                for k,v in self.params.items()}


class EpisodeGenerator:
    """Episode generator with global normalization support."""

    GLOBAL_NORM_FEATURES = [
        # (episode_data key, shape, axis)
        ("policy_obs", "3d", None),  # (T, N, F)
        ("budget_obs", "2d", None),  # (T, N)
        ("budgets", "2d", None),
        ("log_budgets", "2d", None),
        ("raw_costs", "2d", None),
        ("bridge_rewards", "2d", None),
        ("budget_rewards", "2d", None),
        ("contributions", "2d", None),
        ("next_contributions", "2d", None),
    ]

    def __init__(self, regions_file, connectivity_file, region_data_dir):
        self.regions_file = regions_file
        self.connectivity_file = connectivity_file
        self.region_data_dir = region_data_dir

        # Load region info
        with open(regions_file, 'r') as f:
            self.regions = json.load(f)

        with open(connectivity_file, 'rb') as f:
            self.connectivity_data = pickle.load(f)

        # Action mapping
        self.action_map = {"no_action": 0, "minor_repair": 1, "major_repair": 2, "replacement": 3}

        # State features for policy agent observation
        self.policy_obs_features = [
            'STRUCTURAL_EVAL_067', 'bridge_age', 'ADT_029',
            'STRUCTURE_LEN_MT_049', 'importance'
        ]

        print(f"Loaded {len(self.regions)} regions")

        # Sliding window params
        self.enable_sliding_window = CONFIG.get("enable_sliding_window", False)
        self.window_size = CONFIG.get("window_size", 15)
        self.window_stride = CONFIG.get("window_stride", 5)
        self.min_window_size = CONFIG.get("min_window_size", 10)

        if self.enable_sliding_window:
            print(f"Sliding window enabled: size={self.window_size}, stride={self.window_stride}")

    def create_episodes(self, episode_limit=32):
        episodes = []

        for region in self.regions:
            region_id = region['region_id']
            bridge_ids = region['bridge_ids']
            years = region['years']

            print(f"Creating episode for region {region_id}, bridges: {len(bridge_ids)}")
            region_df = pd.read_csv(f'{self.region_data_dir}/region_{region_id}_data.csv')
            with open(f'{self.region_data_dir}/region_{region_id}_budget_params.json', 'r') as f:
                budget_params = json.load(f)

            if len(region_df) == 0:
                continue

            available_years = sorted(region_df['year'].unique())

            if self.enable_sliding_window:
                region_episodes = self._create_sliding_window_episodes(
                    region, region_df, available_years, budget_params, episode_limit
                )
                episodes.extend(region_episodes)
            else:
                selected_years = available_years[:episode_limit]
                region_df_filtered = region_df[region_df['year'].isin(selected_years)]
                episode = self._create_single_episode(region, region_df_filtered, selected_years, budget_params)
                if episode is not None:
                    episodes.append(episode)

        print(f"Created {len(episodes)} episodes")
        return episodes

    def _create_sliding_window_episodes(self, region, region_df, available_years, budget_params, max_episode_limit):
        episodes = []
        region_id = region['region_id']
        if len(available_years) < self.min_window_size:
            print(f"Region {region_id}: insufficient years, skipping sliding window")
            return episodes

        window_count = 0
        for start_idx in range(0, len(available_years) - self.min_window_size + 1, self.window_stride):
            end_idx = min(start_idx + self.window_size, len(available_years))
            if end_idx - start_idx < self.min_window_size:
                if start_idx == 0:
                    end_idx = len(available_years)
                else:
                    break
            window_years = available_years[start_idx:end_idx]
            if len(window_years) > max_episode_limit:
                window_years = window_years[:max_episode_limit]
            window_df = region_df[region_df['year'].isin(window_years)]

            if len(window_df) > 0:
                episode = self._create_single_episode(
                    region, window_df, window_years, budget_params,
                    window_info={'window_id': window_count, 'start_year': window_years[0], 'end_year': window_years[-1]}
                )
                if episode is not None:
                    episodes.append(episode)
                    window_count += 1
            if len(episodes) >= 10:
                break
        print(f"Region {region_id}: created {len(episodes)} sliding-window episodes")
        return episodes

    def _create_single_episode(self, region, region_df, years, budget_params, window_info=None):
        region_id = region['region_id']
        bridge_ids = region['bridge_ids']
        n_bridges = len(bridge_ids)
        n_years = len(years)

        bridge_to_idx = {bridge_id: idx for idx, bridge_id in enumerate(bridge_ids)}

        episode_data = {
            'region_id': region_id,
            'bridge_ids': bridge_ids,
            'years': years,
            'budget_params': budget_params,
            'policy_obs': np.zeros((n_years, n_bridges, len(self.policy_obs_features))),
            'budget_obs': np.zeros((n_years, n_bridges)),
            'budget_levels': np.zeros((n_years, n_bridges), dtype=int),
            'budgets': np.zeros((n_years, n_bridges)),
            'log_budgets': np.zeros((n_years, n_bridges)),
            'raw_costs': np.zeros((n_years, n_bridges)),
            'actions': np.zeros((n_years, n_bridges), dtype=int),
            'bridge_rewards': np.zeros((n_years, n_bridges)),
            'budget_rewards': np.zeros((n_years, n_bridges)),
            'health_states': np.zeros((n_years, n_bridges)),
            'importance': np.zeros((n_years, n_bridges)),
            'contributions': np.zeros((n_years, n_bridges)),
            'next_contributions': np.zeros((n_years, n_bridges)),
            'filled': np.zeros((n_years, n_bridges)),
            'connectivity_matrix': self.connectivity_data[region_id]['matrix'],
        }

        if window_info is not None:
            episode_data['window_info'] = window_info
            episode_data['episode_type'] = 'sliding_window'
        else:
            episode_data['episode_type'] = 'full_sequence'

        if CONFIG["save_global_state"]:
            episode_data['global_info'] = np.zeros((n_years, 10))

        for t, year in enumerate(years):
            year_data = region_df[region_df['year'] == year]
            if CONFIG["save_global_state"]:
                year_stats = self._calculate_year_stats(year_data, n_bridges, budget_params)
                episode_data['global_info'][t] = year_stats

            for _, row in year_data.iterrows():
                bridge_id = row['STRUCTURE_NUMBER_008']
                if bridge_id in bridge_to_idx:
                    idx = bridge_to_idx[bridge_id]
                    for i, feature in enumerate(self.policy_obs_features):
                        episode_data['policy_obs'][t, idx, i] = row[feature]
                    episode_data['budget_obs'][t, idx] = row.get('budget_obs', 0)
                    episode_data['budget_levels'][t, idx] = row['budget_level']
                    episode_data['budgets'][t, idx] = row['TOTAL_IMP_COST_096']
                    episode_data['log_budgets'][t, idx] = row['log_cost']
                    episode_data['raw_costs'][t, idx] = row['TOTAL_IMP_COST_096']
                    episode_data['actions'][t, idx] = self.action_map.get(row['action'], 0)
                    episode_data['bridge_rewards'][t, idx] = row['bridge_reward']
                    episode_data['budget_rewards'][t, idx] = row['budget_reward']
                    episode_data['health_states'][t, idx] = row['STRUCTURAL_EVAL_067']
                    episode_data['importance'][t, idx] = row['importance']
                    episode_data['contributions'][t, idx] = row['contribution']
                    episode_data['next_contributions'][t, idx] = row['next_contribution']
                    episode_data['filled'][t, idx] = 1

        # No normalization here; apply global normalization after all episodes are collected
        return episode_data

    def _calculate_year_stats(self, year_data, total_bridges, budget_params):
        stats = np.zeros(10)
        if len(year_data) > 0:
            health_counts = year_data['STRUCTURAL_EVAL_067'].value_counts()
            total_count = len(year_data)
            for rating in range(10):
                stats[rating] = health_counts.get(rating, 0) / total_count if total_count > 0 else 0
            poor_count = sum(health_counts.get(i, 0) for i in range(0, 4))
            fair_count = sum(health_counts.get(i, 0) for i in range(4, 6))
            good_count = sum(health_counts.get(i, 0) for i in range(6, 8))
            excellent_count = sum(health_counts.get(i, 0) for i in range(8, 10))
            stats[0] = good_count / total_count if total_count > 0 else 0
            stats[1] = fair_count / total_count if total_count > 0 else 0
            stats[2] = poor_count / total_count if total_count > 0 else 0
            stats[3] = excellent_count / total_count if total_count > 0 else 0

            total_contribution = year_data['contribution'].sum()
            avg_contribution = year_data['contribution'].mean()
            max_possible_contribution = total_bridges * 9
            stats[4] = min(1.0, total_contribution / max_possible_contribution) if max_possible_contribution > 0 else 0
            stats[5] = min(1.0, avg_contribution / 9) if avg_contribution > 0 else 0
            total_cost = year_data['TOTAL_IMP_COST_096'].sum()
            btotal = budget_params['btotal']
            stats[6] = 1.0
            stats[7] = min(1.0, total_cost / btotal) if btotal > 0 else 0
            stats[8] = len(year_data) / total_bridges
            stats[9] = min(1.0, np.log1p(len(year_data)) / np.log1p(total_bridges)) if total_bridges > 1 else 0
        return stats

    @staticmethod
    def _collect_all_feature_values(episodes, policy_obs_dim):
        """
        Collect raw values of features to normalize from all episodes for global stats.
        Returns dict: key -> list(np.ndarray)
        """
        values = {
            "policy_obs": [],
            "budget_obs": [],
            "budgets": [],
            "log_budgets": [],
            "raw_costs": [],
            "bridge_rewards": [],
            "budget_rewards": [],
            "contributions": [],
            "next_contributions": [],
        }
        for ep in episodes:
            mask = ep["filled"] == 1  # Only valid entries
            # policy_obs: (T,N,F)
            policy_obs = ep["policy_obs"][mask]
            if policy_obs.ndim == 1:
                policy_obs = policy_obs.reshape(-1, policy_obs_dim)
            values["policy_obs"].append(policy_obs)
            # Others: (T,N)
            for k in values.keys():
                if k != "policy_obs":
                    v = ep[k][mask]
                    values[k].append(v.reshape(-1))
        # Concatenate
        for k in values:
            if len(values[k]) > 0:
                values[k] = np.concatenate(values[k], axis=0)
            else:
                values[k] = np.array([])
        return values

    @staticmethod
    def compute_global_norm_params(episodes, policy_obs_dim):
        """
        Compute global mean/std for all features to normalize.
        """
        norm = NormalizationParams()
        values = EpisodeGenerator._collect_all_feature_values(episodes, policy_obs_dim)
        # policy_obs is 2d: (n, F)
        if values["policy_obs"].ndim == 1:
            values["policy_obs"] = values["policy_obs"].reshape(-1, policy_obs_dim)
        if values["policy_obs"].size > 0:
            mean = np.mean(values["policy_obs"], axis=0)
            std = np.std(values["policy_obs"], axis=0)
            norm.update("policy_obs", mean, std)
        for k in [
            "budget_obs", "budgets", "log_budgets", #"bridge_rewards",
            "budget_rewards", "contributions", "next_contributions"   # Skip "raw_costs" for now
        ]:
            v = values[k]
            if v.size > 0:
                mean = np.mean(v)
                std = np.std(v)
                norm.update(k, mean, std)
        return norm

    @staticmethod
    def apply_global_normalization(episodes, norm_params, policy_obs_dim):
        """
        Apply global normalization to features in all episodes (in-place).
        """
        for ep in episodes:
            mask = ep["filled"] == 1
            # policy_obs
            obs = ep["policy_obs"]
            if obs.size > 0:
                obs_flat = obs[mask]
                if obs_flat.ndim == 1:
                    obs_flat = obs_flat.reshape(-1, policy_obs_dim)
                mean = norm_params.get("policy_obs")["mean"]
                std = norm_params.get("policy_obs")["std"]
                if np.any(std == 0): std = std + 1e-8
                normed = (obs_flat - mean) / std
                obs[mask] = normed
                ep["policy_obs"] = obs
            # Other single-channel
            for k in [
                "budget_obs", "budgets", "log_budgets",  #"bridge_rewards",#"raw_costs",
                "budget_rewards", "contributions", "next_contributions"
            ]:
                v = ep[k]
                v_flat = v[mask]
                mean = norm_params.get(k)["mean"]
                std = norm_params.get(k)["std"]
                if std == 0: std = std + 1e-8
                v[mask] = (v_flat - mean) / std
                ep[k] = v
        return episodes

    def create_replay_buffer(self, episodes, max_episodes=None, save_global_state=True):
        if max_episodes:
            episodes = episodes[:max_episodes]
        if not episodes:
            return None, None
        max_bridges = max(len(ep['bridge_ids']) for ep in episodes)
        policy_obs_dim = len(self.policy_obs_features)
        max_episode_length = max(len(ep['years']) for ep in episodes)
        print(f"Creating ReplayBuffer: max_bridges={max_bridges}, max_episode_length={max_episode_length}")

        scheme = {
            "obs": {"vshape": (policy_obs_dim,), "group": "agents", "dtype": th.float32},
            "budget_obs": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "budget_level": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "budget": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "log_budget": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "raw_cost": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "reward": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "budget_reward": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "health_state": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "importance": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "contribution": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "next_contribution": {"vshape": (1,), "group": "agents", "dtype": th.float32},
            "connectivity": {"vshape": (max_bridges, max_bridges), "dtype": th.float32, "episode_const": True},
            "btotal": {"vshape": (1,), "dtype": th.float32, "episode_const": True},
            "budget_thresholds": {"vshape": (8,), "dtype": th.float32, "episode_const": True},
            "n_bridges_actual": {"vshape": (1,), "dtype": th.long, "episode_const": True},
        }
        if save_global_state:
            scheme["global_info"] = {"vshape": (10,), "dtype": th.float32}
            scheme["state"] = {"vshape": (max_bridges * policy_obs_dim,), "dtype": th.float32}

        groups = {"agents": max_bridges}
        buffer = ReplayBuffer(scheme, groups, len(episodes), max_episode_length, device="cpu")

        for ep_idx, episode in enumerate(episodes):
            n_bridges = len(episode['bridge_ids'])
            episode_length = len(episode['years'])
            budget_params = episode['budget_params']
            ep_batch = EpisodeBatch(scheme, groups, 1, max_episode_length, device="cpu")

            connectivity_padded = np.zeros((max_bridges, max_bridges))
            connectivity_padded[:n_bridges, :n_bridges] = episode['connectivity_matrix']
            ep_batch.data.episode_data["connectivity"][0] = th.from_numpy(connectivity_padded).float()
            ep_batch.data.episode_data["btotal"][0] = th.tensor(budget_params['log_btotal']).float()
            thresholds_padded = np.zeros(8)
            log_thresholds = budget_params['log_budget_thresholds']
            thresholds_padded[:len(log_thresholds)] = log_thresholds
            ep_batch.data.episode_data["budget_thresholds"][0] = th.tensor(thresholds_padded).float()
            ep_batch.data.episode_data["n_bridges_actual"][0] = th.tensor(n_bridges).long()

            for t in range(max_episode_length):
                if t < episode_length:
                    obs_padded = np.zeros((max_bridges, policy_obs_dim))
                    obs_padded[:n_bridges] = episode['policy_obs'][t]
                    budget_obs_padded = np.zeros((max_bridges, 1))
                    budget_obs_padded[:n_bridges, 0] = episode['budget_obs'][t]

                    def pad_feature_fixed(data, target_shape):
                        padded = np.zeros(target_shape)
                        if data.ndim == 1:
                            if target_shape[1] == 1:
                                padded[:len(data), 0] = data
                            else:
                                padded[:len(data)] = data
                        else:
                            padded[:data.shape[0], :data.shape[1]] = data
                        return padded

                    transition_data = {
                        "obs": obs_padded,
                        "budget_obs": budget_obs_padded,
                        "budget_level": pad_feature_fixed(episode['budget_levels'][t], (max_bridges, 1)),
                        "budget": pad_feature_fixed(episode['budgets'][t], (max_bridges, 1)),
                        "log_budget": pad_feature_fixed(episode['log_budgets'][t], (max_bridges, 1)),
                        "raw_cost": pad_feature_fixed(episode['raw_costs'][t], (max_bridges, 1)),
                        "actions": pad_feature_fixed(episode['actions'][t], (max_bridges, 1)),
                        "reward": pad_feature_fixed(episode['bridge_rewards'][t], (max_bridges, 1)),
                        "budget_reward": pad_feature_fixed(episode['budget_rewards'][t], (max_bridges, 1)),
                        "health_state": pad_feature_fixed(episode['health_states'][t], (max_bridges, 1)),
                        "importance": pad_feature_fixed(episode['importance'][t], (max_bridges, 1)),
                        "contribution": pad_feature_fixed(episode['contributions'][t], (max_bridges, 1)),
                        "next_contribution": pad_feature_fixed(episode['next_contributions'][t], (max_bridges, 1)),
                    }
                    if save_global_state:
                        transition_data["global_info"] = episode['global_info'][t].reshape(1, -1)
                        transition_data["state"] = obs_padded.flatten().reshape(1, -1)
                else:
                    transition_data = {
                        "obs": np.zeros((max_bridges, policy_obs_dim)),
                        "budget_obs": np.zeros((max_bridges, 1)),
                        "budget_level": np.zeros((max_bridges, 1)),
                        "budget": np.zeros((max_bridges, 1)),
                        "log_budget": np.zeros((max_bridges, 1)),
                        "raw_cost": np.zeros((max_bridges, 1)),
                        "actions": np.zeros((max_bridges, 1)),
                        "reward": np.zeros((max_bridges, 1)),
                        "budget_reward": np.zeros((max_bridges, 1)),
                        "health_state": np.zeros((max_bridges, 1)),
                        "importance": np.zeros((max_bridges, 1)),
                        "contribution": np.zeros((max_bridges, 1)),
                        "next_contribution": np.zeros((max_bridges, 1)),
                    }
                    if save_global_state:
                        transition_data["global_info"] = np.zeros((1, 10))
                        transition_data["state"] = np.zeros((1, max_bridges * policy_obs_dim))
                ep_batch.update(transition_data, ts=t)
            buffer.insert_episode_batch(ep_batch)

        env_info = {
            "n_agents": max_bridges,
            "n_actions": len(self.action_map),
            "state_shape": max_bridges * policy_obs_dim if save_global_state else None,
            "obs_shape": policy_obs_dim,
            "budget_obs_shape": 1,
            "budget_levels": 8,
            "episode_limit": max_episode_length,
            "has_budget_agent": True,
            "variable_bridge_count": True,
            "max_bridges": max_bridges,
        }
        return buffer, env_info


def split_episodes(episodes, train_ratio=0.8, seed=42):
    idx = np.arange(len(episodes))
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(len(episodes) * train_ratio)
    train_idx = idx[:split]
    test_idx = idx[split:]
    train_episodes = [episodes[i] for i in train_idx]
    test_episodes = [episodes[i] for i in test_idx]
    return train_episodes, test_episodes


if __name__ == "__main__":
    generator = EpisodeGenerator(
        regions_file='paper/dataset/data/regions/regions.json',
        connectivity_file='paper/dataset/data/regions/connectivity_data.pkl',
        region_data_dir='paper/dataset/data/regions/region_data'
    )

    episode_limit = 25 if not CONFIG.get("enable_sliding_window", False) else 32
    episodes = generator.create_episodes(episode_limit=episode_limit)

    train_ratio = CONFIG["train_ratio"]
    seed = CONFIG["random_seed"]
    save_global_state = CONFIG["save_global_state"]

    # Split data
    episodes_train, episodes_test = split_episodes(episodes, train_ratio, seed)
    print(f"Train episodes: {len(episodes_train)}, Test episodes: {len(episodes_test)}")

    # Compute train set mean/std
    policy_obs_dim = len(generator.policy_obs_features)
    norm_params = generator.compute_global_norm_params(episodes_train, policy_obs_dim)

    # Save normalization params to env_info
    normalization_params_dict = norm_params.to_dict()

    # Normalize train and test sets
    episodes_train = generator.apply_global_normalization(episodes_train, norm_params, policy_obs_dim)
    episodes_test = generator.apply_global_normalization(episodes_test, norm_params, policy_obs_dim)

    buffer_train, env_info_train = generator.create_replay_buffer(episodes_train, save_global_state=save_global_state)
    buffer_test, env_info_test = generator.create_replay_buffer(episodes_test, save_global_state=save_global_state)

    # Store normalization params in env_info
    env_info_train["normalization_params"] = normalization_params_dict
    env_info_test["normalization_params"] = normalization_params_dict

    output_dir = 'paper/dataset/data/episodes'
    os.makedirs(output_dir, exist_ok=True)

    # Save buffers and env info
    th.save(buffer_train, os.path.join(output_dir, CONFIG["train_buffer_file"]))
    with open(os.path.join(output_dir, CONFIG["train_env_file"]), 'w') as f:
        json.dump(env_info_train, f, indent=2)
    with open(os.path.join(output_dir, CONFIG["train_episodes_file"]), 'wb') as f:
        pickle.dump(episodes_train, f)

    th.save(buffer_test, os.path.join(output_dir, CONFIG["test_buffer_file"]))
    with open(os.path.join(output_dir, CONFIG["test_env_file"]), 'w') as f:
        json.dump(env_info_test, f, indent=2)
    with open(os.path.join(output_dir, CONFIG["test_episodes_file"]), 'wb') as f:
        pickle.dump(episodes_test, f)

    print(f"Train/Test generation done, results saved to: {output_dir}")
    print(f"Train: {len(episodes_train)}, Test: {len(episodes_test)}")

    if CONFIG.get("enable_sliding_window", False):
        sliding_episodes = [ep for ep in episodes if ep.get('episode_type') == 'sliding_window']
        full_episodes = [ep for ep in episodes if ep.get('episode_type') == 'full_sequence']
        print(f"Sliding window episodes: {len(sliding_episodes)}")
        print(f"Full sequence episodes: {len(full_episodes)}")
        if sliding_episodes:
            window_lengths = [len(ep['years']) for ep in sliding_episodes]
            print(f"Window length: min={min(window_lengths)}, max={max(window_lengths)}, avg={sum(window_lengths)/len(window_lengths):.1f}")

    print(f"Train Buffer: {buffer_train}")
    print(f"Test Buffer: {buffer_test}")

    output_stats_path = os.path.join(output_dir, "region_budget_stats_train.txt")
    with open(output_stats_path, "w", encoding="utf-8") as f_stats:
        header = "\n=== Per-region budget stats (train set) ==="
        print(header)
        f_stats.write(header + "\n")
        for i, episode in enumerate(episodes_train):
            budget_params = episode['budget_params']
            episode_type = episode.get('episode_type', 'unknown')
            window_info = episode.get('window_info', {})
            line = (f"Episode {i} (region {episode['region_id']}, type={episode_type}): "
                    f"bridges={len(episode['bridge_ids'])}, "
                    f"timesteps={len(episode['years'])}, "
                    f"total_budget={budget_params['btotal']:,.2f}")
            if window_info:
                line += f", window={window_info.get('start_year', '')}-{window_info.get('end_year', '')}"
            print(line)
            f_stats.write(line + "\n")