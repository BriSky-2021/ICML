#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励分布分析脚本
检查训练数据集中的奖励分布是否合理
"""

import os
import sys
import numpy as np
import torch as th
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_env_info(env_info_path):
    """加载环境信息"""
    with open(env_info_path, 'r') as f:
        return json.load(f)

def load_buffer(buffer_path, device):
    """加载ReplayBuffer"""
    buffer = th.load(buffer_path)
    buffer.to(device)
    return buffer

def extract_data_from_buffer(buffer):
    """从buffer中提取数据"""
    all_obs, all_actions, episode_budgets, all_health, all_raw_cost, all_rewards, all_log_budgets = [], [], [], [], [], [], []
    actual_n_agents_list = []
    
    num_eps = buffer.episodes_in_buffer
    print(f"加载 {num_eps} 个episodes...")
    
    for i in range(num_eps):
        ep = buffer[i:i+1]
        obs = ep['obs'].cpu().numpy()
        actions = ep['actions'].cpu().numpy()
        btotal = np.expm1(ep['btotal'].cpu().numpy()) if 'btotal' in ep.data.episode_data else 100000
        log_budgets = ep['log_budget'].cpu().numpy()
        
        actual_n_agents = ep['n_bridges_actual'].cpu().numpy().squeeze()
        actual_n_agents_list.append(actual_n_agents)
        
        all_log_budgets.append(log_budgets[0])
        all_obs.append(obs[0])
        all_actions.append(actions[0])
        episode_budgets.append(btotal)
        
        if 'health_state' in ep.data.transition_data:
            health = ep['health_state'].cpu().numpy()
            all_health.append(health[0])
        
        if 'raw_cost' in ep.data.transition_data:
            raw_cost = ep['raw_cost'].cpu().numpy()
            all_raw_cost.append(raw_cost[0])
        else:
            all_raw_cost.append(np.zeros_like(actions[0]))
        
        if 'reward' in ep.data.transition_data:
            rewards = ep['reward'].cpu().numpy()
            all_rewards.append(rewards[0])
        else:
            all_rewards.append(np.zeros_like(actions[0]))
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{num_eps} episodes...")
    
    data = np.stack(all_obs)
    actions = np.stack(all_actions)
    episode_budgets = np.array(episode_budgets).squeeze()
    health = np.stack(all_health) if all_health else None
    raw_cost = np.stack(all_raw_cost) if all_raw_cost else None
    rewards = np.stack(all_rewards) if all_rewards else None
    log_budgets = np.stack(all_log_budgets) if all_log_budgets else None
    actual_n_agents = np.array(actual_n_agents_list)
    
    print(f'数据形状: {data.shape}')
    print(f'奖励形状: {rewards.shape if rewards is not None else "None"}')
    
    return data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents

def analyze_reward_distribution(rewards, raw_cost=None, health=None, save_dir="paper/benchmark/reward_analysis"):
    """分析奖励分布"""
    os.makedirs(save_dir, exist_ok=True)
    
    # ensurerewards是3D数组: [num_eps, T, n_agents]
    if rewards.ndim == 4 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
    assert rewards.ndim == 3, f"rewards维度应为3, 实际为{rewards.ndim}"
    
    num_eps, T, n_agents = rewards.shape
    
    print("\n" + "="*80)
    print("奖励分布分析")
    print("="*80)
    
    # ========== 1. 整体奖励统计 ==========
    print("\n1. 整体奖励统计:")
    all_rewards = rewards.flatten()
    print(f"   总样本数: {len(all_rewards):,}")
    print(f"   均值: {np.mean(all_rewards):.6f}")
    print(f"   标准差: {np.std(all_rewards):.6f}")
    print(f"   最小值: {np.min(all_rewards):.6f}")
    print(f"   最大值: {np.max(all_rewards):.6f}")
    print(f"   中位数: {np.median(all_rewards):.6f}")
    
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct_values = np.percentile(all_rewards, percentiles)
    print(f"\n   分位数分布:")
    for p, v in zip(percentiles, pct_values):
        print(f"     P{p:02d}: {v:.6f}")
    
    # check异常值
    q1, median, q3 = np.percentile(all_rewards, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((all_rewards < lower_bound) | (all_rewards > upper_bound))
    print(f"\n   异常值检测 (IQR方法):")
    print(f"     下界: {lower_bound:.6f}, 上界: {upper_bound:.6f}")
    print(f"     异常值数量: {outliers} ({outliers/len(all_rewards)*100:.2f}%)")
    
    # ========== 2. Episode级别奖励统计 ==========
    print("\n2. Episode级别奖励统计:")
    episode_rewards = rewards.sum(axis=(1, 2))  # [num_eps]
    print(f"   每个episode奖励总和:")
    print(f"     均值: {np.mean(episode_rewards):.6f}")
    print(f"     标准差: {np.std(episode_rewards):.6f}")
    print(f"     最小值: {np.min(episode_rewards):.6f}")
    print(f"     最大值: {np.max(episode_rewards):.6f}")
    print(f"     中位数: {np.median(episode_rewards):.6f}")
    
    # ========== 3. 时间步级别奖励统计 ==========
    print("\n3. 时间步级别奖励统计:")
    time_rewards = rewards.mean(axis=(0, 2))  # [T] - 每个时间步的平均奖励
    print(f"   每个时间步的平均奖励:")
    print(f"     均值范围: [{np.min(time_rewards):.6f}, {np.max(time_rewards):.6f}]")
    print(f"     总体均值: {np.mean(time_rewards):.6f}")
    
    # ========== 4. 智能体级别奖励统计 ==========
    print("\n4. 智能体级别奖励统计:")
    agent_rewards = rewards.mean(axis=(0, 1))  # [n_agents] - 每个智能体的平均奖励
    print(f"   每个智能体的平均奖励:")
    print(f"     均值范围: [{np.min(agent_rewards):.6f}, {np.max(agent_rewards):.6f}]")
    print(f"     总体均值: {np.mean(agent_rewards):.6f}")
    
    # ========== 5. 奖励与成本的关系 ==========
    cost_correlation = None
    if raw_cost is not None:
        print("\n5. 奖励与成本的关系:")
        if raw_cost.ndim == 4 and raw_cost.shape[-1] == 1:
            raw_cost = raw_cost.squeeze(-1)
        
        # ensure维度一致
        if raw_cost.shape != rewards.shape:
            print(f"   警告: raw_cost形状 {raw_cost.shape} 与 rewards形状 {rewards.shape} 不匹配")
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(raw_cost.shape, rewards.shape))
            raw_cost = raw_cost[:min_shape[0], :min_shape[1], :min_shape[2]]
            rewards_for_cost = rewards[:min_shape[0], :min_shape[1], :min_shape[2]]
        else:
            rewards_for_cost = rewards
        
        all_costs = raw_cost.flatten()
        all_rewards_flat = rewards_for_cost.flatten()
        
        # ensure长度相同
        min_len = min(len(all_rewards_flat), len(all_costs))
        all_rewards_flat = all_rewards_flat[:min_len]
        all_costs = all_costs[:min_len]
        
        # compute相关系数
        if len(all_rewards_flat) > 1:
            cost_correlation = np.corrcoef(all_rewards_flat, all_costs)[0, 1]
            print(f"   奖励-成本相关系数: {cost_correlation:.6f}")
        else:
            print(f"   警告: 数据不足, 无法计算相关系数")
        
        # by成本分组的奖励统计
        if len(all_costs) > 0:
            cost_bins = np.percentile(all_costs[all_costs > 0], [0, 25, 50, 75, 100]) if np.any(all_costs > 0) else [0]
            cost_bins = np.concatenate([[0], cost_bins])
            cost_bins = np.unique(cost_bins)
            
            print(f"   不同成本区间的平均奖励:")
            for i in range(len(cost_bins)-1):
                mask = (all_costs >= cost_bins[i]) & (all_costs < cost_bins[i+1])
                if np.sum(mask) > 0:
                    avg_reward = np.mean(all_rewards_flat[mask])
                    print(f"     成本 [{cost_bins[i]:.2f}, {cost_bins[i+1]:.2f}): 平均奖励 = {avg_reward:.6f}, 样本数 = {np.sum(mask):,}")
    
    # ========== 6. 奖励与健康状态的关系 ==========
    health_correlation = None
    if health is not None:
        print("\n6. 奖励与健康状态的关系:")
        # health形状: [num_eps, T+1, n_agents]
        # rewards形状: [num_eps, T, n_agents]
        
        # check维度匹配
        if health.shape[:2] != (num_eps, T+1):
            print(f"   警告: health形状 {health.shape} 与预期不匹配, 调整中...")
            # 调整health到正确的episode数量
            min_eps = min(num_eps, health.shape[0])
            health = health[:min_eps, :, :]
            rewards_for_health = rewards[:min_eps, :, :]
            num_eps_adj = min_eps
        else:
            rewards_for_health = rewards
            num_eps_adj = num_eps
        
        if health.shape[2] != rewards_for_health.shape[2]:
            print(f"   警告: health的智能体维度 {health.shape[2]} 与rewards的智能体维度 {rewards_for_health.shape[2]} 不匹配")
            # 取较小的维度
            min_n_agents = min(health.shape[2], rewards_for_health.shape[2])
            health = health[:, :, :min_n_agents]
            rewards_for_health = rewards_for_health[:, :, :min_n_agents]
            print(f"   已调整到统一维度: {min_n_agents}")
        
        # compute健康变化
        health_change = health[:, 1:, :] - health[:, :-1, :]  # [num_eps, T, n_agents]
        
        # ensure维度一致后再flatten
        if health_change.shape != rewards_for_health.shape:
            print(f"   警告: health_change形状 {health_change.shape} 与 rewards形状 {rewards_for_health.shape} 不匹配, 调整中...")
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(health_change.shape, rewards_for_health.shape))
            health_change = health_change[:min_shape[0], :min_shape[1], :min_shape[2]]
            rewards_for_health = rewards_for_health[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        health_change_flat = health_change.flatten()
        all_rewards_flat = rewards_for_health.flatten()
        
        # ensure长度相同
        min_len = min(len(all_rewards_flat), len(health_change_flat))
        all_rewards_flat = all_rewards_flat[:min_len]
        health_change_flat = health_change_flat[:min_len]
        
        # compute相关系数
        if len(all_rewards_flat) > 1:
            health_correlation = np.corrcoef(all_rewards_flat, health_change_flat)[0, 1]
            print(f"   奖励-健康变化相关系数: {health_correlation:.6f}")
        else:
            print(f"   警告: 数据不足, 无法计算相关系数")
        
        # by健康变化分组的奖励统计
        if len(health_change_flat) > 0:
            health_bins = np.percentile(health_change_flat, [0, 25, 50, 75, 100])
            health_bins = np.unique(health_bins)
            
            print(f"   不同健康变化区间的平均奖励:")
            for i in range(len(health_bins)-1):
                mask = (health_change_flat >= health_bins[i]) & (health_change_flat < health_bins[i+1])
                if np.sum(mask) > 0:
                    avg_reward = np.mean(all_rewards_flat[mask])
                    print(f"     健康变化 [{health_bins[i]:.4f}, {health_bins[i+1]:.4f}): 平均奖励 = {avg_reward:.6f}, 样本数 = {np.sum(mask):,}")
    
    # ========== 7. 生成可视化图表 ==========
    print("\n7. 生成可视化图表...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 7.1 整体奖励分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 直方图
    axes[0, 0].hist(all_rewards, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_rewards), color='r', linestyle='--', label=f'均值: {np.mean(all_rewards):.4f}')
    axes[0, 0].axvline(np.median(all_rewards), color='g', linestyle='--', label=f'中位数: {np.median(all_rewards):.4f}')
    axes[0, 0].set_xlabel('奖励值')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('整体奖励分布直方图')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode奖励分布箱线图
    axes[0, 1].boxplot(episode_rewards, vert=True)
    axes[0, 1].set_ylabel('Episode奖励总和')
    axes[0, 1].set_title('Episode级别奖励分布箱线图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 时间步平均奖励
    axes[1, 0].plot(time_rewards, marker='o', markersize=3)
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('平均奖励')
    axes[1, 0].set_title('各时间步平均奖励')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 奖励累积分布函数
    sorted_rewards = np.sort(all_rewards)
    cumsum = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    axes[1, 1].plot(sorted_rewards, cumsum)
    axes[1, 1].set_xlabel('奖励值')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('奖励累积分布函数 (CDF)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reward_distribution_{ts}.png"), dpi=300, bbox_inches='tight')
    print(f"   已保存: reward_distribution_{ts}.png")
    plt.close()
    
    # 7.2 奖励与成本/健康的关系图
    if raw_cost is not None or health is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if raw_cost is not None:
            # 采样以避免点太多
            sample_size = min(10000, len(all_rewards_flat))
            indices = np.random.choice(len(all_rewards_flat), sample_size, replace=False)
            axes[0].scatter(all_costs[indices], all_rewards_flat[indices], alpha=0.3, s=1)
            axes[0].set_xlabel('成本')
            axes[0].set_ylabel('奖励')
            axes[0].set_title(f'奖励 vs 成本 (采样{sample_size}点)')
            axes[0].grid(True, alpha=0.3)
        
        if health is not None:
            sample_size = min(10000, len(all_rewards_flat))
            indices = np.random.choice(len(all_rewards_flat), sample_size, replace=False)
            axes[1].scatter(health_change_flat[indices], all_rewards_flat[indices], alpha=0.3, s=1)
            axes[1].set_xlabel('健康变化')
            axes[1].set_ylabel('奖励')
            axes[1].set_title(f'奖励 vs 健康变化 (采样{sample_size}点)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"reward_relationships_{ts}.png"), dpi=300, bbox_inches='tight')
        print(f"   已保存: reward_relationships_{ts}.png")
        plt.close()
    
    # ========== 8. 保存统计结果 ==========
    stats = {
        "timestamp": ts,
        "num_episodes": int(num_eps),
        "num_timesteps": int(T),
        "num_agents": int(n_agents),
        "total_samples": int(len(all_rewards)),
        "overall_stats": {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "min": float(np.min(all_rewards)),
            "max": float(np.max(all_rewards)),
            "median": float(np.median(all_rewards)),
            "percentiles": {f"p{p}": float(v) for p, v in zip(percentiles, pct_values)}
        },
        "episode_stats": {
            "mean": float(np.mean(episode_rewards)),
            "std": float(np.std(episode_rewards)),
            "min": float(np.min(episode_rewards)),
            "max": float(np.max(episode_rewards)),
            "median": float(np.median(episode_rewards))
        },
        "outlier_info": {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outliers),
            "outlier_percentage": float(outliers/len(all_rewards)*100)
        }
    }
    
    if raw_cost is not None and cost_correlation is not None:
        stats["cost_reward_correlation"] = float(cost_correlation)
    
    if health is not None and health_correlation is not None:
        stats["health_reward_correlation"] = float(health_correlation)
    
    stats_path = os.path.join(save_dir, f"reward_statistics_{ts}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n   已保存统计结果: {stats_path}")
    
    # ========== 9. 奖励尺度合理性评估 ==========
    print("\n" + "="*80)
    print("奖励尺度合理性评估")
    print("="*80)
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    max_reward = np.max(all_rewards)
    min_reward = np.min(all_rewards)
    
    print(f"\n评估标准:")
    print(f"1. 奖励均值: {mean_reward:.6f}")
    if abs(mean_reward) < 1e-6:
        print("   ⚠️  警告: 奖励均值接近0, 可能不利于学习")
    elif mean_reward > 1000:
        print("   ⚠️  警告: 奖励均值过大, 可能导致数值不稳定")
    else:
        print("   ✓ 奖励均值在合理范围")
    
    print(f"\n2. 奖励标准差: {std_reward:.6f}")
    if std_reward < 1e-6:
        print("   ⚠️  警告: 奖励标准差过小, 奖励几乎无变化")
    elif std_reward > mean_reward * 10:
        print("   ⚠️  警告: 奖励方差过大, 可能影响学习稳定性")
    else:
        print("   ✓ 奖励标准差在合理范围")
    
    print(f"\n3. 奖励范围: [{min_reward:.6f}, {max_reward:.6f}]")
    range_size = max_reward - min_reward
    if range_size < 1e-6:
        print("   ⚠️  警告: 奖励范围过小, 几乎没有区分度")
    elif range_size > 1e6:
        print("   ⚠️  警告: 奖励范围过大, 可能导致数值问题")
    else:
        print("   ✓ 奖励范围在合理范围")
    
    print(f"\n4. 异常值比例: {outliers/len(all_rewards)*100:.2f}%")
    if outliers/len(all_rewards) > 0.05:
        print("   ⚠️  警告: 异常值比例较高, 建议检查奖励计算逻辑")
    else:
        print("   ✓ 异常值比例正常")
    
    print(f"\n5. 零奖励比例: {(all_rewards == 0).sum()/len(all_rewards)*100:.2f}%")
    zero_ratio = (all_rewards == 0).sum() / len(all_rewards)
    if zero_ratio > 0.5:
        print("   ⚠️  警告: 零奖励比例过高, 可能影响学习")
    else:
        print("   ✓ 零奖励比例正常")
    
    print("\n" + "="*80)
    print(f"分析完成! 结果已保存到: {save_dir}")
    print("="*80)


def main():
    """主函数"""
    # load配置
    config_path = "paper/benchmark/flows/config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    device = th.device("cpu")  # 分析时使用CPU即可
    
    # load数据
    print("="*80)
    print("加载训练数据集...")
    print("="*80)
    
    buffer_path = config['data']['buffer_file']
    if not os.path.exists(buffer_path):
        print(f"错误: 数据文件不存在: {buffer_path}")
        sys.exit(1)
    
    buffer = load_buffer(buffer_path, device)
    data, actions, episode_budgets, health, raw_cost, rewards, log_budgets, actual_n_agents = extract_data_from_buffer(buffer)
    
    if rewards is None:
        print("错误: 数据中没有奖励信息!")
        sys.exit(1)
    
    # 分析奖励分布
    analyze_reward_distribution(rewards, raw_cost=raw_cost, health=health)


if __name__ == "__main__":
    main()