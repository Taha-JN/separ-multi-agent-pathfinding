"""
Benchmarking and performance visualization module for SePar.
Executes genuine policy evaluation on simulation environments and plots empirical metrics.
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


def evaluate_policy(env, policy_network, num_episodes=5, max_steps=128, device="cpu"):
    """
    Executes an empirical evaluation of the trained policy network.
    Runs greedy deterministic action selection over multiple evaluation episodes.
    Args:
        env: Environment instance (PogemaEnvWrapper or MockGridEnvWrapper).
        policy_network: Unified PolicyNetwork instance.
        num_episodes: Number of complete evaluation episodes to run.
        max_steps: Maximum timesteps per episode.
        device: Hardware device tensor mapping.
    Returns:
        metrics: Dictionary containing mean_return, throughput, success_rate, and latency_ms.
    """
    policy_network.eval()
    episode_returns = []
    episode_deliveries = []
    step_latencies = []
    total_steps = 0

    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset()
            h = torch.zeros(env.num_agents, policy_network.lstm_hidden, device=device)
            c = torch.zeros(env.num_agents, policy_network.lstm_hidden, device=device)

            ep_return = 0.0
            ep_deliveries = 0

            for step in range(max_steps):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                actions, _, _, (h, c) = policy_network.act(obs, h, c, deterministic=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                step_latencies.append((t_end - t_start) * 1000.0)  # ms

                next_obs, rewards, dones, infos = env.step(actions)
                ep_return += rewards.sum().item()
                delivered = infos.get("delivered_targets", 0) if isinstance(infos, dict) else 0
                ep_deliveries += delivered

                obs = next_obs
                total_steps += 1

                if dones.all():
                    break

            episode_returns.append(ep_return)
            episode_deliveries.append(ep_deliveries)

    mean_return = float(np.mean(episode_returns))
    total_deliveries = sum(episode_deliveries)
    throughput = total_deliveries / max(total_steps, 1)  # Targets delivered per timestep
    mean_latency_ms = float(np.mean(step_latencies)) if step_latencies else 0.0
    success_rate = 1.0 if total_deliveries > 0 else 0.0

    eval_results = {
        "mean_return": mean_return,
        "total_deliveries": total_deliveries,
        "throughput": throughput,
        "mean_latency_ms": mean_latency_ms,
        "num_episodes": num_episodes,
        "total_steps": total_steps
    }

    logger.info(
        f"Evaluation ({num_episodes} eps) | Return: {mean_return:.2f} | "
        f"Throughput: {throughput:.4f} targets/step | Latency: {mean_latency_ms:.2f} ms"
    )
    return eval_results


def plot_training_history(history, save_dir="benchmarks"):
    """
    Plots training loss and return curves from real training history.
    Args:
        history: Dictionary containing lists of 'rl_loss', 'policy_loss', 'value_loss', and 'returns'.
        save_dir: Destination directory for saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    iterations = range(1, len(history.get("rl_loss", [])) + 1)

    # Subplot 1: Losses
    axes[0].plot(iterations, history.get("policy_loss", []), label="Policy Loss", color="royalblue")
    axes[0].plot(iterations, history.get("value_loss", []), label="Value Loss", color="darkorange")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend()

    # Subplot 2: Episode Returns
    if "returns" in history and history["returns"]:
        axes[1].plot(iterations, history["returns"], label="Mean Return", color="forestgreen")
        axes[1].set_xlabel("Training Iteration")
        axes[1].set_ylabel("Average Return")
        axes[1].set_title("Episode Reward Progression")
        axes[1].grid(True, alpha=0.5)
        axes[1].legend()

    out_file = os.path.join(save_dir, "training_progress.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Training progress curves saved to: {out_file}")


def plot_evaluation_benchmark(eval_metrics_map, save_dir="benchmarks"):
    """
    Plots empirical throughput and decision latency across various agent counts.
    Args:
        eval_metrics_map: Dict mapping agent_count (int) to evaluation result dict.
        save_dir: Destination directory for saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    agent_counts = sorted(eval_metrics_map.keys())
    throughputs = [eval_metrics_map[a]["throughput"] for a in agent_counts]
    latencies = [eval_metrics_map[a]["mean_latency_ms"] for a in agent_counts]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:green"
    ax1.set_xlabel("Number of Agents", fontsize=12)
    ax1.set_ylabel("Throughput (targets/step)", color=color, fontsize=12)
    ax1.plot(agent_counts, throughputs, marker="^", color=color, linewidth=2, label="SePar Throughput")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(agent_counts)
    ax1.set_xticklabels([f"$2^{{{int(np.log2(a))}}}$" for a in agent_counts])
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Latency (ms / step)", color=color, fontsize=12)
    ax2.plot(agent_counts, latencies, marker="o", color=color, linestyle="--", linewidth=1.5, label="Decision Latency")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Empirical Multi-Agent Performance & Latency", fontsize=14)
    out_file = os.path.join(save_dir, "Full_Results_Fig5_Fig6.png")
    fig.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Empirical evaluation benchmark curves saved to: {out_file}")
