"""
Configuration module for the SePar (Sequence Pathfinder for MAPD) framework.
Defines simulation environment settings, network dimensions, and training hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import torch


@dataclass
class Config:
    """
    Central configuration for SePar simulation, neural models, and optimization.
    """
    # Environment settings
    num_agents: int = 8
    grid_size: int = 32
    fov_size: int = 13  # Field of View (13x13) as described in the paper
    num_agents_list: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512])
    max_steps: int = 20000  # Maximum evaluation steps for warehouse maps
    is_lifelong: bool = True
    env_name: str = "warehouse_small"  # Supported: "warehouse_small", "warehouse_large", "pogema_mazes"

    # Network and Multi-Agent Transformer (MAT) architecture
    hidden_dim: int = 128
    num_transformer_layers: int = 3
    n_heads: int = 4
    lstm_hidden: int = 128
    num_actions: int = 5  # Actions: [stay, up, down, left, right]

    # Training & Optimization Settings
    train_iterations: int = 5
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    rl_batch_size: int = 64
    il_batch_size: int = 64
    ppo_clip_param: float = 0.2
    ppo_epochs: int = 4
    il_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    randomize_agent_order: bool = True
    unroll_sequence: bool = True
    max_grad_norm: float = 0.5
    eval_episodes: int = 3
    device: Optional[str] = None
    mode: str = "train"  # Options: "train" or "eval"

    def __post_init__(self):
        self.obs_radius = self.fov_size // 2
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        """Serialize configuration to a standard dictionary."""
        d = asdict(self)
        d["obs_radius"] = self.obs_radius
        return d
