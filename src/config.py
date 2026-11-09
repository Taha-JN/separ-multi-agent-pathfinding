"""
Configuration module for the SePar (Sequence Pathfinder for MAPD) framework.
Defines simulation environment settings, network dimensions, and training hyperparameters.
"""

import torch


class Config:
    def __init__(self):
        # Environment settings
        self.num_agents = 8
        self.fov_size = 13  # Field of View (13x13) as described in the paper
        self.num_agents_list = [8, 16, 32, 64, 128, 256, 512]  # Agent scaling list for evaluation
        self.max_steps = 20000  # Maximum evaluation steps for warehouse maps
        self.grid_size = 32
        self.obs_radius = 5
        self.is_lifelong = True
        self.env_name = "warehouse_small"  # Supported: "warehouse_small", "warehouse_large", "pogema_mazes"

        # Network and Multi-Agent Transformer (MAT) architecture
        self.hidden_dim = 128
        self.num_transformer_layers = 3
        self.n_heads = 4
        self.lstm_hidden = 128
        self.num_actions = 5  # Actions: [stay, up, down, left, right]

        # Reinforcement Learning (PPO) & Imitation Learning (IL) settings
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.rl_batch_size = 512
        self.il_batch_size = 512
        self.ppo_clip_param = 0.2
        self.ppo_epochs = 4
        self.il_epochs = 4
        self.max_grad_norm = 0.5

        # Runtime device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = "train"  # Options: "train" or "eval"
