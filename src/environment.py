"""
Environment wrapper module for Multi-Agent Path Finding (MAPF/MAPD).
Wraps the POGEMA (Partially Observable Grid Environment for Multiple Agents) gym interface.
"""

import numpy as np
import torch

try:
    from pogema import pogema_v0, GridConfig
except ImportError:
    pogema_v0 = None
    GridConfig = None


class PogemaEnvWrapper:
    """
    Gym wrapper for POGEMA environment supporting episodic and lifelong multi-agent execution.
    Handles observation batch tensor formatting and device mapping.
    """
    def __init__(self, config=None, num_agents=8, is_lifelong=True, device="cpu"):
        if pogema_v0 is None:
            raise ImportError("POGEMA library is not installed. Install with `pip install pogema`.")

        self.num_agents = config.num_agents if config else num_agents
        self.device = config.device if config else device
        self.is_lifelong = config.is_lifelong if config else is_lifelong
        max_episode_steps = 256 if self.is_lifelong else 128

        self.grid_config = GridConfig(
            num_agents=self.num_agents,
            size=config.grid_size if config else 32,
            obs_radius=config.obs_radius if config else 5,
            max_episode_steps=max_episode_steps,
            on_target="restart" if self.is_lifelong else "finish"
        )
        self.env = pogema_v0(grid_config=self.grid_config)

    def reset(self):
        """Reset environment and return observation tensor."""
        reset_result = self.env.reset()
        # Handle tuple output in newer Gym/Gymnasium API versions
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
        return self._process_obs(obs)

    def step(self, actions_tensor):
        """
        Execute one step across all agents.
        Args:
            actions_tensor: Tensor of discrete action indices [num_agents]
        Returns:
            next_obs_tensor, rewards_tensor, dones_tensor, infos
        """
        if isinstance(actions_tensor, torch.Tensor):
            actions_list = actions_tensor.cpu().numpy().tolist()
        else:
            actions_list = list(actions_tensor)

        step_result = self.env.step(actions_list)

        # Handle 4-tuple vs 5-tuple step API signatures
        if len(step_result) == 4:
            next_obs, rewards, dones, infos = step_result
        elif len(step_result) == 5:
            next_obs, rewards, terminated, truncated, infos = step_result
            dones = [t or tr for t, tr in zip(terminated, truncated)]
        else:
            raise ValueError(f"Unexpected step result signature of length {len(step_result)}")

        next_obs_tensor = self._process_obs(next_obs)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return next_obs_tensor, rewards_tensor, dones_tensor, infos

    def _process_obs(self, obs_list):
        """Format observations into float32 torch tensors with channel-first ordering [Agents, Channels, H, W]."""
        obs_array = np.array(obs_list, dtype=np.float32)
        obs_tensor = torch.tensor(obs_array, device=self.device)
        if len(obs_tensor.shape) == 4 and obs_tensor.shape[-1] < obs_tensor.shape[1]:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        return obs_tensor
