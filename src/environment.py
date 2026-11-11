"""
Environment wrapper module for Multi-Agent Path Finding (MAPF/MAPD).
Wraps the POGEMA (Partially Observable Grid Environment for Multiple Agents) gym interface,
with a native grid world simulation fallback when POGEMA is not installed.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from pogema import pogema_v0, GridConfig
    POGEMA_AVAILABLE = True
except ImportError:
    pogema_v0 = None
    GridConfig = None
    POGEMA_AVAILABLE = False


class PogemaEnvWrapper:
    """
    Gym wrapper for POGEMA environment supporting episodic and lifelong multi-agent execution.
    Handles observation batch tensor formatting and device mapping.
    """
    def __init__(self, config=None, num_agents=8, is_lifelong=True, device="cpu"):
        if not POGEMA_AVAILABLE:
            raise ImportError("POGEMA library is not installed. Install with `pip install pogema`.")

        self.num_agents = config.num_agents if config else num_agents
        self.device = config.device if config else device
        self.is_lifelong = config.is_lifelong if config else is_lifelong
        self.env_name = getattr(config, "env_name", "warehouse_small") if config else "warehouse_small"
        max_episode_steps = 256 if self.is_lifelong else 128

        grid_kwargs = {
            "num_agents": self.num_agents,
            "obs_radius": config.obs_radius if config else 5,
            "max_episode_steps": max_episode_steps,
            "on_target": "restart" if self.is_lifelong else "finish"
        }

        # Load specific map topology if requested (e.g. warehouse or mazes)
        if self.env_name and (self.env_name.startswith("warehouse") or "maze" in self.env_name):
            try:
                self.grid_config = GridConfig(map_name=self.env_name, **grid_kwargs)
            except Exception:
                self.grid_config = GridConfig(size=config.grid_size if config else 32, **grid_kwargs)
        else:
            self.grid_config = GridConfig(size=config.grid_size if config else 32, **grid_kwargs)

        self.env = pogema_v0(grid_config=self.grid_config)

    def reset(self):
        """Reset environment and return observation tensor."""
        reset_result = self.env.reset()
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


class MockGridEnvWrapper:
    """
    Lightweight, deterministic Multi-Agent Grid World simulation environment.
    Serves as an operational standalone fallback when external POGEMA dependencies are unavailable.
    Provides identical observation tensor shapes [Num_Agents, 3, FOV, FOV] and Gym interface.
    """
    def __init__(self, config=None, num_agents=8, grid_size=32, fov_size=13, is_lifelong=True, device="cpu"):
        self.num_agents = config.num_agents if config else num_agents
        self.grid_size = config.grid_size if config else grid_size
        self.fov_size = config.fov_size if config else fov_size
        self.is_lifelong = config.is_lifelong if config else is_lifelong
        self.env_name = getattr(config, "env_name", "warehouse_small") if config else "warehouse_small"
        self.device = config.device if config else device
        self.max_steps = 256 if self.is_lifelong else 128
        self.current_step = 0
        self.radius = self.fov_size // 2

        self.rng = np.random.default_rng(seed=42)
        self.agent_positions = []
        self.target_positions = []
        self.agent_done_flags = np.zeros(self.num_agents, dtype=bool)
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Add perimeter obstacles
        self.obstacles[0, :] = 1.0
        self.obstacles[-1, :] = 1.0
        self.obstacles[:, 0] = 1.0
        self.obstacles[:, -1] = 1.0

        # Topology-aware layout generation
        if self.env_name and self.env_name.startswith("warehouse"):
            for r in range(3, self.grid_size - 3, 3):
                for c in range(3, self.grid_size - 3):
                    if c % 5 != 0:  # Cross-aisles
                        self.obstacles[r, c] = 1.0
        elif self.env_name and "maze" in self.env_name:
            for r in range(2, self.grid_size - 2, 2):
                for c in range(2, self.grid_size - 2):
                    if (r // 2) % 2 == 0 and c < self.grid_size - 5:
                        self.obstacles[r, c] = 1.0
                    elif (r // 2) % 2 == 1 and c > 4:
                        self.obstacles[r, c] = 1.0

        # Precompute obstacle-free navigable cells
        self.free_cells = [
            (r, c) for r in range(1, self.grid_size - 1)
            for c in range(1, self.grid_size - 1)
            if self.obstacles[r, c] == 0.0
        ]

    def reset(self):
        """Reset agent and target locations."""
        self.current_step = 0
        self.agent_done_flags = np.zeros(self.num_agents, dtype=bool)
        chosen = self.rng.choice(len(self.free_cells), size=self.num_agents * 2, replace=False)
        self.agent_positions = [np.array(self.free_cells[idx]) for idx in chosen[:self.num_agents]]
        self.target_positions = [np.array(self.free_cells[idx]) for idx in chosen[self.num_agents:]]
        return self._generate_observations()

    def _generate_observations(self):
        """Constructs 3-channel local field-of-view (FOV) tensors for each agent."""
        obs = np.zeros((self.num_agents, 3, self.fov_size, self.fov_size), dtype=np.float32)
        for i, pos in enumerate(self.agent_positions):
            pr, pc = pos[0], pos[1]
            for dr in range(-self.radius, self.radius + 1):
                for dc in range(-self.radius, self.radius + 1):
                    gr, gc = pr + dr, pc + dc
                    fr, fc = dr + self.radius, dc + self.radius
                    # Channel 0: Obstacles / Boundaries
                    if gr < 0 or gr >= self.grid_size or gc < 0 or gc >= self.grid_size:
                        obs[i, 0, fr, fc] = 1.0
                    elif self.obstacles[gr, gc] == 1.0:
                        obs[i, 0, fr, fc] = 1.0

            # Channel 1: Self & Neighboring Active Agents
            for other_idx, other_pos in enumerate(self.agent_positions):
                if not self.is_lifelong and self.agent_done_flags[other_idx] and other_idx != i:
                    continue  # Completed inactive agents do not obstruct or render as active obstacles
                dr = other_pos[0] - pr
                dc = other_pos[1] - pc
                if abs(dr) <= self.radius and abs(dc) <= self.radius:
                    obs[i, 1, dr + self.radius, dc + self.radius] = 1.0

            # Channel 2: Agent Target
            tr, tc = self.target_positions[i]
            dr = tr - pr
            dc = tc - pc
            if abs(dr) <= self.radius and abs(dc) <= self.radius:
                obs[i, 2, dr + self.radius, dc + self.radius] = 1.0

        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, actions_tensor):
        """
        Execute actions for all agents.
        Actions: 0: Stay, 1: Up (-1, 0), 2: Down (+1, 0), 3: Left (0, -1), 4: Right (0, +1).
        """
        self.current_step += 1
        if isinstance(actions_tensor, torch.Tensor):
            actions = actions_tensor.cpu().numpy().tolist()
        else:
            actions = list(actions_tensor)

        move_map = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        rewards = []
        dones = []
        delivered = 0

        # Collision detection & position updates using MAPF physics (dead code eliminated)
        curr_positions = [pos.copy() for pos in self.agent_positions]
        intended_positions = []
        for i, act in enumerate(actions):
            if not self.is_lifelong and self.agent_done_flags[i]:
                # Inactive agent does not move
                intended_positions.append(curr_positions[i].copy())
                continue
            dr, dc = move_map.get(int(act), (0, 0))
            new_r = int(np.clip(curr_positions[i][0] + dr, 0, self.grid_size - 1))
            new_c = int(np.clip(curr_positions[i][1] + dc, 0, self.grid_size - 1))
            intended_positions.append(np.array([new_r, new_c]))

        # Active agents participating in MAPF collision resolution
        active_indices = [i for i in range(self.num_agents) if self.is_lifelong or not self.agent_done_flags[i]]

        # Iterative collision resolution for obstacle, edge (swap), and vertex conflicts
        for _ in range(self.num_agents + 1):
            changed = False

            # 1. Obstacle collisions (active agents only)
            for i in active_indices:
                r, c = intended_positions[i]
                if self.obstacles[r, c] == 1.0:
                    if not np.array_equal(intended_positions[i], curr_positions[i]):
                        intended_positions[i] = curr_positions[i].copy()
                        changed = True

            # 2. Edge collisions (active agents swapping positions)
            for idx_a in range(len(active_indices)):
                i = active_indices[idx_a]
                for idx_b in range(idx_a + 1, len(active_indices)):
                    j = active_indices[idx_b]
                    if (np.array_equal(intended_positions[i], curr_positions[j]) and
                        np.array_equal(intended_positions[j], curr_positions[i])):
                        if not np.array_equal(intended_positions[i], curr_positions[i]):
                            intended_positions[i] = curr_positions[i].copy()
                            changed = True
                        if not np.array_equal(intended_positions[j], curr_positions[j]):
                            intended_positions[j] = curr_positions[j].copy()
                            changed = True

            # 3. Vertex collisions (multiple active agents targeting the same cell)
            occupancy = {}
            for i in active_indices:
                cell = (int(intended_positions[i][0]), int(intended_positions[i][1]))
                occupancy.setdefault(cell, []).append(i)

            for cell, occupants in occupancy.items():
                if len(occupants) > 1:
                    for i in occupants:
                        if not np.array_equal(intended_positions[i], curr_positions[i]):
                            intended_positions[i] = curr_positions[i].copy()
                            changed = True

            if not changed:
                break

        self.agent_positions = intended_positions

        for i in range(self.num_agents):
            if not self.is_lifelong and self.agent_done_flags[i]:
                # Completed agent in episodic mode: zero additional rewards, done flag stays True
                rewards.append(0.0)
                dones.append(1.0)
                continue

            dist = np.abs(self.agent_positions[i] - self.target_positions[i]).sum()
            if dist == 0:
                rewards.append(1.0)
                delivered += 1
                if self.is_lifelong:
                    # Respawn target at new random obstacle-free location
                    rand_idx = self.rng.integers(0, len(self.free_cells))
                    self.target_positions[i] = np.array(self.free_cells[rand_idx])
                    dones.append(1.0 if self.current_step >= self.max_steps else 0.0)
                else:
                    self.agent_done_flags[i] = True
                    dones.append(1.0)
            else:
                rewards.append(-0.01)
                dones.append(1.0 if self.current_step >= self.max_steps else 0.0)

        next_obs = self._generate_observations()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        infos = {"delivered_targets": delivered, "step": self.current_step}

        return next_obs, rewards_tensor, dones_tensor, infos


def create_env(config=None):
    """
    Factory function to instantiate the preferred environment.
    Defaults to PogemaEnvWrapper if POGEMA is installed; otherwise falls back to MockGridEnvWrapper.
    """
    if POGEMA_AVAILABLE:
        logger.info("Initializing POGEMA environment wrapper.")
        return PogemaEnvWrapper(config=config)
    else:
        logger.info("POGEMA not detected. Initializing high-fidelity native GridWorld simulation wrapper.")
        return MockGridEnvWrapper(config=config)
