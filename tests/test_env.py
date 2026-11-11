"""
Unit tests for environment wrapper and grid interaction.
"""

import unittest
import torch
from src.config import Config
from src.environment import PogemaEnvWrapper, MockGridEnvWrapper, create_env


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.device = "cpu"
        self.config.num_agents = 4

    def test_mock_env_reset_and_step(self):
        """Test fallback grid environment reset and step dynamics."""
        env = MockGridEnvWrapper(config=self.config)
        obs = env.reset()
        self.assertEqual(obs.shape, (self.config.num_agents, 3, self.config.fov_size, self.config.fov_size))

        actions = torch.randint(0, self.config.num_actions, (self.config.num_agents,))
        next_obs, rewards, dones, infos = env.step(actions)

        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(rewards.shape, (self.config.num_agents,))
        self.assertEqual(dones.shape, (self.config.num_agents,))
        self.assertIn("delivered_targets", infos)

    def test_create_env_factory(self):
        """Verify factory function instantiates a compliant environment."""
        env = create_env(config=self.config)
        self.assertIsNotNone(env)
        obs = env.reset()
        self.assertEqual(obs.shape, (self.config.num_agents, 3, self.config.fov_size, self.config.fov_size))

    def test_vertex_collision_blocked(self):
        """
        MAPF Physics Invariant:
        Two agents moving to the same cell (vertex conflict) must be blocked
        and remain in their original positions.
        """
        import numpy as np
        self.config.num_agents = 2
        env = MockGridEnvWrapper(config=self.config)
        env.reset()
        # Place agent 0 at (5, 5) and agent 1 at (5, 7)
        env.agent_positions = [np.array([5, 5]), np.array([5, 7])]
        env.target_positions = [np.array([1, 1]), np.array([2, 2])]

        # Agent 0 moves right (4 -> (5, 6)), Agent 1 moves left (3 -> (5, 6))
        actions = torch.tensor([4, 3])
        env.step(actions)

        # Both must remain at original positions
        self.assertTrue(np.array_equal(env.agent_positions[0], [5, 5]), "Agent 0 was not blocked on vertex collision")
        self.assertTrue(np.array_equal(env.agent_positions[1], [5, 7]), "Agent 1 was not blocked on vertex collision")

    def test_edge_collision_blocked(self):
        """
        MAPF Physics Invariant:
        Two agents attempting to swap positions across the same edge in one step
        must be blocked and remain in their original positions.
        """
        import numpy as np
        self.config.num_agents = 2
        env = MockGridEnvWrapper(config=self.config)
        env.reset()
        # Place agent 0 at (5, 5) and agent 1 at (5, 6)
        env.agent_positions = [np.array([5, 5]), np.array([5, 6])]
        env.target_positions = [np.array([1, 1]), np.array([2, 2])]

        # Agent 0 moves right (4 -> (5, 6)), Agent 1 moves left (3 -> (5, 5))
        actions = torch.tensor([4, 3])
        env.step(actions)

        # Both must remain at original positions
        self.assertTrue(np.array_equal(env.agent_positions[0], [5, 5]), "Agent 0 was not blocked on edge collision")
        self.assertTrue(np.array_equal(env.agent_positions[1], [5, 6]), "Agent 1 was not blocked on edge collision")

    def test_episodic_inactive_agent_single_reward_and_no_obstruction(self):
        """
        Verify that in episodic mode (is_lifelong=False), when an agent reaches its target:
        1. It receives +1.0 delivery reward exactly once and is marked inactive.
        2. Stationary steps after delivery yield 0.0 reward and do not re-increment delivered count.
        3. Completed inactive agents do not block active agents moving to that cell.
        """
        import numpy as np
        self.config.num_agents = 2
        self.config.is_lifelong = False
        env = MockGridEnvWrapper(config=self.config)
        env.reset()

        # Agent 0 at (5, 5) with target at (5, 6)
        # Agent 1 at (5, 7) with target at (1, 1)
        env.agent_positions = [np.array([5, 5]), np.array([5, 7])]
        env.target_positions = [np.array([5, 6]), np.array([1, 1])]

        # Step 1: Agent 0 moves right (4 -> (5, 6)) to target; Agent 1 stays (0)
        actions = torch.tensor([4, 0])
        _, rewards, dones, infos = env.step(actions)

        self.assertEqual(rewards[0].item(), 1.0, "Agent 0 did not receive completion reward")
        self.assertEqual(dones[0].item(), 1.0, "Agent 0 was not marked done")
        self.assertEqual(infos["delivered_targets"], 1, "Delivered count should be 1")
        self.assertTrue(env.agent_done_flags[0], "Agent 0 was not flagged inactive")

        # Step 2: Agent 0 stays (0) at target cell (5, 6); Agent 1 moves left (3 -> (5, 6)) into Agent 0's cell
        actions2 = torch.tensor([0, 3])
        _, rewards2, dones2, infos2 = env.step(actions2)

        # Agent 0 should receive 0.0 reward, delivered count must remain 1
        self.assertEqual(rewards2[0].item(), 0.0, "Inactive agent received duplicate reward")
        self.assertEqual(infos2["delivered_targets"], 0, "Inactive agent re-incremented delivered count")

        # Agent 1 must NOT be blocked by inactive Agent 0 and successfully occupy (5, 6)
        self.assertTrue(np.array_equal(env.agent_positions[1], [5, 6]), "Active agent was blocked by inactive agent")

    def test_lifelong_target_respawns_only_on_free_cells(self):
        """
        Verify that in lifelong mode (is_lifelong=True), whenever targets are delivered,
        the newly respawned targets are strictly placed on obstacle-free cells.
        """
        import numpy as np
        self.config.num_agents = 2
        self.config.is_lifelong = True
        self.config.env_name = "warehouse_small"
        env = MockGridEnvWrapper(config=self.config)
        env.reset()

        # Simulate 50 deliveries to test stochastic target respawns
        for _ in range(50):
            for i in range(env.num_agents):
                # Teleport agent adjacent to target and step into it
                target = env.target_positions[i].copy()
                env.agent_positions[i] = target.copy()

            # Stay action (0) triggers delivery when dist == 0
            actions = torch.tensor([0, 0])
            env.step(actions)

            # Assert every respawned target is strictly on a free cell
            for i in range(env.num_agents):
                tr, tc = env.target_positions[i]
                self.assertEqual(
                    env.obstacles[tr, tc], 0.0,
                    f"Target {i} was respawned on an obstacle cell ({tr}, {tc})"
                )


if __name__ == "__main__":
    unittest.main()
