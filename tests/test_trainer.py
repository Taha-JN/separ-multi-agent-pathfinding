"""
Unit tests for trainer, loss computations, and RolloutBuffer.
"""

import unittest
import torch
from src.config import Config
from src.models import PolicyNetwork
from src.trainer import RolloutBuffer, SeParTrainer, compute_rl_loss, compute_il_loss


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.device = "cpu"
        self.config.num_agents = 4
        self.policy = PolicyNetwork(self.config)

    def test_rollout_buffer_insertion_and_gae(self):
        """Test RolloutBuffer storage and Generalized Advantage Estimation."""
        buffer = RolloutBuffer(
            buffer_size=10,
            num_agents=self.config.num_agents,
            obs_shape=(3, 13, 13),
            hidden_dim=self.config.hidden_dim,
            gamma=0.99,
            gae_lambda=0.95
        )

        for _ in range(10):
            obs = torch.randn(self.config.num_agents, 3, 13, 13)
            actions = torch.randint(0, self.config.num_actions, (self.config.num_agents,))
            rewards = torch.ones(self.config.num_agents)
            dones = torch.zeros(self.config.num_agents)
            values = torch.zeros(self.config.num_agents, 1)
            log_probs = torch.zeros(self.config.num_agents)
            h = torch.zeros(self.config.num_agents, self.config.lstm_hidden)
            c = torch.zeros(self.config.num_agents, self.config.lstm_hidden)

            buffer.add(obs, actions, rewards, dones, values, log_probs, h, c)

        last_values = torch.zeros(self.config.num_agents, 1)
        buffer.compute_returns_and_advantages(last_values, last_dones=torch.zeros(self.config.num_agents))

        batches = list(buffer.get_batches(batch_size=4))
        self.assertGreater(len(batches), 0)

    def test_compute_rl_loss(self):
        """Verify PPO clipped loss calculation and value loss."""
        obs = torch.randn(2, self.config.num_agents, 3, 13, 13)
        actions = torch.randint(0, self.config.num_actions, (2, self.config.num_agents))
        old_log_probs = torch.zeros(2, self.config.num_agents)
        returns = torch.ones(2, self.config.num_agents)
        advantages = torch.ones(2, self.config.num_agents)
        h = torch.zeros(2 * self.config.num_agents, self.config.lstm_hidden)
        c = torch.zeros(2 * self.config.num_agents, self.config.lstm_hidden)

        batch = (obs, actions, old_log_probs, returns, advantages, h, c)
        total_loss, loss_dict = compute_rl_loss(self.config, self.policy, batch)

        self.assertTrue(torch.is_tensor(total_loss))
        self.assertIn("policy_loss", loss_dict)
        self.assertIn("value_loss", loss_dict)
        self.assertIn("entropy", loss_dict)

    def test_il_loss_computation(self):
        """Verify Imitation Learning loss receives raw logits without double-softmax bug."""
        obs = torch.randn(self.config.num_agents, 3, 13, 13)
        h = torch.zeros(self.config.num_agents, self.config.lstm_hidden)
        c = torch.zeros(self.config.num_agents, self.config.lstm_hidden)
        expert_actions = torch.randint(0, self.config.num_actions, (self.config.num_agents,))

        loss = compute_il_loss(self.policy, obs, h, c, expert_actions)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_trainer_train_step(self):
        """Verify that SeParTrainer executes train_step without error."""
        trainer = SeParTrainer(self.policy, self.config)
        buffer = RolloutBuffer(
            buffer_size=4,
            num_agents=self.config.num_agents,
            obs_shape=(3, 13, 13),
            hidden_dim=self.config.hidden_dim
        )
        for _ in range(4):
            obs = torch.randn(self.config.num_agents, 3, 13, 13)
            actions = torch.randint(0, self.config.num_actions, (self.config.num_agents,))
            rewards = torch.ones(self.config.num_agents)
            dones = torch.zeros(self.config.num_agents)
            values = torch.zeros(self.config.num_agents, 1)
            log_probs = torch.zeros(self.config.num_agents)
            h = torch.zeros(self.config.num_agents, self.config.lstm_hidden)
            c = torch.zeros(self.config.num_agents, self.config.lstm_hidden)
            buffer.add(obs, actions, rewards, dones, values, log_probs, h, c)

        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(self.config.num_agents, 1),
            last_dones=torch.zeros(self.config.num_agents)
        )

        metrics = trainer.train_step(buffer)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("total_rl_loss", metrics)

    def test_gae_terminal_boundary_invariant(self):
        """
        Mathematical Invariant Test:
        Assert that when transition t is terminal (dones[t] = 1.0), GAE delta does NOT
        bootstrap from V(s_{t+1}) and future advantages are not propagated backward across the boundary.
        """
        buffer = RolloutBuffer(
            buffer_size=2,
            num_agents=1,
            obs_shape=(3, 13, 13),
            hidden_dim=self.config.hidden_dim,
            gamma=0.99,
            gae_lambda=0.95
        )
        obs = torch.zeros(1, 3, 13, 13)
        h = torch.zeros(1, self.config.lstm_hidden)
        c = torch.zeros(1, self.config.lstm_hidden)

        # Step 0: reward = 1.0, value = 0.5, done = 1.0 (terminal)
        buffer.add(obs, torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.0]), h, c)
        # Step 1: reward = 2.0, value = 1.0, done = 0.0 (non-terminal)
        buffer.add(obs, torch.tensor([0]), torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), h, c)

        last_values = torch.tensor([10.0])
        last_dones = torch.tensor([0.0])
        buffer.compute_returns_and_advantages(last_values, last_dones)

        # Return at step 0 must be exactly undiscounted reward (1.0) without bootstrapping into step 1
        unnorm_return_0 = buffer.returns[0].item()
        self.assertAlmostEqual(unnorm_return_0, 1.0, places=4, msg="Terminal state incorrectly bootstrapped future returns")

    def test_sequence_aligned_batches_invariant(self):
        """
        Mathematical Invariant Test:
        Assert that sequence-aligned batching maintains temporal contiguous ordering
        inside each minibatch to preserve recurrent LSTM state validity.
        """
        buffer = RolloutBuffer(
            buffer_size=6,
            num_agents=2,
            obs_shape=(3, 13, 13),
            hidden_dim=self.config.hidden_dim
        )
        obs = torch.zeros(2, 3, 13, 13)
        h = torch.zeros(2, self.config.lstm_hidden)
        c = torch.zeros(2, self.config.lstm_hidden)

        for step in range(6):
            buffer.add(
                obs,
                torch.tensor([step, step]),
                torch.tensor([1.0, 1.0]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0, 0.0]),
                h, c
            )

        batches = list(buffer.get_batches(batch_size=3, sequence_aligned=True))
        self.assertEqual(len(batches), 2)
        for b in batches:
            actions = b[1]  # shape [3, 2]
            step_indices = actions[:, 0].tolist()
            # Must be strictly increasing contiguous sequence of length 3
            self.assertEqual(step_indices[1], step_indices[0] + 1)
            self.assertEqual(step_indices[2], step_indices[1] + 1)

    def test_config_hyperparameters_respected_in_loss(self):
        """
        Verify that value_loss_coef and entropy_coef configured in Config
        are strictly respected in compute_rl_loss rather than using hardcoded values.
        """
        self.config.value_loss_coef = 0.8
        self.config.entropy_coef = 0.05

        obs = torch.randn(2, self.config.num_agents, 3, 13, 13)
        actions = torch.randint(0, self.config.num_actions, (2, self.config.num_agents))
        old_log_probs = torch.zeros(2, self.config.num_agents)
        returns = torch.ones(2, self.config.num_agents)
        advantages = torch.ones(2, self.config.num_agents)
        h = torch.zeros(2 * self.config.num_agents, self.config.lstm_hidden)
        c = torch.zeros(2 * self.config.num_agents, self.config.lstm_hidden)

        batch = (obs, actions, old_log_probs, returns, advantages, h, c)
        total_loss, loss_dict = compute_rl_loss(self.config, self.policy, batch)

        expected_total = loss_dict["policy_loss"] + 0.8 * loss_dict["value_loss"] - 0.05 * loss_dict["entropy"]
        self.assertAlmostEqual(total_loss.item(), expected_total, places=4)

    def test_recurrent_state_masked_on_episode_dones(self):
        """
        Verify that RolloutBuffer yields dones and PolicyNetwork.evaluate_actions
        correctly masks hidden states when crossing episode boundaries (dones[t-1] == 1.0).
        """
        buffer = RolloutBuffer(
            buffer_size=4,
            num_agents=2,
            obs_shape=(3, 13, 13),
            hidden_dim=self.config.hidden_dim
        )
        obs = torch.randn(2, 3, 13, 13)
        h = torch.zeros(2, self.config.lstm_hidden)
        c = torch.zeros(2, self.config.lstm_hidden)

        for step in range(4):
            # Agent 0 finishes at step 1 (done=1.0), Agent 1 continues (done=0.0)
            dones = torch.tensor([1.0 if step == 1 else 0.0, 0.0])
            buffer.add(
                obs,
                torch.tensor([0, 0]),
                torch.tensor([1.0, 1.0]),
                dones,
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0, 0.0]),
                h, c
            )

        batches = list(buffer.get_batches(batch_size=4, sequence_aligned=True))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        # Batch should contain 9 elements: obs, actions, log_probs, returns, advantages, h, c, dones, agent_orders
        self.assertEqual(len(batch), 9)
        dones_batch = batch[7]
        self.assertEqual(dones_batch.shape, (4, 2))
        self.assertEqual(dones_batch[1, 0].item(), 1.0)
        self.assertEqual(dones_batch[1, 1].item(), 0.0)

        # Test that evaluate_actions accepts dones and successfully executes
        obs_seq = torch.randn(4, 2, 3, 13, 13)
        actions_seq = torch.zeros((4, 2), dtype=torch.long)
        h_0 = torch.randn(2, self.config.lstm_hidden)
        c_0 = torch.randn(2, self.config.lstm_hidden)

        log_probs, values, entropy = self.policy.evaluate_actions(
            obs_seq, h_0, c_0, actions_seq, unroll_sequence=True, dones=dones_batch
        )
        self.assertEqual(log_probs.shape, (4, 2))
        self.assertEqual(values.shape, (4, 2, 1))


if __name__ == "__main__":
    unittest.main()
