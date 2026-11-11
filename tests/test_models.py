"""
Unit tests for neural network architectures (ObservationFeatureExtractor, MultiAgentTransformer, PolicyNetwork).
"""

import unittest
import torch
from src.config import Config
from src.models import ObservationFeatureExtractor, MultiAgentTransformer, PolicyNetwork


class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.device = "cpu"
        self.num_agents = 4
        self.config.num_agents = self.num_agents

    def test_feature_extractor_parameters_registered(self):
        """Verify that CNN and LSTM parameters are registered in __init__ and captured by an optimizer."""
        extractor = ObservationFeatureExtractor(
            input_channels=3,
            hidden_dim=self.config.hidden_dim,
            lstm_hidden=self.config.lstm_hidden
        )
        params = list(extractor.parameters())
        self.assertGreater(len(params), 0, "Parameters should be registered in __init__")

        # Test optimizer instantiation before forward pass
        optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)
        self.assertEqual(len(optimizer.param_groups[0]['params']), len(params))

    def test_feature_extractor_forward(self):
        """Test feature extractor forward pass tensor dimensions."""
        extractor = ObservationFeatureExtractor(
            input_channels=3,
            hidden_dim=self.config.hidden_dim,
            lstm_hidden=self.config.lstm_hidden
        )
        x = torch.randn(self.num_agents, 3, 13, 13)
        h_prev = torch.zeros(self.num_agents, self.config.lstm_hidden)
        c_prev = torch.zeros(self.num_agents, self.config.lstm_hidden)

        emb, (h_next, c_next) = extractor(x, h_prev, c_prev)
        self.assertEqual(emb.shape, (self.num_agents, self.config.hidden_dim))
        self.assertEqual(h_next.shape, (self.num_agents, self.config.lstm_hidden))
        self.assertEqual(c_next.shape, (self.num_agents, self.config.lstm_hidden))

    def test_multi_agent_transformer_action_embedding(self):
        """Verify that MultiAgentTransformer embeds discrete actions without dimension mismatch."""
        mat = MultiAgentTransformer(self.config)
        z_seq = torch.randn(2, self.num_agents, self.config.hidden_dim)
        action_seq = torch.randint(0, self.config.num_actions, (2, self.num_agents))

        logits, values = mat(z_seq, action_seq)
        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (2, self.num_agents, self.config.num_actions))
        self.assertEqual(values.shape, (2, self.num_agents, 1))

    def test_policy_network_act(self):
        """Test unified PolicyNetwork step and action sampling."""
        policy = PolicyNetwork(self.config)
        obs = torch.randn(self.num_agents, 3, 13, 13)
        h_prev = torch.zeros(self.num_agents, self.config.lstm_hidden)
        c_prev = torch.zeros(self.num_agents, self.config.lstm_hidden)

        actions, log_probs, values, (h_next, c_next) = policy.act(obs, h_prev, c_prev)
        self.assertEqual(actions.shape, (self.num_agents,))
        self.assertEqual(log_probs.shape, (self.num_agents,))
        self.assertEqual(values.shape, (self.num_agents, 1))

    def test_policy_network_evaluate_actions(self):
        """Test parallel action evaluation and entropy computation."""
        policy = PolicyNetwork(self.config)
        obs = torch.randn(2, self.num_agents, 3, 13, 13)
        h_prev = torch.zeros(2 * self.num_agents, self.config.lstm_hidden)
        c_prev = torch.zeros(2 * self.num_agents, self.config.lstm_hidden)
        actions = torch.randint(0, self.config.num_actions, (2, self.num_agents))

        log_probs, values, entropy = policy.evaluate_actions(obs, h_prev, c_prev, actions)
        self.assertEqual(log_probs.shape, (2, self.num_agents))
        self.assertEqual(values.shape, (2, self.num_agents, 1))
        self.assertEqual(entropy.shape, (2, self.num_agents))

    def test_end_to_end_gradient_propagation(self):
        """Ensure gradients propagate through both feature extractor and transformer."""
        policy = PolicyNetwork(self.config)
        obs = torch.randn(self.num_agents, 3, 13, 13)
        h = torch.zeros(self.num_agents, self.config.lstm_hidden)
        c = torch.zeros(self.num_agents, self.config.lstm_hidden)
        actions = torch.randint(0, self.config.num_actions, (1, self.num_agents))

        log_probs, values, _ = policy.evaluate_actions(obs.unsqueeze(0), h, c, actions)
        loss = log_probs.sum() + values.sum()
        loss.backward()

        # Verify CNN gradient
        self.assertIsNotNone(policy.feature_extractor.conv_net[0].weight.grad)
        # Verify Transformer encoder gradient
        self.assertIsNotNone(policy.transformer.transformer_encoder.layers[0].linear1.weight.grad)
        # Verify Action embedding gradient
        self.assertIsNotNone(policy.transformer.action_embed.weight.grad)

    def test_rollout_and_evaluation_distribution_alignment(self):
        """
        Mathematical Invariant Test:
        Assert that for identical states and sampled actions, the log probability evaluated
        via parallel teacher-forcing (evaluate_actions) matches the log probability produced
        during autoregressive rollout (act), guaranteeing a PPO importance sampling ratio of 1.0.
        """
        torch.manual_seed(42)
        policy = PolicyNetwork(self.config)
        policy.eval()

        obs = torch.randn(self.num_agents, 3, 13, 13)
        h = torch.zeros(self.num_agents, self.config.lstm_hidden)
        c = torch.zeros(self.num_agents, self.config.lstm_hidden)

        with torch.no_grad():
            actions, rollout_log_probs, rollout_values, _ = policy.act(obs, h, c, deterministic=False)
            eval_log_probs, eval_values, _ = policy.evaluate_actions(
                obs.unsqueeze(0), h, c, actions.unsqueeze(0)
            )

        eval_log_probs = eval_log_probs.squeeze(0)
        eval_values = eval_values.squeeze(0)

        # 1. Check absolute difference of log probs
        log_prob_diff = (rollout_log_probs - eval_log_probs).abs().max().item()
        self.assertLess(log_prob_diff, 1e-4, f"Rollout and evaluation log_probs diverged by {log_prob_diff}")

        # 2. Assert PPO importance sampling ratio is approximately 1.0
        ratio = torch.exp(eval_log_probs - rollout_log_probs)
        ratio_ones = torch.ones_like(ratio)
        self.assertTrue(
            torch.allclose(ratio, ratio_ones, atol=1e-4),
            f"PPO importance ratio diverges from 1.0: {ratio.tolist()}"
        )

        # 3. Assert value function outputs match
        value_diff = (rollout_values - eval_values).abs().max().item()
        self.assertLess(value_diff, 1e-4, f"Rollout and evaluation values diverged by {value_diff}")

    def test_lstm_bptt_gradient_propagation(self):
        """
        Mathematical Invariant Test:
        Assert that sequence-based LSTM unrolling connects the computational graph across time,
        enabling true Backpropagation Through Time (BPTT) with non-zero gradients on recurrent weights.
        """
        policy = PolicyNetwork(self.config)
        policy.train()

        T = 4
        N = self.num_agents
        obs_seq = torch.randn(T, N, 3, 13, 13)
        h_0 = torch.zeros(N, self.config.lstm_hidden, requires_grad=True)
        c_0 = torch.zeros(N, self.config.lstm_hidden)
        actions_seq = torch.randint(0, self.config.num_actions, (T, N))

        log_probs, values, _ = policy.evaluate_actions(
            obs_seq, h_0, c_0, actions_seq, unroll_sequence=True
        )
        loss = log_probs.sum() + values.sum()
        loss.backward()

        # Recurrent hidden-to-hidden weight must receive gradients across time steps
        lstm_hh_grad = policy.feature_extractor.lstm.weight_hh.grad
        self.assertIsNotNone(lstm_hh_grad, "LSTM weight_hh received no gradient")
        self.assertGreater(lstm_hh_grad.norm().item(), 0.0, "LSTM weight_hh gradient is zero")
        self.assertIsNotNone(h_0.grad, "Initial hidden state h_0 received no gradient through time")
        self.assertGreater(h_0.grad.norm().item(), 0.0, "Initial hidden state gradient is zero")

    def test_kv_cache_equivalence(self):
        """
        Performance Invariant Test:
        Assert that O(N) single-step autoregressive decoding using the KV cache produces identical
        hidden representations and action distributions as full O(N^2) TransformerDecoder passes.
        """
        mat = MultiAgentTransformer(self.config)
        mat.eval()

        z_seq = torch.randn(1, self.num_agents, self.config.hidden_dim)
        refined_obs = mat.transformer_encoder(z_seq)
        actions = torch.randint(0, self.config.num_actions, (1, self.num_agents))
        sos = torch.full((1, 1), self.config.num_actions, dtype=torch.long)
        dec_in = torch.cat([sos, actions[:, :-1]], dim=1)

        # Baseline full parallel decoder
        tgt_emb = mat.action_embed(dec_in)
        tgt_mask = torch.triu(torch.ones(self.num_agents, self.num_agents, dtype=torch.bool), diagonal=1)
        full_dec_out = mat.transformer_decoder(tgt_emb, refined_obs, tgt_mask=tgt_mask)

        # Step-by-step KV-cached decoder
        cache = mat.init_cache(refined_obs)
        step_outs = []
        for t in range(self.num_agents):
            tok = tgt_emb[:, t:t+1, :]
            out_t = mat.decode_step(tok, cache)
            step_outs.append(out_t)
        step_dec_out = torch.cat(step_outs, dim=1)

        max_diff = (full_dec_out - step_dec_out).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"KV-cached decoding diverged from full transformer by {max_diff}")

    def test_randomized_agent_order_alignment(self):
        """
        Symmetry Invariant Test:
        Assert that when a non-trivial random permutation of agents is specified,
        evaluate_actions properly factors and inverts the permutation, preserving exact log-prob alignment.
        """
        policy = PolicyNetwork(self.config)
        policy.eval()

        obs = torch.randn(self.num_agents, 3, 13, 13)
        h = torch.zeros(self.num_agents, self.config.lstm_hidden)
        c = torch.zeros(self.num_agents, self.config.lstm_hidden)

        # Custom random permutation
        custom_order = torch.randperm(self.num_agents)

        with torch.no_grad():
            actions, rollout_log_probs, _, _ = policy.act(
                obs, h, c, deterministic=False, agent_order=custom_order
            )
            eval_log_probs, _, _ = policy.evaluate_actions(
                obs.unsqueeze(0), h, c, actions.unsqueeze(0), agent_order=custom_order
            )

        eval_log_probs = eval_log_probs.squeeze(0)
        diff = (rollout_log_probs - eval_log_probs).abs().max().item()
        self.assertLess(diff, 1e-4, f"Permuted rollout and evaluation log_probs diverged by {diff}")


if __name__ == "__main__":
    unittest.main()
