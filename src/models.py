"""
Neural network architectures for SePar:
1. ObservationFeatureExtractor: VGG-style spatial convolutional blocks coupled with LSTM recurrent memory.
2. MultiAgentTransformer (MAT): Transformer Encoder-Decoder architecture for inter-agent attention and action decoding.
3. PolicyNetwork: Unified actor-critic network coupling feature extraction and multi-agent coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationFeatureExtractor(nn.Module):
    """
    Extracts spatial and temporal latent representations from local agent grid views.
    Combines VGG convolutional blocks with an LSTM cell to retain temporal history.
    """
    def __init__(self, input_channels=3, hidden_dim=128, lstm_hidden=128):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden

        # CNN feature extractor instantiated in __init__ for proper optimizer registration
        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, self.hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(self.hidden_dim, self.lstm_hidden)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for perception and memory encoding.
        Args:
            x: Spatial observation tensor [Batch, Channels, Height, Width] or [Agents, Channels, Height, Width]
            h_prev, c_prev: Hidden and cell state tensors [Batch/Agents, lstm_hidden]
        Returns:
            h_next: Current timestep latent embedding z_t
            (h_next, c_next): Updated recurrent state tuple
        """
        conv_out = self.conv_net(x)
        h_next, c_next = self.lstm(conv_out, (h_prev, c_prev))
        return h_next, (h_next, c_next)


class MultiAgentTransformer(nn.Module):
    """
    Multi-Agent Transformer (MAT) enabling implicit inter-agent coordination.
    - Encoder: Self-attention over all agent observation embeddings.
    - Decoder: Auto-regressive action decoding with causal masking and KV-cache acceleration.
    - Heads: State-value estimation head and discrete action distribution head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_actions = config.num_actions
        self.num_agents = config.num_agents
        self.n_heads = config.n_heads

        # Discrete action embedding (+1 for the Start-of-Sequence token)
        self.action_embed = nn.Embedding(self.num_actions + 1, self.hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.n_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.num_transformer_layers
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=config.n_heads,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=config.num_transformer_layers
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def init_cache(self, refined_obs):
        """
        Precomputes cross-attention Key/Value projections from refined_obs across all decoder
        layers to eliminate O(N^2) redundant cross-attention projections during autoregressive rollout.
        """
        cache = []
        nhead = self.n_heads
        d_model = self.hidden_dim
        d_head = d_model // nhead
        batch_size = refined_obs.size(0)

        for layer in self.transformer_decoder.layers:
            mha = layer.multihead_attn
            w_k = mha.in_proj_weight[d_model:2 * d_model]
            w_v = mha.in_proj_weight[2 * d_model:]
            b_k = mha.in_proj_bias[d_model:2 * d_model]
            b_v = mha.in_proj_bias[2 * d_model:]
            mem_k = F.linear(refined_obs, w_k, b_k).view(batch_size, -1, nhead, d_head).transpose(1, 2)
            mem_v = F.linear(refined_obs, w_v, b_v).view(batch_size, -1, nhead, d_head).transpose(1, 2)
            cache.append({
                "mem_k": mem_k,
                "mem_v": mem_v,
                "sa_k": [],
                "sa_v": []
            })
        return cache

    def decode_step(self, token_emb, cache):
        """
        Executes single-step autoregressive decoding using incremental self-attention KV-caching
        and precomputed cross-attention keys and values. Reduces generation latency from O(N^2) to O(N).
        Args:
            token_emb: Input token embedding [Batch, 1, Hidden_Dim]
            cache: Layerwise KV cache list
        Returns:
            x: Decoded output vector for current agent step [Batch, 1, Hidden_Dim]
        """
        x = token_emb
        nhead = self.n_heads
        d_model = self.hidden_dim
        d_head = d_model // nhead
        scale = d_head ** 0.5
        batch_size = x.size(0)

        for l_idx, layer in enumerate(self.transformer_decoder.layers):
            c = cache[l_idx]
            sa = layer.self_attn
            sa_w_q = sa.in_proj_weight[:d_model]
            sa_w_k = sa.in_proj_weight[d_model:2 * d_model]
            sa_w_v = sa.in_proj_weight[2 * d_model:]
            sa_b_q = sa.in_proj_bias[:d_model]
            sa_b_k = sa.in_proj_bias[d_model:2 * d_model]
            sa_b_v = sa.in_proj_bias[2 * d_model:]

            q = F.linear(x, sa_w_q, sa_b_q).view(batch_size, 1, nhead, d_head).transpose(1, 2)
            k = F.linear(x, sa_w_k, sa_b_k).view(batch_size, 1, nhead, d_head).transpose(1, 2)
            v = F.linear(x, sa_w_v, sa_b_v).view(batch_size, 1, nhead, d_head).transpose(1, 2)
            c["sa_k"].append(k)
            c["sa_v"].append(v)

            all_k = torch.cat(c["sa_k"], dim=2)
            all_v = torch.cat(c["sa_v"], dim=2)
            sa_score = torch.matmul(q, all_k.transpose(-2, -1)) / scale
            sa_weights = F.softmax(sa_score, dim=-1)
            sa_attn = torch.matmul(sa_weights, all_v).transpose(1, 2).contiguous().view(batch_size, 1, d_model)
            sa_out = sa.out_proj(sa_attn)
            x1 = layer.norm1(x + sa_out)

            # Cross-attention against cached memory K and V
            mha = layer.multihead_attn
            w_q = mha.in_proj_weight[:d_model]
            b_q = mha.in_proj_bias[:d_model]
            q_c = F.linear(x1, w_q, b_q).view(batch_size, 1, nhead, d_head).transpose(1, 2)
            ca_score = torch.matmul(q_c, c["mem_k"].transpose(-2, -1)) / scale
            ca_weights = F.softmax(ca_score, dim=-1)
            ca_attn = torch.matmul(ca_weights, c["mem_v"]).transpose(1, 2).contiguous().view(batch_size, 1, d_model)
            ca_out = mha.out_proj(ca_attn)
            x2 = layer.norm2(x1 + ca_out)

            # Feed-forward network
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            x = layer.norm3(x2 + ff_out)
        return x

    def forward(self, z_seq, action_seq=None):
        """
        Forward pass for multi-agent transformer.
        Args:
            z_seq: Agent observation embeddings [Batch, Num_Agents, Hidden_Dim] or [Num_Agents, Hidden_Dim]
            action_seq: Prior or current agent action sequence [Batch, Num_Agents] or [Num_Agents]
        Returns:
            action_logits: Unnormalized action logits [Batch, Num_Agents, Num_Actions] or None
            state_values: Critic state-value estimations [Batch, Num_Agents, 1]
        """
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(0)

        refined_obs = self.transformer_encoder(z_seq)
        state_values = self.value_head(refined_obs)

        if action_seq is not None:
            if action_seq.dim() == 1:
                action_seq = action_seq.unsqueeze(0)

            batch_size, seq_len = action_seq.shape

            # Teacher-forcing with Start-of-Sequence (SOS = num_actions) token prefix
            sos_tokens = torch.full((batch_size, 1), self.num_actions, dtype=torch.long, device=action_seq.device)
            decoder_input = torch.cat([sos_tokens, action_seq[:, :-1]], dim=1)

            tgt_emb = self.action_embed(decoder_input)

            # Causal square subsequent boolean mask enforcing autoregressive order without dtype casting warnings
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=z_seq.device, dtype=torch.bool), diagonal=1)

            decoder_out = self.transformer_decoder(tgt_emb, refined_obs, tgt_mask=tgt_mask)
            action_logits = self.action_head(decoder_out)
            return action_logits, state_values

        return None, state_values


class PolicyNetwork(nn.Module):
    """
    Unified Actor-Critic policy network coupling the spatial-temporal perception
    backbone with the Multi-Agent Transformer coordination module.
    Enables joint end-to-end optimization of both perception and coordination weights.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_actions = config.num_actions
        self.num_agents = config.num_agents
        self.hidden_dim = config.hidden_dim
        self.lstm_hidden = config.lstm_hidden

        self.feature_extractor = ObservationFeatureExtractor(
            input_channels=3,
            hidden_dim=config.hidden_dim,
            lstm_hidden=config.lstm_hidden
        )
        self.transformer = MultiAgentTransformer(config)

    def forward(self, obs, h_prev, c_prev, action_seq=None):
        """
        Forward pass for training evaluation.
        Args:
            obs: Observation tensor [Batch * Num_Agents, 3, H, W] or [Num_Agents, 3, H, W]
            h_prev, c_prev: Recurrent hidden and cell states
            action_seq: Action tensor [Batch, Num_Agents] or [Num_Agents]
        Returns:
            action_logits: Unnormalized logits [Batch, Num_Agents, Num_Actions]
            state_values: Value estimates [Batch, Num_Agents, 1]
            (h_next, c_next): Updated recurrent states
        """
        embeddings, (h_next, c_next) = self.feature_extractor(obs, h_prev, c_prev)

        if embeddings.dim() == 2 and embeddings.shape[0] == self.num_agents:
            z_seq = embeddings.unsqueeze(0)
        else:
            z_seq = embeddings.view(-1, self.num_agents, self.hidden_dim)

        action_logits, state_values = self.transformer(z_seq, action_seq=action_seq)
        return action_logits, state_values, (h_next, c_next)

    def act(self, obs, h_prev, c_prev, deterministic=False, agent_order=None):
        """
        Autoregressively sample actions for all agents in execution mode.
        Accelerated using incremental KV-caching (O(N) latency) and dynamic agent ordering
        to preserve permutation symmetry.
        Args:
            obs: Agent observation tensors [Num_Agents, 3, H, W]
            h_prev, c_prev: Recurrent state tensors [Num_Agents, lstm_hidden]
            deterministic: If True, selects greedy argmax action.
            agent_order: Optional 1D tensor specifying custom agent decoding permutation.
        Returns:
            actions: Chosen discrete actions [Num_Agents]
            log_probs: Log probabilities of chosen actions [Num_Agents]
            state_values: Value estimates [Num_Agents, 1]
            (h_next, c_next): Updated recurrent states
        """
        embeddings, (h_next, c_next) = self.feature_extractor(obs, h_prev, c_prev)
        z_seq = embeddings.unsqueeze(0)  # [1, Num_Agents, Hidden_Dim]

        refined_obs = self.transformer.transformer_encoder(z_seq)
        state_values = self.transformer.value_head(refined_obs).squeeze(0)  # [Num_Agents, 1]

        # Determine agent decoding order to ensure permutation symmetry
        if agent_order is None:
            if getattr(self.config, "randomize_agent_order", False) and not deterministic:
                agent_order = torch.randperm(self.num_agents, device=obs.device)
            else:
                agent_order = torch.arange(self.num_agents, device=obs.device)
        self.last_agent_order = agent_order

        # Initialize KV Cache for O(N) autoregressive decoding
        cache = self.transformer.init_cache(refined_obs)

        sos = torch.full((1, 1), self.num_actions, dtype=torch.long, device=obs.device)
        curr_token_emb = self.transformer.action_embed(sos)

        sampled_actions = torch.empty(self.num_agents, dtype=torch.long, device=obs.device)
        sampled_log_probs = torch.empty(self.num_agents, device=obs.device)

        for i in range(self.num_agents):
            agent_idx = agent_order[i].item() if agent_order.dim() > 0 else agent_order.item()
            dec_out = self.transformer.decode_step(curr_token_emb, cache)
            step_logits = self.transformer.action_head(dec_out.squeeze(1))

            dist = torch.distributions.Categorical(logits=step_logits)
            if deterministic:
                action = torch.argmax(step_logits, dim=-1)
            else:
                action = dist.sample()

            sampled_actions[agent_idx] = action.squeeze(0)
            sampled_log_probs[agent_idx] = dist.log_prob(action).squeeze(0)

            # Embed sampled action as next input token
            curr_token_emb = self.transformer.action_embed(action.unsqueeze(1))

        return sampled_actions, sampled_log_probs, state_values, (h_next, c_next)

    def evaluate_actions(self, obs, h_prev, c_prev, action_seq, agent_order=None, unroll_sequence=None, dones=None):
        """
        Parallel action evaluation using teacher forcing with causal mask for PPO updates.
        Supports sequence-based LSTM unrolling (Backpropagation Through Time), agent permutation restoration,
        and recurrent state masking across episode boundaries.
        Args:
            obs: Observations [Batch, Num_Agents, 3, H, W] or flattened [Batch * Num_Agents, 3, H, W]
            h_prev, c_prev: Recurrent states
            action_seq: Action sequence [Batch, Num_Agents]
            agent_order: Optional [Num_Agents] or [Batch, Num_Agents] permutation order
            unroll_sequence: If True and obs has 5 dims with T > 1, unroll LSTM through time.
            dones: Optional [Batch, Num_Agents] done tensor for resetting recurrent states at episode ends.
        Returns:
            log_probs: Log probabilities [Batch, Num_Agents]
            state_values: Value predictions [Batch, Num_Agents, 1]
            entropy: Distribution entropy [Batch, Num_Agents]
        """
        batch_size = action_seq.shape[0]

        if unroll_sequence is None:
            unroll_sequence = getattr(self.config, "unroll_sequence", True)

        if unroll_sequence and obs.dim() == 5 and obs.shape[0] > 1:
            # Sequence-based Recurrent PPO: unroll LSTM through time chunk [T, Num_Agents, ...]
            seq_len = obs.shape[0]
            h = h_prev[0] if h_prev.dim() == 3 else h_prev[:self.num_agents]
            c = c_prev[0] if c_prev.dim() == 3 else c_prev[:self.num_agents]
            embeddings_list = []
            for t in range(seq_len):
                if dones is not None and t > 0:
                    done_prev = dones[t - 1]
                    mask = (1.0 - done_prev).unsqueeze(-1) if done_prev.dim() == 1 else (1.0 - done_prev)
                    h = h * mask
                    c = c * mask
                emb_t, (h, c) = self.feature_extractor(obs[t], h, c)
                embeddings_list.append(emb_t)
            z_seq = torch.stack(embeddings_list, dim=0)  # [T, Num_Agents, hidden_dim]
        else:
            if obs.dim() == 5:
                obs_flat = obs.view(-1, *obs.shape[2:])
                h_flat = h_prev.view(-1, self.lstm_hidden)
                c_flat = c_prev.view(-1, self.lstm_hidden)
            else:
                obs_flat = obs
                h_flat = h_prev
                c_flat = c_prev
            embeddings, _ = self.feature_extractor(obs_flat, h_flat, c_flat)
            z_seq = embeddings.view(batch_size, self.num_agents, self.hidden_dim)

        if agent_order is None:
            agent_order = getattr(self, "last_agent_order", None)

        if agent_order is not None:
            if agent_order.dim() == 1:
                agent_order_expanded = agent_order.unsqueeze(0).expand(batch_size, -1)
            else:
                agent_order_expanded = agent_order

            perm_actions = torch.gather(action_seq, 1, agent_order_expanded)
            action_logits, state_values = self.transformer(z_seq, action_seq=perm_actions)

            dist = torch.distributions.Categorical(logits=action_logits)
            perm_log_probs = dist.log_prob(perm_actions)
            perm_entropy = dist.entropy()

            log_probs = torch.empty_like(perm_log_probs)
            log_probs.scatter_(1, agent_order_expanded, perm_log_probs)
            entropy = torch.empty_like(perm_entropy)
            entropy.scatter_(1, agent_order_expanded, perm_entropy)
        else:
            action_logits, state_values = self.transformer(z_seq, action_seq=action_seq)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(action_seq)
            entropy = dist.entropy()

        return log_probs, state_values, entropy
