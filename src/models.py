"""
Neural network architectures for SePar:
1. ObservationFeatureExtractor: VGG-style spatial convolutional blocks coupled with LSTM recurrent memory.
2. MultiAgentTransformer (MAT): Transformer Encoder-Decoder architecture for inter-agent attention and action decoding.
"""

import torch
import torch.nn as nn


class ObservationFeatureExtractor(nn.Module):
    """
    Extracts spatial and temporal latent representations from local agent grid views.
    Combines VGG convolutional blocks with an LSTM cell to retain temporal history.
    """
    def __init__(self, input_channels=3, hidden_dim=128, lstm_hidden=128, device="cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.device = device
        self.conv_net = None  # Built dynamically based on input channel dimension
        self.lstm = nn.LSTMCell(self.hidden_dim, self.lstm_hidden).to(self.device)

    def _build_conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, self.hidden_dim),
            nn.ReLU()
        ).to(self.device)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for perception and memory encoding.
        Args:
            x: Spatial observation tensor [Batch or Agents, Channels, Height, Width]
            h_prev, c_prev: Hidden and cell state tensors [Batch or Agents, lstm_hidden]
        Returns:
            h_next: Current timestep latent embedding z_t
            (h_next, c_next): Updated recurrent state tuple
        """
        if self.conv_net is None:
            self.conv_net = self._build_conv(x.shape[1])

        conv_out = self.conv_net(x)
        h_next, c_next = self.lstm(conv_out, (h_prev, c_prev))
        return h_next, (h_next, c_next)


class MultiAgentTransformer(nn.Module):
    """
    Multi-Agent Transformer (MAT) enabling implicit inter-agent coordination.
    - Encoder: Self-attention over all agent observation embeddings.
    - Decoder: Auto-regressive action probability decoding with causal masking.
    - Heads: State-value estimation head and discrete action distribution head.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_actions = config.num_actions

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.num_transformer_layers
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=config.num_transformer_layers
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def forward(self, z_seq, action_seq=None):
        """
        Args:
            z_seq: Agent observation embeddings [Batch, Num_Agents, Hidden_Dim]
            action_seq: Prior agent action sequence for auto-regressive decoding (optional)
        Returns:
            action_probs: Predicted action probabilities over discrete actions (or None)
            state_values: Critic state-value estimations [Batch, Num_Agents, 1]
        """
        if len(z_seq.shape) == 2:
            z_seq = z_seq.unsqueeze(0)

        refined_obs = self.transformer_encoder(z_seq)
        state_values = self.value_head(refined_obs)

        if action_seq is not None:
            if len(action_seq.shape) == 2:
                action_seq = action_seq.unsqueeze(0)
            decoder_out = self.transformer_decoder(action_seq, refined_obs)
            action_logits = self.action_head(decoder_out)
            action_probs = torch.softmax(action_logits, dim=-1)
            return action_probs, state_values

        return None, state_values
