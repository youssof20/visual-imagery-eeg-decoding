"""
EEGTransformer: neural network for decoding EEG imagery categories.

Takes a single-trial EEG segment (32 channels × 750 time points) and predicts
which of 10 imagery categories (or 3 supercategories) the person was imagining.
  Block 1: spatial convolution (which electrodes matter)
  Block 2: temporal convolution (which time windows matter)
  Block 3: Transformer encoder (attention over time)
  Block 4: classifier (10 or 3 classes)

Used by train_decoder.py (cross-subject) and within_subject_decoder.py.
"""

import math
import torch
import torch.nn as nn


class EEGTransformer(nn.Module):
    """
    Input: (B, 1, 32, 750)  [batch, 1, channels, time]
    Block 1: Conv2d(1, 32, (32,1)) -> BN -> ELU -> Dropout(0.25)  -> (B, 32, 1, 750)
    Block 2: Conv2d(32, 64, (1,16), stride=(1,4)) -> BN -> ELU -> Dropout(0.25) -> (B, 64, 1, 184)
    Block 3: Reshape (B, 184, 64), sinusoidal pos, TransformerEncoder 2 layers, 4 heads, dim_feedforward=128
    Block 4: GAP -> Linear(64, 64) -> ELU -> Dropout(0.5) -> Linear(64, n_classes)
    """

    def __init__(self, n_classes=10, n_channels=32, n_times=750, dropout=0.25, num_layers=2, nhead=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        # Block 1: spatial
        self.spatial_conv = nn.Conv2d(1, 32, kernel_size=(n_channels, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(dropout)
        # Block 2: temporal -> (n_times-16)//4+1
        self.temporal_conv = nn.Conv2d(32, 64, kernel_size=(1, 16), stride=(1, 4))
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(dropout)
        self.seq_len = (n_times - 16) // 4 + 1
        d_model = 64
        self.d_model = d_model
        # Block 3: transformer (configurable for --fast: 1 layer, 2 heads)
        self.pos_scale = 1.0
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop3 = nn.Dropout(dropout)
        # Block 4: classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 64)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def _sinusoidal_encoding(self, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model, device=device)
        for i in range(seq_len):
            for j in range(0, d_model, 2):
                pe[i, j] = math.sin(i / 10000 ** (j / d_model))
                if j + 1 < d_model:
                    pe[i, j + 1] = math.cos(i / 10000 ** (j / d_model))
        return pe * self.pos_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 750)
        x = self.spatial_conv(x)   # (B, 32, 1, 750)
        x = self.bn1(x)
        x = torch.nn.functional.elu(x)
        x = self.drop1(x)

        x = self.temporal_conv(x)  # (B, 64, 1, 184)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)
        x = self.drop2(x)

        # (B, 64, 184) -> (B, 184, 64)
        x = x.squeeze(2).permute(0, 2, 1)
        seq_len = x.size(1)
        pe = self._sinusoidal_encoding(seq_len, self.d_model, x.device)
        x = x + pe.unsqueeze(0)
        x = self.transformer(x)
        x = self.drop3(x)

        # GAP over time: (B, 184, 64) -> (B, 64)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        x = torch.nn.functional.elu(x)
        x = self.drop4(x)
        x = self.fc2(x)
        return x

    def get_spatial_conv_weights(self) -> torch.Tensor:
        """Return (32, 32) spatial conv weights for topomap: out_channels x in_channels."""
        w = self.spatial_conv.weight  # (32, 1, 32, 1)
        return w.squeeze(-1).squeeze(1)  # (32, 32)


def get_spatial_weights_for_topomap(model: EEGTransformer):
    """
    Extract per-input-channel importance from Block 1 spatial conv.
    Sum absolute weights over the 32 output filters -> (32,) importance per channel.
    """
    w = model.get_spatial_conv_weights()  # (32, 32)
    if w is None:
        return None
    importance = w.abs().sum(dim=0)  # (32,)
    return importance.detach().cpu().numpy()
