"""
Lightweight Neural Network Architectures for On-Device Anomaly Detection.
Engineered for edge microcontrollers and industrial gateways with strict compute/memory bounds:
- LightweightAutoencoder (MLP Dense Autoencoder: ~10,000 parameters, <50 KB FP32)
- Conv1DAutoencoder (Temporal 1D-Convolutional Autoencoder: ~8,000 parameters, <35 KB FP32)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class LightweightAutoencoder(nn.Module):
    """
    Compact Multi-Layer Perceptron Autoencoder for on-device reconstruction.
    Input: Flat window tensor [batch_size, window_size * num_sensors]
    Output: Reconstructed window tensor [batch_size, window_size * num_sensors]
    """

    def __init__(
        self,
        window_size: int = 16,
        input_dim: int = 8,
        hidden_dims: Tuple[int, int] = (32, 16),
        latent_dim: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.total_input_dim = window_size * input_dim
        self.latent_dim = latent_dim

        # Encoder: 128 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[1], latent_dim),
        )

        # Decoder: 8 -> 16 -> 32 -> 128
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[0], self.total_input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = x.view(x.size(0), -1)
        return self.encoder(flat_x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(-1, self.window_size, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = x.view(x.size(0), -1)
        z = self.encoder(flat_x)
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, self.window_size, self.input_dim)


class Conv1DAutoencoder(nn.Module):
    """
    Lightweight 1D-Convolutional Autoencoder for capturing temporal correlations across sensors.
    Input: Window tensor [batch_size, window_size, num_sensors]
    Permutes to [batch_size, num_sensors, window_size] for 1D convolution.
    """

    def __init__(
        self,
        window_size: int = 16,
        input_dim: int = 8,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: (B, 8, 16) -> (B, 16, 8) -> (B, 32, 4)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.fc_latent = nn.Linear(32 * (window_size // 4), latent_dim)
        self.fc_unlatent = nn.Linear(latent_dim, 32 * (window_size // 4))

        # Decoder: (B, 32, 4) -> (B, 16, 8) -> (B, 8, 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, input_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to (B, C, L): (B, input_dim, window_size)
        x_in = x.transpose(1, 2)
        h = self.encoder_conv(x_in)
        flat_h = h.view(h.size(0), -1)
        z = self.fc_latent(flat_h)

        h_rec = self.fc_unlatent(z).view(h.size(0), 32, self.window_size // 4)
        out = self.decoder_conv(h_rec)
        # Transpose back to (B, window_size, input_dim)
        return out.transpose(1, 2)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_kb(model: nn.Module) -> float:
    """Calculates model parameter memory footprint in Kilobytes (FP32)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024.0


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Factory method to construct model based on configuration."""
    arch = config.get("architecture", "LightweightAutoencoder")
    window_size = config.get("window_size", 16)
    input_dim = config.get("input_dim", 8)
    latent_dim = config.get("latent_dim", 8)
    hidden_dims = tuple(config.get("hidden_dims", [32, 16]))
    dropout = config.get("dropout", 0.05)

    if arch == "LightweightAutoencoder":
        return LightweightAutoencoder(
            window_size=window_size,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
    elif arch == "Conv1DAutoencoder":
        return Conv1DAutoencoder(
            window_size=window_size,
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

