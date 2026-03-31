"""Baseline MLP model for IV surface reconstruction."""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple MLP baseline for dense surface reconstruction from sparse inputs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: define architecture
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError
