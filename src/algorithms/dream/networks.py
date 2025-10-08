from __future__ import annotations
import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        modules = []
        last = in_dim
        for _ in range(layers):
            modules += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden
        modules += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*modules)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AdvantageNet(nn.Module):
    """Predicts per-action advantages A(obs)[a]."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.body = MLP(obs_dim, hidden, hidden=hidden, layers=layers)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.body(obs)
        return self.head(z)  # [B, A]

class BaselineNet(nn.Module):
    """Scalar baseline b(obs) to reduce variance of advantage targets."""
    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden=hidden, layers=layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # [B]
