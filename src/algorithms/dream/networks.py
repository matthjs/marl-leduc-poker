from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class QNet(nn.Module):
    # ... (QNet class remains unchanged)
    """Predicts per-action Q-values Q(obs)[a]. Used as a variance-reduction baseline."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.body = MLP(obs_dim, hidden, hidden=hidden, layers=layers)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.body(obs)
        return self.head(z) # [B, A]


class AverageNet(nn.Module):
    """Predicts the average policy π̄(a|obs)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.body = MLP(obs_dim, hidden, hidden=hidden, layers=layers)
        self.head = nn.Linear(hidden, act_dim)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns probabilities over actions. If a row's mask has no legal actions,
        we fall back to a uniform distribution for that row.
        """
        logits = self.head(self.body(obs))  # [B, A]

        if mask is not None:
            # ensure boolean mask where True = illegal
            if mask.dtype != torch.bool:
                illegal = (mask <= 0)
            else:
                illegal = ~mask
            logits = logits.masked_fill(illegal, -1e9)

        probs = F.softmax(logits, dim=-1)  # [B, A]

        # Handle rows with no legal actions (all mask==0) -> uniform
        if mask is not None:
            no_legal = (mask.sum(dim=1, keepdim=True) == 0)  # [B,1]
            if no_legal.any():
                uniform = torch.full_like(probs, 1.0 / self.act_dim)
                probs = torch.where(no_legal, uniform, probs)

        return probs  # [B, A]
