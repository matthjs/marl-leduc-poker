from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List
import numpy as np

# -----------------------------
# Data records
# -----------------------------

@dataclass
class RegretSample:
    """Baseline-adjusted advantage vector for a single infoset at iteration iter_t."""
    obs: np.ndarray      # [obs_dim], float32
    mask: np.ndarray     # [act_dim], float32 in {0,1}
    adv: np.ndarray      # [act_dim], float32 (masked; centered advantages)
    iter_t: int          # iteration index for Linear-CFR weighting

@dataclass
class QTransition:
    """Transition for training the Q-network (Expected SARSA baseline)."""
    obs: np.ndarray          # [obs_dim], float32
    action: int
    next_obs: np.ndarray     # [obs_dim], float32
    done: bool
    next_mask: np.ndarray    # [act_dim], float32 in {0,1}
    pi_next: np.ndarray      # [act_dim], float32, on-policy probs at next state
    ret_g: float             # immediate return (terminal payoff) or bootstrapped return
    iter_t: int              # iteration index for optional weighting

@dataclass
class PolicySample:
    """Supervision sample for distilling the current RM policy into the AverageNet."""
    obs: np.ndarray      # [obs_dim], float32
    mask: np.ndarray     # [act_dim], float32 in {0,1}
    pi: np.ndarray       # [act_dim], float32, sums to 1 over legal actions
    weight: float        # Linear-CFR style weight

# -----------------------------
# Buffers
# -----------------------------

class RegretBuffer:
    """Ring buffer for per-infoset advantage vectors."""
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[RegretSample] = []
        self.idx = 0

    def push(self, sample: RegretSample):
        # Minimal shape checks
        if not isinstance(sample.adv, np.ndarray) or not isinstance(sample.mask, np.ndarray):
            raise TypeError("RegretSample.adv and mask must be numpy arrays")
        if sample.adv.shape != sample.mask.shape:
            raise ValueError(f"adv shape {sample.adv.shape} must match mask shape {sample.mask.shape}")

        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.idx] = sample
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[RegretSample]:
        if not self.data:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

class QBuffer:
    """Ring buffer for Q-network transitions."""
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[QTransition] = []
        self.idx = 0

    def push(self, sample: QTransition):
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.idx] = sample
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[QTransition]:
        if not self.data:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

class PolicyBuffer:
    """Reservoir-sampling buffer for average-policy supervision."""
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[PolicySample] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, s: PolicySample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(s)
        else:
            j = random.randint(0, self.n_seen - 1)
            if j < self.capacity:
                self.data[j] = s

    def sample(self, batch_size: int) -> List[PolicySample]:
        if not self.data:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))
