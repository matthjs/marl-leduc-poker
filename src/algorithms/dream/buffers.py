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
    """
    Per-iteration, baseline-adjusted advantage target for a single infoset.
    adv is a full action-vector (masked on illegal actions), not a scalar.
    """
    obs: np.ndarray      # [obs_dim], float32
    mask: np.ndarray     # [act_dim], float32 in {0,1}
    adv: np.ndarray      # [act_dim], float32 (baseline-adjusted, centered)
    iter_t: int          # iteration index for Linear-CFR weighting

@dataclass
class QTransition:
    """
    Stores a single transition for training the Q-network baseline.
    Note: For the Problem A fix we still allow terminal-only samples,
    but you'll likely switch to real next states when you do Problem C.
    """
    obs: np.ndarray          # [obs_dim], float32
    action: int
    next_obs: np.ndarray     # [obs_dim], float32
    done: bool
    next_mask: np.ndarray    # [act_dim], float32 in {0,1}
    pi_next: np.ndarray      # [act_dim], float32, on-policy action probs at next state
    ret_g: float             # scalar final payoff (or bootstrapped return)
    iter_t: int              # iteration index for Linear-CFR weighting

@dataclass
class PolicySample:
    """
    Supervision sample for the average policy network (optional DREAM variant).
    """
    obs: np.ndarray      # [obs_dim], float32
    mask: np.ndarray     # [act_dim], float32 in {0,1}
    pi: np.ndarray       # [act_dim], float32, sums to 1 on legal actions
    weight: float        # scalar Linear-CFR weight

# -----------------------------
# Buffers
# -----------------------------

class RegretBuffer:
    """
    Ring buffer for per-iteration advantage vectors.
    """
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[RegretSample] = []
        self.idx = 0

    def push(self, sample: RegretSample):
        # Optional lightweight sanity checks (fail fast during debugging)
        # Ensure dtype/shape consistency to avoid silent broadcasting later.
        if not isinstance(sample.adv, np.ndarray):
            raise TypeError("RegretSample.adv must be a numpy array")
        if not isinstance(sample.mask, np.ndarray):
            raise TypeError("RegretSample.mask must be a numpy array")
        if sample.adv.shape != sample.mask.shape:
            raise ValueError(f"adv shape {sample.adv.shape} must match mask shape {sample.mask.shape}")

        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.idx] = sample
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[RegretSample]:
        if len(self.data) == 0:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

class QBuffer:
    """Replay buffer for Q-network transitions."""
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
        if len(self.data) == 0:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

class PolicyBuffer:
    """
    Reservoir-sampling buffer for average-policy supervision.
    Keeps a uniform sample over all pushed items.
    """
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[PolicySample] = []
        self.n_seen = 0

    def __len__(self):
        return len(self.data)

    def push(self, s: PolicySample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(s)
        else:
            # Reservoir sampling: replace with prob = capacity / n_seen
            j = random.randint(0, self.n_seen - 1)
            if j < self.capacity:
                self.data[j] = s

    def sample(self, batch_size: int) -> List[PolicySample]:
        if len(self.data) == 0:
            return []
        return random.sample(self.data, k=min(batch_size, len(self.data)))
