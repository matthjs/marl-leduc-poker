from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List
import numpy as np
from collections import deque

@dataclass
class AdvantageSample:
    obs: np.ndarray     # infoset observation
    mask: np.ndarray    # legal mask (0/1)
    action: int         # chosen action
    adv_target_v: float # Pre-computed advantage target: (V_sampled - Q(s,a)) clipped
    iter_t: int         # Iteration count for Linear CFR weighting

@dataclass
class QTransition:
    # ... (QTransition remains unchanged)
    """Stores a single transition for training the Q-network baseline."""
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    done: bool
    next_mask: np.ndarray
    pi_next: np.ndarray  # On-policy action probabilities at the next state
    ret_g: float         # Final episode payoff for this player
    iter_t: int         # Iteration count for Linear CFR weighting

@dataclass
class PolicySample:
    obs: np.ndarray      # [obs_dim], float32
    mask: np.ndarray     # [act_dim], float32 in {0,1}
    pi: np.ndarray       # [act_dim], float32, sums to 1 on legal actions
    weight: float        # scalar, linear CFR weight

class AdvantageBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.data: List[AdvantageSample] = []
        self.idx = 0

    def push(self, sample: AdvantageSample):
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.idx] = sample
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[AdvantageSample]:
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)

class QBuffer:
    """Replay buffer for Q-network transitions."""
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.data: List[QTransition] = []
        self.idx = 0

    def push(self, sample: QTransition):
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            self.data[self.idx] = sample
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[QTransition]:
        return random.sample(self.data, k=min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)


class PolicyBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.data: list[PolicySample] = []
        self.n_seen = 0

    def __len__(self):
        return len(self.data)

    def push(self, s: PolicySample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(s)
        else:
            j = random.randint(0, self.n_seen - 1)
            if j < self.capacity:
                self.data[j] = s

    def sample(self, batch_size: int):
        batch = random.sample(self.data, k=min(batch_size, len(self.data)))
        return batch