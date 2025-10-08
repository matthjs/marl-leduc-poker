from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class AdvantageSample:
    obs: np.ndarray     # infoset observation
    mask: np.ndarray    # legal mask (0/1)
    action: int         # chosen action
    ret_g: float        # terminal payoff G from acting player's perspective

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
