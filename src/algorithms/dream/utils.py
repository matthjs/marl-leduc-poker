from __future__ import annotations
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_masked(dist: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = (dist * mask).astype(np.float32)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        legal = np.where(mask > 0)[0]
        out = np.zeros_like(p, dtype=np.float32)
        if len(legal) > 0:
            out[legal] = 1.0 / len(legal)
        return out
    return p / s
