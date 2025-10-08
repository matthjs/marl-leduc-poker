from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import AdvantageNet, BaselineNet
from .buffers import AdvantageBuffer, AdvantageSample
from .utils import normalize_masked

class DreamAgent:
    """
    DREAM agent for two-player zero-sum Leduc Poker.
    - Advantage network A_theta(obs)[a]
    - Baseline network b_phi(obs)
    - Outcome-sampling CFR rollouts: target A = G - b(obs)
    - Policy = regret-matching over positive advantages
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        device: str = "cpu",
        advantage_coeff: float = 1.0,
        max_grad_norm: float = 5.0,
        hidden: int = 256,
        layers: int = 2,
        adv_clip: float = 5.0,   # clip for (G - b)
    ):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.adv_clip = float(adv_clip)

        self.adv_net = AdvantageNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)
        self.baseline = BaselineNet(obs_dim, hidden=hidden, layers=layers).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.adv_net.parameters()) + list(self.baseline.parameters()),
            lr=lr,
        )

        self.buffer = AdvantageBuffer()
        self.advantage_coeff = advantage_coeff
        self.max_grad_norm = max_grad_norm
        self.eps = 0.10  # ε-greedy exploration; decay it in your main loop

        # Average-policy tables
        self.strategy_sum: Dict[bytes, np.ndarray] = {}
        self.strategy_visits: Dict[bytes, int] = {}

    # ---------- Acting ----------

    def policy(self, obs: np.ndarray, mask: np.ndarray, use_average: bool = True) -> np.ndarray:
        """Return a prob. dist. over actions for one observation."""
        key = obs.tobytes()
        if use_average and key in self.strategy_sum:
            s = self.strategy_sum[key]
            n = max(1, self.strategy_visits.get(key, 1))
            dist = s / float(n)
            return normalize_masked(dist, mask)
        # fallback: regret-matching from current net
        return self._rm_from_net(obs, mask)

    def act(self, obs: np.ndarray, mask: np.ndarray, use_average: bool = True) -> int:
        legal = np.where(mask > 0)[0]
        if np.random.rand() < self.eps and len(legal) > 0:
            return np.random.choice(legal)
        pi = self.policy(obs, mask, use_average=use_average)
        return np.random.choice(legal, p=pi[legal])

    # ---------- Rollouts ----------

    def outcome_sampling_traj(self, env, player_i: int, opponent: "DreamAgent" | None = None) -> float:
        """
        Generate one outcome-sampled trajectory and push (s, a, G) samples
        for states where player_i acted. Opponent uses its *average* policy.
        """
        env.reset()
        obs = env.get_observation()
        mask = env.get_mask()
        done = env.terminal
        traj: List[Tuple[np.ndarray, np.ndarray, int, int]] = []

        while not done:
            p = env.current
            if p == player_i:
                a = self.act(obs, mask, use_average=False)
            else:
                a = opponent.act(obs, mask, use_average=True) if opponent is not None \
                    else self.act(obs, mask, use_average=True)

            traj.append((obs.copy(), mask.copy(), p, a))
            _ = env.step(a)
            done = env.terminal
            if not done:
                obs = env.get_observation()
                mask = env.get_mask()

        r0, r1 = env.get_rewards()
        payoff = float([r0, r1][player_i])

        # Store samples and update avg policy tables
        for obs_s, mask_s, p_s, a_s in traj:
            if p_s != player_i:
                continue
            self.buffer.push(AdvantageSample(
                obs=obs_s,
                mask=mask_s.astype(np.float32),
                action=int(a_s),
                ret_g=payoff,
            ))
            rm = self._rm_from_net(obs_s, mask_s)
            key = obs_s.tobytes()
            self.strategy_sum[key] = self.strategy_sum.get(key, np.zeros(self.act_dim, dtype=np.float32)) + rm
            self.strategy_visits[key] = self.strategy_visits.get(key, 0) + 1

        return payoff

    # ---------- Training ----------

    def train_step(self, batch_size: int = 2048):
        if len(self.buffer) == 0:
            return {"loss": 0.0, "adv_mse": 0.0, "baseline_mse": 0.0}

        batch = self.buffer.sample(batch_size)
        obs  = torch.tensor(np.stack([b.obs    for b in batch], axis=0), dtype=torch.float32, device=self.device)
        acts = torch.tensor(np.array([b.action for b in batch]),        dtype=torch.long,    device=self.device)
        rets = torch.tensor(np.array([b.ret_g  for b in batch]),        dtype=torch.float32, device=self.device)

        # Baseline learns to predict the return G
        pred_b = self.baseline(obs)  # [B]
        baseline_loss = F.smooth_l1_loss(pred_b, rets)

        # Advantage target uses a detached baseline
        with torch.no_grad():
            adv_target = (rets - pred_b.detach()).clamp(-self.adv_clip, self.adv_clip)  # [B]

        pred_adv_all    = self.adv_net(obs)                               # [B, A]
        pred_adv_chosen = pred_adv_all.gather(1, acts.unsqueeze(1)).squeeze(1)  # [B]
        adv_loss = F.smooth_l1_loss(pred_adv_chosen, adv_target)

        loss = self.advantage_coeff * adv_loss + baseline_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.adv_net.parameters()) + list(self.baseline.parameters()),
            self.max_grad_norm,
        )
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "adv_mse": float(adv_loss.item()),
            "baseline_mse": float(baseline_loss.item()),
        }

    # ---------- Internals ----------

    def _rm_from_net(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Regret-matching over positive advantages from the net."""
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            adv = self.adv_net(o).squeeze(0).cpu().numpy()  # [A]
        pos = np.maximum(adv, 0.0).astype(np.float32)
        return normalize_masked(pos, mask)
