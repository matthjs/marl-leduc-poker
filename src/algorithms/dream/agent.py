from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


from .networks import AdvantageNet, QNet, AverageNet
from .buffers import (
    AdvantageBuffer, AdvantageSample,
    QBuffer, QTransition,
    PolicyBuffer, PolicySample
)


class DreamAgent:
    """
    Model-free deep regret minimization agent with advantage baselines.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-4,
        lr_q: float = 1e-3,
        lr_avg: float = 1e-3,
        device: str = "cpu",
        max_grad_norm: float = 5.0,
        hidden: int = 256,
        layers: int = 2,
        adv_clip: float = 8.0,
        q_target_tau: float = 0.01,
        gamma: float = 1.0,
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_grad_norm = max_grad_norm
        self.adv_clip = adv_clip
        self.q_target_tau = q_target_tau
        self.gamma = gamma

        # --- Regret accumulation memory ---
        self.regret_sum: Dict[bytes, np.ndarray] = {}

        # --- Networks ---
        self.adv_net = AdvantageNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)
        self.q_net = QNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)
        self.q_target_net = deepcopy(self.q_net).to(self.device).eval()
        self.avg_net = AverageNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)

        # --- Optimizers ---
        self.adv_opt = torch.optim.Adam(self.adv_net.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.avg_opt = torch.optim.Adam(self.avg_net.parameters(), lr=lr_avg)

        # --- Buffers ---
        self.buffer = AdvantageBuffer()
        self.q_buffer = QBuffer()
        self.policy_buffer = PolicyBuffer()

        # --- Other ---
        self.eps = 0.10
        self.iter_count: int = 1

    # ------------------------------------------------------------
    # Policy utilities
    # ------------------------------------------------------------
    def policy(self, obs, mask, use_average=True):
        if use_average:
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                pi = self.avg_net(obs_t, mask_t).squeeze(0).cpu().numpy()
            return pi
        return self._rm_from_net(obs, mask)

    def act(self, obs: np.ndarray, mask: np.ndarray, use_average: bool = True) -> int:
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0
        if np.random.rand() < self.eps:
            return np.random.choice(legal)
        pi = self.policy(obs, mask, use_average=use_average)
        return np.random.choice(legal, p=pi[legal])

    def _get_q_value(self, obs: np.ndarray, action: int) -> float:
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            q_vals = self.q_net(obs_t).squeeze(0).cpu().numpy()
        return float(q_vals[action])

    # ------------------------------------------------------------
    # Outcome sampling trajectory (real environment)
    # ------------------------------------------------------------
    def outcome_sampling_traj(self, env, player_i: int, opponent: "DreamAgent" | None = None) -> float:
        """
        Collect one real-environment trajectory and compute model-free advantage targets.
        """
        env.reset()
        obs, mask, done = env.last()
        opp_reach = 1.0

        prob_private_deal = 1.0 / (6.0 * 5.0)
        prob_public_deal  = 1.0 / 4.0

        adv_traj = []          # (obs_s, mask_s, p_s, a_s, opp_reach, stage_s)
        q_trans_partial = []   # (obs, a, next_obs, done, next_mask, reward)

        while not done:
            p = env.current
            stage = env.stage
            obs_here  = obs.copy()
            mask_here = mask.copy()

            if p == player_i:
                a = self.act(obs, mask, use_average=False)
                rm = self._rm_from_net(obs_here, mask_here)
                chance_w = (prob_private_deal if stage == 0 else prob_private_deal * prob_public_deal)
                base_w   = float(opp_reach * chance_w)
                linear_cfr_weight = base_w * self.iter_count
                self.policy_buffer.push(PolicySample(
                    obs=obs_here.astype(np.float32),
                    mask=mask_here.astype(np.float32),
                    pi=rm,
                    weight=linear_cfr_weight
                ))
            else:
                eps   = opponent.eps if opponent is not None else self.eps
                legal = np.where(mask > 0)[0]
                dist_no_eps = (opponent.policy(obs, mask, use_average=True)
                               if opponent else self.policy(obs, mask, use_average=True))
                dist = dist_no_eps.copy()
                if eps > 0 and len(legal) > 0:
                    unif = np.zeros_like(dist); unif[legal] = 1.0 / len(legal)
                    dist = (1.0 - eps) * dist + eps * unif
                a = int(np.random.choice(legal, p=dist[legal]))
                opp_reach *= float(dist_no_eps[a])

            if p == player_i:
                adv_traj.append((obs_here, mask_here, p, a, opp_reach, stage))

            env.step(a)
            next_obs, next_mask, done = env.last()
            next_player = env.current
            r0, r1 = env.get_rewards()
            reward = float([r0, r1][player_i])

            q_trans_partial.append((obs_here, a, next_obs, done, next_mask, reward))

            if not done:
                obs, mask, _ = env.last()

        r0, r1 = env.get_rewards()
        payoff = float([r0, r1][player_i])

        # -------- compute advantages using real environment transitions --------
        for obs_s, mask_s, p_s, a_s, opp_w, stage_s in adv_traj:
            q_sa = self._get_q_value(obs_s, a_s)

            # find the corresponding transition
            match = next((t for t in q_trans_partial if np.allclose(t[0], obs_s) and t[1] == a_s), None)
            if match is not None:
                _, _, next_obs, done_t, next_mask, reward = match
                if not done_t:
                    with torch.inference_mode():
                        q_next = self.q_net(torch.from_numpy(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
                    pi_next = self.policy(next_obs, next_mask, use_average=True)
                    v_sampled = reward + self.gamma * np.sum(pi_next * q_next * next_mask)
                else:
                    v_sampled = reward
            else:
                v_sampled = 0.0

            advantage_target = v_sampled - q_sa
            advantage_target = np.clip(advantage_target, -self.adv_clip, self.adv_clip)

            # Importance weight (nearly 1 since model-free)
            pi_target   = self.policy(obs_s, mask_s, use_average=False)
            pi_behavior = pi_target.copy()
            legal = np.where(mask_s > 0)[0]
            if self.eps > 0:
                unif = np.zeros_like(pi_behavior)
                if len(legal) > 0:
                    unif[legal] = 1.0 / len(legal)
                pi_behavior = (1.0 - self.eps) * pi_behavior + self.eps * unif

            rho = 1.0
            if pi_behavior[a_s] > 1e-8:
                rho = np.clip(pi_target[a_s] / pi_behavior[a_s], 0.0, 5.0)
            advantage_target *= rho

            key = self._obs_key(obs_s)
            if key not in self.regret_sum:
                self.regret_sum[key] = np.zeros(self.act_dim, dtype=np.float32)
            self.regret_sum[key][a_s] = max(self.regret_sum[key][a_s] + advantage_target, 0.0)

            self.buffer.push(AdvantageSample(
                obs_s.astype(np.float32),
                mask_s.astype(np.float32),
                int(a_s),
                float(advantage_target),
                self.iter_count
            ))

        # Q-learning targets buffer
        for obs_t, act_t, next_obs_t, done_t, next_mask_t, reward_t in q_trans_partial:
            pi_next = np.zeros(self.act_dim, dtype=np.float32)
            if not done_t:
                pi_next = self.policy(next_obs_t, next_mask_t, use_average=True)
            self.q_buffer.push(QTransition(
                obs=obs_t.astype(np.float32),
                action=int(act_t),
                next_obs=next_obs_t.astype(np.float32) if not done_t else np.zeros_like(obs_t, dtype=np.float32),
                done=done_t,
                next_mask=next_mask_t.astype(np.float32) if not done_t else np.zeros(self.act_dim, dtype=np.float32),
                pi_next=pi_next,
                ret_g=payoff,
                iter_t=self.iter_count
            ))
        return payoff

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def train_step(self, batch_size: int = 2048):
        metrics = {"adv_loss": 0.0, "q_loss": 0.0}

        # Q-network
        if len(self.q_buffer) > 0:
            q_batch = self.q_buffer.sample(batch_size)
            obs = torch.from_numpy(np.stack([b.obs for b in q_batch]).astype(np.float32)).to(self.device)
            acts = torch.from_numpy(np.asarray([b.action for b in q_batch])).long().to(self.device)
            next_obs = torch.from_numpy(np.stack([b.next_obs for b in q_batch]).astype(np.float32)).to(self.device)
            dones = torch.from_numpy(np.asarray([b.done for b in q_batch])).bool().to(self.device)
            next_mask = torch.from_numpy(np.stack([b.next_mask for b in q_batch]).astype(np.float32)).to(self.device)
            pi_next = torch.from_numpy(np.stack([b.pi_next for b in q_batch]).astype(np.float32)).to(self.device)
            rets = torch.from_numpy(np.asarray([b.ret_g for b in q_batch]).astype(np.float32)).to(self.device)

            q_sa = self.q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next_all = self.q_target_net(next_obs) * next_mask
                v_next = (pi_next * q_next_all).sum(dim=1)
                target = torch.where(dones, rets, rets + self.gamma * v_next)

            q_loss = F.smooth_l1_loss(q_sa, target)
            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_opt.step()
            metrics["q_loss"] = q_loss.item()

            for p, t in zip(self.q_net.parameters(), self.q_target_net.parameters()):
                t.data.mul_(1 - self.q_target_tau).add_(self.q_target_tau * p.data)

        # Advantage/Regret network
        if len(self.buffer) > 0:
            adv_batch = self.buffer.sample(batch_size)
            obs  = torch.from_numpy(np.stack([b.obs for b in adv_batch]).astype(np.float32)).to(self.device)
            acts = torch.from_numpy(np.asarray([b.action for b in adv_batch])).long().to(self.device)
            target_vals = np.array(
                [getattr(b, "adv_target_v", getattr(b, "adv", 0.0)) for b in adv_batch],
                dtype=np.float32
            )
            regret_targets = torch.from_numpy(target_vals).to(self.device)
            if regret_targets.numel() > 1:
                rt_mean = regret_targets.mean()
                rt_std  = regret_targets.std().clamp_min(1e-3)
                regret_targets = (regret_targets - rt_mean) / rt_std

            pred_regret = self.adv_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
            regret_loss = F.huber_loss(pred_regret, regret_targets, delta=1.0)

            self.adv_opt.zero_grad(set_to_none=True)
            regret_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.adv_net.parameters(), self.max_grad_norm)
            if grad_norm > 10.0:
                print(f"[warn] adv grad_norm high: {float(grad_norm):.2f}")
            self.adv_opt.step()
            metrics["adv_loss"] = float(regret_loss.item())

        # Average policy network
        if hasattr(self, "policy_buffer") and len(self.policy_buffer) > 0:
            pol_batch = self.policy_buffer.sample(batch_size)
            obs_tensor  = torch.from_numpy(np.stack([b.obs  for b in pol_batch]).astype(np.float32)).to(self.device)
            mask_tensor = torch.from_numpy(np.stack([b.mask for b in pol_batch]).astype(np.float32)).to(self.device)
            target_pi   = torch.from_numpy(np.stack([b.pi   for b in pol_batch]).astype(np.float32)).to(self.device)
            weights     = torch.tensor([b.weight for b in pol_batch], dtype=torch.float32, device=self.device)

            pred_pi = self.avg_net(obs_tensor, mask_tensor).clamp_min(1e-8)
            ce = -(target_pi * pred_pi.log()).sum(dim=1)
            entropy = -(pred_pi * pred_pi.log()).sum(dim=1)
            w = weights / (weights.sum() + 1e-8)
            lambda_ent0 = 0.05
            lambda_ent = float(lambda_ent0 / np.sqrt(max(1, self.iter_count)))
            avg_policy_loss = (w * ce).sum() - lambda_ent * entropy.mean()

            self.avg_opt.zero_grad(set_to_none=True)
            avg_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_net.parameters(), self.max_grad_norm)
            self.avg_opt.step()
            metrics["avg_loss"] = float(avg_policy_loss.item())
        else:
            metrics["avg_loss"] = 0.0

        metrics["loss"] = metrics["adv_loss"] + metrics["q_loss"] + metrics["avg_loss"]
        return metrics

    # ------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------
    def increment_iteration(self):
        self.iter_count += 1

    def policy_from_avg_net(self, obs, mask):
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.avg_net(obs_t, mask_t).squeeze(0).cpu().numpy()

    def _rm_from_net(self, obs, mask):
        key = self._obs_key(obs)
        if key in self.regret_sum:
            regrets = self.regret_sum[key]
            pos = np.maximum(regrets, 0.0)
        else:
            o = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                adv = self.adv_net(o).squeeze(0).cpu().numpy()
            pos = np.maximum(adv, 0.0)

        eta = 1e-2 / np.sqrt(max(1, self.iter_count))
        prior = (mask > 0).astype(np.float32)
        dist = pos + eta * prior
        dist *= prior
        s = dist.sum()
        if s <= 0:
            n = prior.sum()
            dist = prior / n if n > 0 else prior
        else:
            dist /= s
        return dist.astype(np.float32)

    def _obs_key(self, obs: np.ndarray) -> bytes:
        return np.asarray(obs, dtype=np.float32).tobytes()
