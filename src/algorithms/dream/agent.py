from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .networks import RegretNet, QNet, AverageNet
from .buffers import (
    RegretBuffer, RegretSample,   # NOTE: RegretSample must have fields: obs, mask, adv, iter_t
    QBuffer, QTransition,
    PolicyBuffer, PolicySample
)

class DreamAgent:
    """
    Model-free deep regret minimization agent with advantage baselines (DREAM-style).
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-4,
        lr_q: float = 1e-4,
        lr_avg: float = 1e-4,
        device: str = "cpu",
        max_grad_norm: float = 5.0,
        hidden: int = 256,
        layers: int = 2,
        adv_clip: float = 6.0,
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
        self.use_avg_net = False

        # Q targets & regularization (kept from your code)
        self.q_target_clip = 2.0
        self.q_output_l2 = 3e-4

        # --- Networks ---
        self.regret_net = RegretNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)
        self.q_net = QNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)
        self.q_target_net = deepcopy(self.q_net).to(self.device).eval()
        self.avg_net = AverageNet(obs_dim, act_dim, hidden=hidden, layers=layers).to(self.device)

        # --- Optimizers ---
        self.regret_opt = torch.optim.Adam(self.regret_net.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.avg_opt = torch.optim.Adam(self.avg_net.parameters(), lr=lr_avg)

        # --- Buffers ---
        self.buffer = RegretBuffer()
        self.q_buffer = QBuffer()
        self.policy_buffer = PolicyBuffer()

        # --- Other ---
        self.iter_count: int = 1

        self.log_every = 100
        self._q_log_step = 0
        self._q_ema = {"td": None, "q": None, "tgt": None}

    # ------------------------------------------------------------
    # Policy utilities
    # ------------------------------------------------------------
    def policy(self, obs, mask, use_average=True):
        if not getattr(self, "use_avg_net", False):
            return self._regret_matching(obs, mask)
        if use_average:
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                pi = self.avg_net(obs_t, mask_t).squeeze(0).cpu().numpy()

            # Normalize over legal actions defensively
            legal = np.where(mask > 0)[0]
            out = np.zeros_like(pi, dtype=np.float32)
            s = float(pi[legal].sum())
            if s > 1e-12 and len(legal) > 0:
                out[legal] = pi[legal] / s
            elif len(legal) > 0:
                out[legal] = 1.0 / len(legal)
            return out

        return self._regret_matching(obs, mask)

    def act(self, obs: np.ndarray, mask: np.ndarray, use_average: bool = True) -> int:
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0
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
    def outcome_sampling_traj(self, env, player_i: int, opponent=None,
                            strict_on_policy: bool = True) -> float:
        """
        Outcome-sampling rollout that:
        (1) builds DREAM per-iteration advantages (Problem A),
        (2) stores TD transitions for the Q baseline (Problem C),
        (3) fixes reach / importance weights (Problem D).

        If strict_on_policy=True (default), opponents & chance are sampled from their
        target distributions → IS ratio = 1, so no variance blow-up and no brittle constants.
        If you pass strict_on_policy=False, we compute w(I) = target_reach / sampling_reach
        over opponent & chance to the traverser infoset I and weight Â by w(I).
        """
        # ---------- tiny helpers ----------
        def rm_policy(o, m):
            pi = self._regret_matching(o, m)
            legal = (m > 0)
            s = float(pi[legal].sum())
            if s <= 0:
                pi = legal.astype(np.float32) / max(1, int(legal.sum()))
            else:
                pi[~legal] = 0.0
                pi /= s
            return pi

        def opp_target_policy(o, m):
            # "target" opponent policy (what CFR evaluates against).
            if opponent is not None:
                return opponent.policy(o, m, use_average=getattr(opponent, "use_avg_net", False))
            return self.policy(o, m, use_average=getattr(self, "use_avg_net", False))

        def opp_behavior_policy(o, m):
            # If strict_on_policy, behavior == target (keeps IS=1). Otherwise, you
            # may define a different behavior policy here (e.g., ε-greedy).
            if strict_on_policy:
                return opp_target_policy(o, m)
            # Example off-policy behavior: same as target by default; customize if needed.
            return opp_target_policy(o, m)

        def chance_prob(prev_obs, action, next_obs):
            """
            Returns (p_target, p_behavior) for the chance move. If your env provides
            exact chance probabilities, plug them here. Otherwise, return (1.0, 1.0).
            """
            # Try a few common env hooks:
            if hasattr(env, "chance_prob"):
                p = float(env.chance_prob(prev_obs, action, next_obs))
                return (p, p) if strict_on_policy else (p, p)  # adjust if behavior differs
            if hasattr(env, "last_chance_prob"):
                p = float(env.last_chance_prob())
                return (p, p) if strict_on_policy else (p, p)
            # Fallback: unknown chance → assume matched target/behavior
            return (1.0, 1.0)

        def is_chance_turn():
            # Generic detector for chance node; adapt if your env exposes it differently.
            # Many poker envs fold chance into "opponent" turns, in which case leave False.
            return getattr(env, "current", None) == -1 or getattr(env, "is_chance", False)

        # ---------- rollout ----------
        env.reset()
        obs, mask, done = env.last()

        # For IS: reach trackers up to the current node (opponents × chance only)
        reach_sampling = 1.0
        reach_target   = 1.0

        # We’ll keep our decisions for later (to build advantages) and store
        # the w(I) that applies at each infoset.
        decisions = []

        # Also collect Q transitions online (Problem C)
        while not done:
            if env.current == player_i:
                # ---- traverser infoset I ----
                obs_here, mask_here = obs.copy(), mask.copy()
                pi_here = rm_policy(obs_here, mask_here)

                # Importance weight for this infoset I (opponents × chance to I)
                linear_w = float(reach_target) * float(self.iter_count)

                if getattr(self, "use_avg_net", False):
                    self.policy_buffer.push(PolicySample(
                        obs=obs_here.astype(np.float32),
                        mask=mask_here.astype(np.float32),
                        pi=pi_here.astype(np.float32),   # current RM policy
                        weight=linear_w,
                    ))

                # Sample our action from current policy (no epsilon)
                a = int(np.random.choice(np.where(mask_here > 0)[0], p=pi_here[mask_here > 0]))

                # Execute our action
                prev_obs = obs.copy()
                env.step(a)
                obs, mask, done = env.last()

                # Roll forward opponents/chance to next traverser decision or terminal
                while not done and env.current != player_i:
                    if is_chance_turn():
                        # CHANCE: sample from its (behavior) distribution; update reach
                        # We don't know the distribution explicitly, so we assume the env
                        # samples an outcome. If env exposes prob for the sampled outcome,
                        # chance_prob(prev_state, action, next_state) should return it.
                        # We need the realized action to get its prob; many envs don't expose
                        # "chance action id". If not available, we assume matched target/behavior.
                        p_t, p_b = chance_prob(prev_obs, None, obs)  # action unknown → best effort
                        reach_sampling *= float(p_b)
                        reach_target   *= float(p_t)
                        # advance already happened above when env sampled chance into `obs`
                        pass
                    else:
                        # OPPONENT: compute behavior and target policy over legal actions
                        pi_beh = opp_behavior_policy(obs, mask)
                        pi_tgt = opp_target_policy(obs, mask)
                        legal = np.where(mask > 0)[0]
                        a_opp = int(np.random.choice(legal, p=pi_beh[legal]))
                        # update reach
                        reach_sampling *= float(pi_beh[a_opp])
                        reach_target   *= float(pi_tgt[a_opp])
                        # step
                        prev_obs = obs.copy()
                        env.step(a_opp)
                        obs, mask, done = env.last()

                # ---- push Q transition (Expected SARSA) ----
                if done:
                    r0, r1 = env.get_rewards()
                    payoff = float([r0, r1][player_i])
                    r_immediate = payoff
                    next_obs  = np.zeros_like(obs,  dtype=np.float32)
                    next_mask = np.zeros_like(mask, dtype=np.float32)
                    pi_next   = np.zeros_like(mask, dtype=np.float32)
                else:
                    r_immediate = 0.0
                    next_obs  = obs.astype(np.float32)
                    next_mask = mask.astype(np.float32)
                    pi_next   = rm_policy(obs, mask).astype(np.float32)

                self.q_buffer.push(QTransition(
                    obs=obs_here.astype(np.float32),
                    action=int(a),
                    next_obs=next_obs,
                    done=bool(done),
                    next_mask=next_mask,
                    pi_next=pi_next,
                    ret_g=float(r_immediate),
                    iter_t=self.iter_count,
                ))
                w_IS = 1.0 if strict_on_policy else (reach_target / max(reach_sampling, 1e-12))
                if not strict_on_policy:
                    pa = float(pi_here[a])                     # mu(a|I)
                    w_IS *= 1.0 / max(pa, 1e-12)               # multiply by 1/mu(a|I)
                decisions.append((obs_here, mask_here, a, pi_here, float(w_IS)))

            else:
                # Opponent step at the very beginning (rare but possible)
                pi_beh = opp_behavior_policy(obs, mask)
                pi_tgt = opp_target_policy(obs, mask)
                legal = np.where(mask > 0)[0]
                a_opp = int(np.random.choice(legal, p=pi_beh[legal]))
                reach_sampling *= float(pi_beh[a_opp])
                reach_target   *= float(pi_tgt[a_opp])
                prev_obs = obs.copy()
                env.step(a_opp)
                obs, mask, done = env.last()

        # terminal payoff for advantages
        r0, r1 = env.get_rewards()
        payoff = float([r0, r1][player_i])

        # ---------- build DREAM advantages (Problem A) with IS weight ----------
        returns = payoff
        for (obs_s, mask_s, a_s, pi_s, w_I) in reversed(decisions):
            # Q baseline
            obs_t = torch.from_numpy(obs_s.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                q_vals = self.q_net(obs_t).squeeze(0).cpu().numpy()

            X = q_vals.copy()
            X[a_s] = returns
            X_minus_B = X - q_vals
            center = float((pi_s * X_minus_B * mask_s).sum())
            hatA = (X_minus_B - center) * mask_s

            # Apply IS weight if off-policy; otherwise this is a no-op (=1)
            hatA *= float(w_I)

            if self.adv_clip is not None:
                hatA = np.clip(hatA, -self.adv_clip, self.adv_clip)

            self.buffer.push(RegretSample(
                obs=obs_s.astype(np.float32),
                mask=mask_s.astype(np.float32),
                adv=hatA.astype(np.float32),
                iter_t=self.iter_count,
            ))

        return payoff


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def train_step(self, batch_size: int = 2048):
        metrics = {"regret_loss": 0.0, "q_loss": 0.0}

        # ---------------- Q-network (unchanged for A-fix) ----------------
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
                if self.q_target_clip is not None:
                    target = target.clamp(-self.q_target_clip, self.q_target_clip)

            q_loss = F.smooth_l1_loss(q_sa, target)
            if self.q_output_l2 and self.q_output_l2 > 0.0:
                q_loss = q_loss + self.q_output_l2 * (q_sa.pow(2).mean())
            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_opt.step()
            metrics["q_loss"] = q_loss.item()

            # soft target update
            for p, t in zip(self.q_net.parameters(), self.q_target_net.parameters()):
                t.data.mul_(1 - self.q_target_tau).add_(self.q_target_tau * p.data)

        # ---------------- Advantage/Regret network (A-fix) ----------------
        if len(self.buffer) > 0:
            batch = self.buffer.sample(batch_size)
            obs  = torch.from_numpy(np.stack([b.obs  for b in batch]).astype(np.float32)).to(self.device)         # [B, obs_dim]
            mask = torch.from_numpy(np.stack([b.mask for b in batch]).astype(np.float32)).to(self.device)         # [B, A]
            targ = torch.from_numpy(np.stack([b.adv  for b in batch]).astype(np.float32)).to(self.device)         # [B, A]
            iters = torch.tensor([b.iter_t for b in batch], dtype=torch.float32, device=self.device)              # [B]

            pred = self.regret_net(obs)                                                                           # [B, A]

            # mask illegal actions
            pred = pred * mask
            targ = targ * mask

            # Linear-CFR iteration weights (normalize for stability)
            w = iters / (iters.mean().clamp_min(1.0))
            w = w.view(-1, 1)                                                                                    # [B,1]

            loss = ((pred - targ) ** 2)
            loss = (loss * w).sum(dim=1).mean()

            self.regret_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.regret_net.parameters(), self.max_grad_norm)
            self.regret_opt.step()

            metrics["regret_loss"] = float(loss.item())

        # ---------------- Average policy net (optional DREAM variant) ----------------
        if getattr(self, "use_avg_net", False) and hasattr(self, "policy_buffer") and len(self.policy_buffer) > 0:
            pol_batch = self.policy_buffer.sample(batch_size)
            obs_tensor  = torch.from_numpy(np.stack([b.obs  for b in pol_batch]).astype(np.float32)).to(self.device)
            mask_tensor = torch.from_numpy(np.stack([b.mask for b in pol_batch]).astype(np.float32)).to(self.device)
            target_pi   = torch.from_numpy(np.stack([b.pi   for b in pol_batch]).astype(np.float32)).to(self.device)
            weights     = torch.tensor([b.weight for b in pol_batch], dtype=torch.float32, device=self.device)

            pred_pi = self.avg_net(obs_tensor, mask_tensor).clamp_min(1e-8)
            ce = -(target_pi * pred_pi.log()).sum(dim=1)
            w = weights / (weights.sum() + 1e-8)
            avg_policy_loss = (w * ce).sum()

            self.avg_opt.zero_grad(set_to_none=True)
            avg_policy_loss.backward()
            nn.utils.clip_grad_norm_(self.avg_net.parameters(), self.max_grad_norm)
            self.avg_opt.step()
            metrics["avg_loss"] = float(avg_policy_loss.item())
        else:
            metrics["avg_loss"] = 0.0

        #print(f"[dbg] avg_buf_len={len(self.policy_buffer)} avg_loss={metrics.get('avg_loss', 0):.6f}")
        metrics["loss"] = metrics["regret_loss"] + metrics["q_loss"] + metrics["avg_loss"]
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

    def _regret_matching(self, obs, mask):
        """
        Regret matching on **predicted advantages** (DREAM):
        π(a) ∝ max(Â(a), 0); if all ≤ 0, uniform over legal.
        """
        o = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            adv = self.regret_net(o).squeeze(0).cpu().numpy()

        adv = adv * mask
        positive = np.maximum(adv, 0.0)

        # tiny prior to avoid zero-prob sinks
        eta = 5e-3 / np.sqrt(max(1, self.iter_count))
        prior = (mask > 0).astype(np.float32)

        dist = positive + eta * prior
        dist *= prior
        s = dist.sum()
        if s <= 0:
            n = prior.sum()
            dist = prior / n if n > 0 else prior
        else:
            dist /= s
        return dist.astype(np.float32)

    def _ema(self, prev, val, beta=0.9):
        return val if prev is None else beta * prev + (1 - beta) * val
