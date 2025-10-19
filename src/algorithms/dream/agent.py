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
        self.q_target_clip = 2.0    
        self.q_output_l2 = 3e-4     

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
        self.iter_count: int = 1


        self.log_every = 100          # print diagnostics every N train_step() calls
        self._q_log_step = 0          # internal counter for printing
        self._q_ema = {"td": None, "q": None, "tgt": None}  # EMAs for smoother logs


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
            return pi
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
    def outcome_sampling_traj(self, env, player_i: int, opponent=None) -> float:
        """
        Hybrid MC/TD counterfactual rollout (Option B) WITHOUT epsilon-greedy:
        - No ε-mixing anywhere (opponent or self)
        - Behavior policy == target policy at our decisions → IS weight ρ = 1
        - Backward MC processing; Q-net for counterfactuals
        """
        env.reset()
        obs, mask, done = env.last()
        opp_reach = 1.0

        prob_private = 1.0 / (6.0 * 5.0)
        prob_public  = 1.0 / 4.0
        traj = []
        q_transitions = []

        # ---------------- rollout phase ----------------
        while not done:
            p = env.current
            stage = env.stage
            obs_here, mask_here = obs.copy(), mask.copy()

            if p == player_i:
                # Our move: sample from regret-matching policy (no epsilon)
                a = self.act(obs, mask, use_average=False)
                rm = self._regret_matching(obs_here, mask_here)
                chance_w = prob_private if stage == 0 else prob_private * prob_public

                traj.append({
                    "obs": obs_here,
                    "mask": mask_here,
                    "action": a,
                    "opp_reach": opp_reach,
                    "chance_reach": chance_w,
                    "stage": stage,
                    "policy": rm,
                })

                # Linear CFR weighting for average policy network
                if getattr(self, "use_avg_net", False):
                    linear_w = opp_reach * chance_w * self.iter_count
                    self.policy_buffer.push(PolicySample(
                        obs=obs_here.astype(np.float32),
                        mask=mask_here.astype(np.float32),
                        pi=rm,
                        weight=linear_w,
                    ))
            else:
                # Opponent move: sample from opponent's average policy (no epsilon)
                legal = np.where(mask > 0)[0]
                if opponent:
                    dist_no_eps = opponent.policy(obs, mask, use_average=getattr(opponent, "use_avg_net", False))
                else:
                    dist_no_eps = self.policy(obs, mask, use_average=getattr(self, "use_avg_net", False))
                a = int(np.random.choice(legal, p=dist_no_eps[legal]))
                opp_reach *= float(dist_no_eps[a])

            env.step(a)
            obs, mask, done = env.last()

        # terminal payoff
        r0, r1 = env.get_rewards()
        payoff = float([r0, r1][player_i])

        # ---------------- backward pass ----------------
        returns = payoff
        cumulative_rho = 1.0  # on-policy ⇒ stays 1.0

        for step in reversed(traj):
            obs_s, mask_s = step["obs"], step["mask"]
            a_s = step["action"]
            opp_w = step["opp_reach"]
            ch_w  = step["chance_reach"]

            # Q-values from network
            obs_t = torch.from_numpy(obs_s.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                q_vals = self.q_net(obs_t).squeeze(0).cpu().numpy()

            # (kept for completeness / future off-policy tweaks)
            pi_target = self._regret_matching(obs_s, mask_s)
            pi_behavior = pi_target
            rho_t = 1.0
            cumulative_rho *= rho_t  # remains 1.0

            q_played_mc = returns  # MC return for the played action

            v_baseline = (pi_target * q_vals * mask_s).sum()

            # Regret for all legal actions: Q(s,a) - Q(s,a_played), scaled by opp reach
            legal = np.where(mask_s > 0)[0]
            for a in legal:
                q_estimate = q_played_mc if a == a_s else q_vals[a]
                regret = (q_estimate - v_baseline) * cumulative_rho * opp_w * ch_w
                regret = float(np.clip(regret, -self.adv_clip, self.adv_clip))

                self.buffer.push(AdvantageSample(
                    obs_s.astype(np.float32),
                    mask_s.astype(np.float32),
                    int(a),
                    float(regret),
                    self.iter_count,
                ))

            # Q-learning transition for the played action (terminal from this infoset view)
            self.q_buffer.push(QTransition(
                obs=obs_s.astype(np.float32),
                action=int(a_s),
                next_obs=np.zeros_like(obs_s, dtype=np.float32),
                done=True,
                next_mask=np.zeros(self.act_dim, dtype=np.float32),
                pi_next=np.zeros(self.act_dim, dtype=np.float32),
                ret_g=q_played_mc,
                iter_t=self.iter_count,
            ))

            # For games with intermediate rewards, you'd do:
            # returns = step_reward + self.gamma * returns

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

            # ---------- Q-network diagnostics ----------
            with torch.no_grad():
                # Basic stats
                q_mean = q_sa.mean().item()
                q_std  = q_sa.std().item()
                tgt_mean = target.mean().item()
                tgt_std  = target.std().item()

                td = (q_sa - target)
                td_abs_mean = td.abs().mean().item()
                td_abs_max  = td.abs().max().item()

                # How many Qs fall in a plausible payoff range for Leduc ([-2, 2])?
                in_range = ((q_sa >= -2.0) & (q_sa <= 2.0)).float().mean().item()

                # Smooth EMAs for readability
                self._q_ema["td"]  = self._ema(self._q_ema["td"],  td_abs_mean)
                self._q_ema["q"]   = self._ema(self._q_ema["q"],   q_mean)
                self._q_ema["tgt"] = self._ema(self._q_ema["tgt"], tgt_mean)

                # Print every N steps
                self._q_log_step += 1
                tgt_in_range = ((target >= -2.0) & (target <= 2.0)).float().mean().item()
                if (self._q_log_step % self.log_every) == 0:
                    print(
                        "[Q] mean={:.3f}±{:.3f} (Q in[-2,2]={:.1f}%) | "
                        "target mean={:.3f}±{:.3f} (tgt in[-2,2]={:.1f}%) | "
                        "TD |mean|={:.3f} (EMA {:.3f}) max={:.3f}".format(
                            q_mean, q_std, 100.0 * in_range,
                            tgt_mean, tgt_std, 100.0 * tgt_in_range,
                            td_abs_mean, (self._q_ema['td'] or td_abs_mean), td_abs_max
                        )
                    )

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

            iter_weights = torch.tensor([b.iter_t for b in adv_batch], dtype=torch.float32, device=self.device)

            pred_regret = self.adv_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
            huber_loss = F.huber_loss(pred_regret, regret_targets, reduction='none')
            regret_loss = (iter_weights * huber_loss).sum() / (iter_weights.sum().clamp_min(1e-8))

            self.adv_opt.zero_grad(set_to_none=True)
            regret_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.adv_net.parameters(), self.max_grad_norm)
            if grad_norm > 10.0:
                print(f"[warn] adv grad_norm high: {float(grad_norm):.2f}")
            self.adv_opt.step()
            metrics["adv_loss"] = float(regret_loss.item())

        # Average policy network
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

    def _regret_matching(self, obs, mask):
        o = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            adv = self.adv_net(o).squeeze(0).cpu().numpy()


        pos = np.maximum(adv, 0.0)
        eta = 5e-3 / np.sqrt(max(1, self.iter_count))
        prior = (mask > 0).astype(np.float32)
        dist = pos + eta * prior  # Ensures exploration
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
    
    def _ema(self, prev, val, beta=0.9):
        return val if prev is None else beta * prev + (1 - beta) * val

