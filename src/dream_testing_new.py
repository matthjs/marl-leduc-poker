from algorithms.dream.agent import DreamAgent
from environment.leduc_env import LeducEnv
import numpy as np
import torch
import copy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms.dream.sdcfr import SDCFROpponent, evaluate_sdcfr, evaluate_sdcfr_both_seats
from algorithms.dream.networks import RegretNet
import random
from copy import deepcopy


# ----------------------------
# Opponents for training/eval
# ----------------------------
class AvgPolicyOpponent:
    """
    Frozen copy of an agent's average policy network.
    Used for stable self-play data collection.
    """
    def __init__(self, src_agent):
        self.net = deepcopy(src_agent.avg_net).eval()
        self.device = next(self.net.parameters()).device

    @torch.inference_mode()
    def policy(self, obs, mask, use_average=True):
        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
        pi = self.net(obs_t, mask_t).squeeze(0).cpu().numpy()
        return pi

class SDCFRMeanPolicyOpponent:
    def __init__(self, sd: SDCFROpponent, decay=0.98):
        self.sd = sd
        # use the real fields from sdcfr.py
        self.has_snaps = hasattr(sd, "_snap_nets") and hasattr(sd, "_weights") and (len(sd._snap_nets) > 0)
        if self.has_snaps:
            self.nets = [net.eval() for net in sd._snap_nets]
            w = np.asarray(sd._weights, dtype=np.float64)
            if np.max(w) - np.min(w) < 1e-9:
                steps = np.arange(len(self.nets), dtype=np.float64)
                w = decay ** (len(self.nets) - 1 - steps)
            self.weights = w / max(1e-12, w.sum())
            self.device = next(self.nets[0].parameters()).device

    @torch.inference_mode()
    def policy(self, obs, mask, use_average=True):
        if not self.has_snaps:
            # fall back to SD-CFR’s per-episode snapshot sampling
            return self.sd.policy(obs, mask)

        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)

        acc = np.zeros_like(mask, dtype=np.float32)
        legal = (mask > 0).astype(np.float32)

        for w, net in zip(self.weights, self.nets):
            adv = net(obs_t).squeeze(0).cpu().numpy()     # advantages from RegretNet
            adv = adv * legal
            pos = np.maximum(adv, 0.0)
            # tiny prior to avoid zero sinks
            eta = 1e-3
            scores = pos + eta * legal                    # unnormalized RM scores
            s = float(scores[legal > 0].sum())
            if s > 1e-12 and legal.sum() > 0:
                pi_t = np.zeros_like(scores, dtype=np.float32)
                pi_t[legal > 0] = scores[legal > 0] / s
            elif legal.sum() > 0:
                pi_t = legal / float(int(legal.sum()))
            else:
                pi_t = np.zeros_like(scores, dtype=np.float32)

            acc += float(w) * pi_t

        # final safety renormalization on legal actions
        z = float(acc[legal > 0].sum())
        if z > 1e-12 and legal.sum() > 0:
            acc[legal > 0] /= z
        elif legal.sum() > 0:
            acc[legal > 0] = 1.0 / int(legal.sum())

        return acc


# ---------------
# Main Experiment
# ---------------
class Experiment:
    def __init__(self):
        # Initialize agents
        env = LeducEnv()
        obs_dim = env.get_observation().shape[0]
        act_dim = 4  # 0:Call, 1:Raise, 2:Fold, 3:Check (adjust if different)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.agent0 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)
        self.agent1 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)

        # Equalize init to reduce early drift
        self.agent1.regret_net.load_state_dict(self.agent0.regret_net.state_dict())
        self.agent1.q_net.load_state_dict(self.agent0.q_net.state_dict())

        # Use average policy nets during policy() calls for evaluation
        self.agent0.use_avg_net = True
        self.agent1.use_avg_net = True
        self._warmup_before_snapshots = 100

        def make_regret_net():
            return RegretNet(obs_dim, act_dim, hidden=256, layers=2)

        # SD-CFR snapshot managers (for evaluation only)
        self.sd0 = SDCFROpponent(net_ctor=make_regret_net, device=device)  # represents seat-0 opponent (agent1 snapshots)
        self.sd1 = SDCFROpponent(net_ctor=make_regret_net, device=device)  # represents seat-1 opponent (agent0 snapshots)

        self._snapshot_every = 50

    # ---------- utils ----------
    def _normalize_over_legal(self, probs, mask):
        legal = np.where(mask > 0)[0]
        out = np.zeros_like(probs, dtype=np.float32)
        s = probs[legal].sum()
        if s <= 1e-12:
            out[legal] = 1.0 / max(1, len(legal))
        else:
            out[legal] = probs[legal] / s
        return out

    # ---------- training loop ----------
    def run(self, seed=42, iters=2000, eval_every=50, trajs_per_iter=128, batch_size=8192):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, exploitabilities, SD-CFR exploitabilities, and performance vs fixed opponents.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reused rollout envs (one per seat)
        roll_env0 = LeducEnv()
        roll_env1 = LeducEnv()

        # Stable opponents for data collection (frozen copies of avg nets)
        opp0 = AvgPolicyOpponent(self.agent1)
        opp1 = AvgPolicyOpponent(self.agent0)
        refresh_every = 20  # refresh frozen copies periodically

        iterations = []
        exploits = []
        exploits_sdcfr = []  # new: deterministic SD-CFR mean exploitability
        vs_random = []
        vs_always_call = []
        vs_always_raise = []

        # ---------- initial evaluation ----------
        print("Initial evaluation...")
        iterations.append(0)

        exp_current = self.eval_profile()  # current agents' avg nets
        exploits.append(exp_current)

        if len(getattr(self.sd0, "_weights", [])) > 0 and len(getattr(self.sd1, "_weights", [])) > 0:
            sdcfr_mean_0 = SDCFRMeanPolicyOpponent(self.sd0)
            sdcfr_mean_1 = SDCFRMeanPolicyOpponent(self.sd1)
            exp_sdcfr = self.eval_profile_for_players(sdcfr_mean_0, sdcfr_mean_1)
            exploits_sdcfr.append(exp_sdcfr)
            sdcfr_str = f"{exp_sdcfr:.6f}"
        else:
            exploits_sdcfr.append(float("nan"))
            sdcfr_str = "N/A"

        vs_random.append(self.eval_vs_fixed_opponent(self.policy_random, episodes=100))
        vs_always_call.append(self.eval_vs_fixed_opponent(self.policy_always_call, episodes=100))
        vs_always_raise.append(self.eval_vs_fixed_opponent(self.policy_always_raise, episodes=100))

        print(f"[Iter 0000] Exploitability (current): {exp_current:.6f} | "
              f"Exploitability (SD-CFR): {sdcfr_str} | "
              f"vs Random: {vs_random[-1]:.3f} | vs Call: {vs_always_call[-1]:.3f} | vs Raise: {vs_always_raise[-1]:.3f}")

        # ---------- training iterations ----------
        for it in tqdm(range(1, iters + 1), ascii=True, desc="DREAM"):
            # refresh frozen avg opponents
            if it % refresh_every == 0:
                opp0 = AvgPolicyOpponent(self.agent1)
                opp1 = AvgPolicyOpponent(self.agent0)

            # outcome-sampling data collection against frozen avg opponents
            for _ in range(trajs_per_iter):
                if np.random.rand() < 0.5:
                    self.agent0.outcome_sampling_traj(roll_env0, player_i=0, opponent=opp0)
                    self.agent1.outcome_sampling_traj(roll_env1, player_i=1, opponent=opp1)
                else:
                    self.agent0.outcome_sampling_traj(roll_env0, player_i=1, opponent=opp0)
                    self.agent1.outcome_sampling_traj(roll_env1, player_i=0, opponent=opp1)

            # optimize
            _ = self.agent0.train_step(batch_size=batch_size)
            _ = self.agent1.train_step(batch_size=batch_size)

            # CFR iteration counters (for averaging inside agents)
            self.agent0.increment_iteration()
            self.agent1.increment_iteration()

            # periodically snapshot regret nets into SD-CFR banks (for eval)
            if it % self._snapshot_every == 0 and it >= self._warmup_before_snapshots:
                t1, s1 = self.agent1.export_regret_snapshot()
                t0, s0 = self.agent0.export_regret_snapshot()
                self.sd0.add_snapshot(t1, s1)  # seat-0 opponent is agent1
                self.sd1.add_snapshot(t0, s0)  # seat-1 opponent is agent0

            # periodic evaluation
            if it % eval_every == 0:
                iterations.append(it)

                # current-policy exploitability
                exp_current = self.eval_profile()
                exploits.append(exp_current)

                # deterministic SD-CFR mean exploitability
                if len(getattr(self.sd0, "_weights", [])) > 0 and len(getattr(self.sd1, "_weights", [])) > 0:
                    sdcfr_mean_0 = SDCFRMeanPolicyOpponent(self.sd0)
                    sdcfr_mean_1 = SDCFRMeanPolicyOpponent(self.sd1)
                    exp_sdcfr = self.eval_profile_for_players(sdcfr_mean_0, sdcfr_mean_1)
                    exploits_sdcfr.append(exp_sdcfr)

                    sd_single = evaluate_sdcfr(LeducEnv, self.sd0, self.sd1, episodes=120)
                    sd_both   = evaluate_sdcfr_both_seats(LeducEnv, self.sd0, self.sd1, episodes=240)
                    print(f"   [SD-CFR] single {sd_single}  both {sd_both}")
                    print(f"   [Exploitability] current {exp_current:.6f}  sd-cfr-mean {exp_sdcfr:.6f}")
                else:
                    exploits_sdcfr.append(float("nan"))
                    print(f"   [Exploitability] current {exp_current:.6f}  sd-cfr-mean N/A (no snapshots)")

                # fixed-opponent evals
                vs_random.append(self.eval_vs_fixed_opponent(self.policy_random, episodes=100))
                vs_always_call.append(self.eval_vs_fixed_opponent(self.policy_always_call, episodes=100))
                vs_always_raise.append(self.eval_vs_fixed_opponent(self.policy_always_raise, episodes=100))

                print(f"\n[Iter {it:04d}] Exploitability (current): {exploits[-1]:.6f} | "
                      f"vs Random: {vs_random[-1]:.3f} | vs Call: {vs_always_call[-1]:.3f} | vs Raise: {vs_always_raise[-1]:.3f}")

        return {
            'iterations': iterations,
            'exploitability': exploits,
            'exploitability_sdcfr': exploits_sdcfr,   # NEW
            'vs_random': vs_random,
            'vs_always_call': vs_always_call,
            'vs_always_raise': vs_always_raise
        }

    # ---------- fixed opponents ----------
    def policy_random(self, obs, mask):
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        probs[legal] = 1.0 / len(legal)
        return probs

    def policy_always_call(self, obs, mask):
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        if 0 in legal:
            probs[0] = 1.0
        else:
            probs[legal] = 1.0 / len(legal)
        return probs

    def policy_always_raise(self, obs, mask):
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        if 1 in legal:
            probs[1] = 1.0
        else:
            probs[legal] = 1.0 / len(legal)
        return probs

    # ---------- eval vs fixed opponents ----------
    def eval_vs_fixed_opponent(self, opponent_policy, episodes=100):
        total_reward = 0.0
        for ep in range(episodes):
            agent_seat = ep % 2
            env = LeducEnv()
            env.reset()
            done = env.terminal
            while not done:
                obs = env.get_observation()
                mask = env.get_mask()
                current = env.current
                if current == agent_seat:
                    agent = self.agent0 if agent_seat == 0 else self.agent1
                    probs = agent.policy(obs, mask, use_average=True)
                else:
                    probs = opponent_policy(obs, mask)
                pi = self._normalize_over_legal(probs, mask)
                legal = np.where(mask > 0)[0]
                action = np.random.choice(legal, p=pi[legal])
                env.step(action)
                done = env.terminal
            rewards = env.get_rewards()
            total_reward += rewards[agent_seat]
        return total_reward / episodes

    # ---------- exploitability evaluation (current agents) ----------
    def eval_profile(self):
        """
        Compute exploitability for the current agents' average policies.
        """
        return self.eval_profile_for_players(self.agent0, self.agent1)

    # ---------- generic exploitability evaluation ----------
    def eval_profile_for_players(self, player0, player1):
        """
        Compute exploitability for a supplied policy profile player0 vs player1.
        Each player must expose: policy(obs, mask, use_average=True) -> probs
        """
        unique_decks = [
            ['Q','K','K','J','J','Q'], ['Q','Q','K','J','J','K'],
            ['Q','K','K','J','Q','J'], ['Q','Q','K','J','K','J'],
            ['J','K','K','Q','Q','J'], ['J','J','K','Q','Q','K'],
            ['J','K','K','Q','J','Q'], ['J','J','K','Q','K','Q'],
            ['J','Q','Q','K','K','J'], ['J','J','Q','K','K','Q'],
            ['J','Q','Q','K','J','K'], ['J','J','Q','K','Q','K'],
            ['J','Q','Q','J','K','K'], ['J','K','K','J','Q','Q'],
            ['J','J','J','Q','K','K'], ['J','K','K','Q','J','J'],
            ['Q','Q','K','K','J','J'], ['J','J','K','K','Q','Q'],
            ['J','Q','K','J','Q','K'], ['J','K','Q','J','K','Q'],
            ['J','Q','K','Q','J','K'], ['J','K','Q','Q','K','J'],
            ['J','Q','K','K','J','Q'], ['J','K','Q','K','Q','J']
        ]
        unique_deck_probs = [
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/15, 1/15, 1/15, 1/15, 1/15, 1/15
        ]

        def _norm_policy(p, m):
            return self._normalize_over_legal(p, m)

        def _br(env, br_player, other_player):
            obs, mask, done = env.last()
            if done:
                return env.get_rewards()[br_player]
            if env.current == br_player:
                best = -np.inf
                for a in env.legal_actions():
                    env_copy = copy.deepcopy(env)
                    env_copy.step(a)
                    best = max(best, _br(env_copy, br_player, other_player))
                return best
            else:
                raw = other_player.policy(obs, mask, use_average=True)
                pi  = _norm_policy(raw, mask)
                ev = 0.0
                for a in env.legal_actions():
                    p = float(pi[a])
                    if p == 0.0:
                        continue
                    env_copy = copy.deepcopy(env)
                    env_copy.step(a)
                    ev += p * _br(env_copy, br_player, other_player)
                return ev

        def _v(env, p0, p1):
            obs, mask, done = env.last()
            if done:
                return env.get_rewards()
            raw = p0.policy(obs, mask, True) if env.current == 0 else p1.policy(obs, mask, True)
            pi  = _norm_policy(raw, mask)
            ev0 = ev1 = 0.0
            for a in env.legal_actions():
                p = float(pi[a])
                if p == 0.0:
                    continue
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                v0, v1 = _v(env_copy, p0, p1)
                ev0 += p * v0
                ev1 += p * v1
            return ev0, ev1

        exploitability = 0.0
        for deck, prob in zip(unique_decks, unique_deck_probs):
            env = LeducEnv(deck)
            v_br0 = _br(env, br_player=0, other_player=player1)
            env.reset(deck)
            v_br1 = _br(env, br_player=1, other_player=player0)
            env.reset(deck)
            v_pi0, v_pi1 = _v(env, player0, player1)
            exploitability += prob * ((v_br0 - v_pi0) + (v_br1 - v_pi1))

        return float(exploitability)


# ----------------
# Plotting helpers
# ----------------
def plot_all_metrics(results, title="DREAM Training Progress", outfile=None):
    """
    Create a comprehensive plot with multiple subplots:
    1. Exploitability (current policy) over iterations
    2. Performance vs fixed opponents
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    iters = results['iterations']

    # Subplot 1: Exploitability (current)
    axes[0].plot(iters, results['exploitability'], marker='o', linewidth=2, color='red', label='Exploitability (current)')
    # If SD-CFR series present and not all NaN, overlay
    if 'exploitability_sdcfr' in results:
        sdcfr = np.array(results['exploitability_sdcfr'], dtype=float)
        if np.any(np.isfinite(sdcfr)):
            axes[0].plot(iters, sdcfr, marker='x', linewidth=2, label='Exploitability (SD-CFR mean)')

    axes[0].set_xlabel("Training Iterations", fontsize=12)
    axes[0].set_ylabel("Exploitability", fontsize=12)
    axes[0].set_title("Exploitability (Lower is Better)", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Subplot 2: Performance vs Fixed Opponents
    axes[1].plot(iters, results['vs_random'], marker='s', linewidth=2, label='vs Random', color='blue')
    axes[1].plot(iters, results['vs_always_call'], marker='^', linewidth=2, label='vs Always-Call', color='green')
    axes[1].plot(iters, results['vs_always_raise'], marker='v', linewidth=2, label='vs Always-Raise', color='orange')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    axes[1].set_xlabel("Training Iterations", fontsize=12)
    axes[1].set_ylabel("Average Reward per Game", fontsize=12)
    axes[1].set_title("Performance vs Fixed Opponents (Higher is Better)", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='best')

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile + "_all_metrics.png", dpi=200, bbox_inches='tight')

    plt.show()


def plot_exploitability(iters, exploitabilities, title="Exploitability over training (DREAM)", outfile=None):
    """Legacy function for backward compatibility"""
    plt.figure(figsize=(8, 4.5))
    plt.plot(iters, exploitabilities, marker='o', linewidth=1)
    plt.xlabel("Training iterations")
    plt.ylabel("Exploitability (reward units)")
    plt.title(title + " (vs iterations)")
    plt.grid(True)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile + "_iterations.png", dpi=200)
    plt.show()


# -------------
# Entry point
# -------------
if __name__ == "__main__":
    t0 = time.perf_counter()
    exp = Experiment()
    results = exp.run(iters=2000, eval_every=50)

    # Plot all metrics
    plot_all_metrics(results, outfile='DREAM_training')

    # Also create individual exploitability plot (current only)
    plot_exploitability(results['iterations'], results['exploitability'], outfile='DREAM_exploitability')

    t1 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"Training completed in {t1 - t0:.2f} seconds")
    print(f"{'='*60}")

    # Final printout
    last_idx = -1
    exp_cur = results['exploitability'][last_idx]
    exp_sd  = results.get('exploitability_sdcfr', [float('nan')])[last_idx]
    print(f"\nFinal Results:")
    print(f"  Exploitability (current): {exp_cur:.6f}")
    if np.isfinite(exp_sd):
        print(f"  Exploitability (SD-CFR):  {exp_sd:.6f}")
    else:
        print(f"  Exploitability (SD-CFR):  N/A")
    print(f"  vs Random:                {results['vs_random'][last_idx]:.3f}")
    print(f"  vs Always-Call:           {results['vs_always_call'][last_idx]:.3f}")
    print(f"  vs Always-Raise:          {results['vs_always_raise'][last_idx]:.3f}")
    print(f"{'='*60}")
