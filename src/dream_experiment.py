from algorithms.dream.agent import DreamAgent
from environment.leduc_env import LeducEnv
import numpy as np
import torch
import copy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import csv
import datetime

# ----------------------------
# Opponent for training/eval
# ----------------------------
class AvgPolicyOpponent:
    """
    Frozen copy of an agent's average policy network.
    Used for stable self-play data collection.
    """
    def __init__(self, src_agent):
        self.net = copy.deepcopy(src_agent.avg_net).eval()
        self.device = next(self.net.parameters()).device

    @torch.inference_mode()
    def policy(self, obs, mask, use_average=True):
        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(self.device)
        pi = self.net(obs_t, mask_t).squeeze(0).cpu().numpy()
        return pi


# ---------------
# Main Experiment
# ---------------
class Experiment:
    def __init__(self):
        # Initialize agents
        env = LeducEnv()
        obs_dim = env.get_observation().shape[0]
        act_dim = 4 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.agent0 = DreamAgent(obs_dim, act_dim, lr=2e-3, device=device)
        self.agent1 = DreamAgent(obs_dim, act_dim, lr=2e-3, device=device)

        # Equalize init to reduce early drift
        self.agent1.regret_net.load_state_dict(self.agent0.regret_net.state_dict())
        self.agent1.q_net.load_state_dict(self.agent0.q_net.state_dict())

        # Use average-policy nets during policy() calls for evaluation
        self.agent0.use_avg_net = True
        self.agent1.use_avg_net = True

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
    def run(self, seed=42, iters=2000, eval_every=50, trajs_per_iter=256, batch_size=8192, num_train_steps=3):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers and exploitabilities for the current policy.
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

        # ---------- initial evaluation ----------
        print("Initial evaluation...")
        iterations.append(0)
        exp_current = self.eval_profile()  # current agents' avg nets
        exploits.append(exp_current)
        print(f"[Iter 0000] Exploitability (current): {exp_current:.6f}")

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
            for _ in range(num_train_steps):
                _ = self.agent0.train_step(batch_size=batch_size)
                _ = self.agent1.train_step(batch_size=batch_size)

            # CFR iteration counters (for averaging inside agents)
            self.agent0.increment_iteration()
            self.agent1.increment_iteration()

            # periodic evaluation
            if it % eval_every == 0:
                iterations.append(it)
                exp_current = self.eval_profile()
                exploits.append(exp_current)
                print(f"\n[Iter {it:04d}] Exploitability (current): {exploits[-1]:.6f}")

        return {
            'iterations': iterations,
            'exploitability': exploits,
        }

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


def save_results_to_csv(results, filename=None):
    """
    Save training metrics to a CSV file.

    Args:
        results: Dictionary containing 'iterations' and 'exploitability'
        filename: Optional filename. If None, generates timestamp-based name.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DREAM_results_{timestamp}.csv"

    
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Prepare data
    iters = results['iterations']
    exploits = results['exploitability']

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(['Iteration', 'Exploitability_Current'])

        # Data rows
        for i in range(len(iters)):
            writer.writerow([iters[i], f"{exploits[i]:.6f}"])

    print(f"\n✓ Results saved to: {filename}")
    return filename


def save_final_summary(results, filename=None):
    """
    Save a summary of final results to a separate CSV.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DREAM_summary_{timestamp}.csv"

    if not filename.endswith('.csv'):
        filename += '.csv'

    last_idx = -1
    exp_cur = results['exploitability'][last_idx]
    final_iter = results['iterations'][last_idx]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Final_Iteration', final_iter])
        writer.writerow(['Exploitability_Current', f"{exp_cur:.6f}"])

    print(f"✓ Summary saved to: {filename}")
    return filename


# -------------
# Entry point
# -------------
if __name__ == "__main__":
    t0 = time.perf_counter()
    exp = Experiment()
    results = exp.run(iters=1000, eval_every=50)

    csv_file = save_results_to_csv(results, filename="DREAM_training_results.csv")
    summary_file = save_final_summary(results, filename="DREAM_final_summary.csv")

    plot_exploitability(results['iterations'], results['exploitability'], outfile='DREAM_exploitability')

    t1 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"Training completed in {t1 - t0:.2f} seconds")
    print(f"{'='*60}")

    # Final printout
    last_idx = -1
    exp_cur = results['exploitability'][last_idx]
    print(f"\nFinal Results:")
    print(f"  Exploitability (current): {exp_cur:.6f}")
    print(f"{'='*60}")
