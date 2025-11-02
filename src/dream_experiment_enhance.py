from algorithms.dream.agent import DreamAgent
from environment.leduc_env import LeducEnv
import itertools
import numpy as np
import torch
import copy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm





class Experiment:
    def __init__(self):
        # Initialize agents
        env = LeducEnv()
        obs_dim = env.get_observation().shape[0]
        act_dim = 4  # 0:Call, 1:Raise, 2:Fold, 3:Check  (adjust if different)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.agent0 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)
        self.agent1 = DreamAgent(obs_dim, act_dim, lr=5e-4, device=device)

        # Equalize init to reduce early drift
        # Fixed: Use 'q_net' instead of 'baseline'
        self.agent1.regret_net.load_state_dict(self.agent0.regret_net.state_dict())
        self.agent1.q_net.load_state_dict(self.agent0.q_net.state_dict())
        
        # Enable average policy network
        self.agent0.use_avg_net = True
        self.agent1.use_avg_net = True


    def _normalize_over_legal(self, probs, mask):
        legal = np.where(mask > 0)[0]
        out = np.zeros_like(probs, dtype=np.float32)
        s = probs[legal].sum()
        if s <= 1e-12:
            out[legal] = 1.0 / max(1, len(legal))
        else:
            out[legal] = probs[legal] / s
        return out

    def run(self, seed=42, iters=2000, eval_every=50, trajs_per_iter=64, batch_size=4096):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, exploitabilities, and performance vs fixed opponents.
        """
        # Reused rollout envs (one per seat)
        roll_env0 = LeducEnv()
        roll_env1 = LeducEnv()

        iterations = []
        exploits = []
        vs_random = []
        vs_always_call = []
        vs_always_raise = []

        # Initial evaluation before training
        print("Initial evaluation...")
        iterations.append(0)
        exploits.append(self.eval_profile())
        vs_random.append(self.eval_vs_fixed_opponent(self.policy_random, episodes=100))
        vs_always_call.append(self.eval_vs_fixed_opponent(self.policy_always_call, episodes=100))
        vs_always_raise.append(self.eval_vs_fixed_opponent(self.policy_always_raise, episodes=100))
        print(f"[Iter 0000] Exploitability: {exploits[-1]:.6f} | vs Random: {vs_random[-1]:.3f} | vs Call: {vs_always_call[-1]:.3f} | vs Raise: {vs_always_raise[-1]:.3f}")

        for it in tqdm(range(1, iters+1), ascii=True, desc="DREAM"):
            # Perform one training iteration
            # Note: DREAM doesn't use epsilon-greedy exploration
            # The exploration comes from regret matching
            
            for _ in range(trajs_per_iter):
                # Alternate which player goes first to balance data collection
                if np.random.rand() < 0.5:
                    self.agent0.outcome_sampling_traj(roll_env0, player_i=0, opponent=self.agent1)
                    self.agent1.outcome_sampling_traj(roll_env1, player_i=1, opponent=self.agent0)
                else:
                    self.agent0.outcome_sampling_traj(roll_env0, player_i=1, opponent=self.agent1)
                    self.agent1.outcome_sampling_traj(roll_env1, player_i=0, opponent=self.agent0)
            
            m0 = self.agent0.train_step(batch_size=batch_size)
            m1 = self.agent1.train_step(batch_size=batch_size)
            
            # Increment CFR iteration counter
            self.agent0.increment_iteration()
            self.agent1.increment_iteration()

            if (it) % eval_every == 0:
                # Record progress every eval_every iterations
                iterations.append(it)
                exploits.append(self.eval_profile())
                vs_random.append(self.eval_vs_fixed_opponent(self.policy_random, episodes=100))
                vs_always_call.append(self.eval_vs_fixed_opponent(self.policy_always_call, episodes=100))
                vs_always_raise.append(self.eval_vs_fixed_opponent(self.policy_always_raise, episodes=100))
                
                print(f"\n[Iter {it:04d}] Exploitability: {exploits[-1]:.6f} | vs Random: {vs_random[-1]:.3f} | vs Call: {vs_always_call[-1]:.3f} | vs Raise: {vs_always_raise[-1]:.3f}")
                
        return {
            'iterations': iterations,
            'exploitability': exploits,
            'vs_random': vs_random,
            'vs_always_call': vs_always_call,
            'vs_always_raise': vs_always_raise
        }
    
    # ===== Fixed Opponent Policies =====
    
    def policy_random(self, obs, mask):
        """Random policy: uniform over legal actions"""
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        probs[legal] = 1.0 / len(legal)
        return probs
    
    def policy_always_call(self, obs, mask):
        """Always call (action 0) if legal, else random"""
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        if 0 in legal:
            probs[0] = 1.0
        else:
            probs[legal] = 1.0 / len(legal)
        return probs
    
    def policy_always_raise(self, obs, mask):
        """Always raise (action 1) if legal, else random"""
        legal = np.where(mask > 0)[0]
        probs = np.zeros_like(mask, dtype=np.float32)
        if 1 in legal:
            probs[1] = 1.0
        else:
            probs[legal] = 1.0 / len(legal)
        return probs
    
    # ===== Evaluation Against Fixed Opponents =====
    
    def eval_vs_fixed_opponent(self, opponent_policy, episodes=100):
        """
        Evaluate agent0's average reward against a fixed opponent policy.
        Plays from both seats and averages the results.
        
        Returns: average reward per game
        """
        total_reward = 0.0
        
        for ep in range(episodes):
            # Alternate seats
            agent_seat = ep % 2
            
            env = LeducEnv()
            env.reset()
            done = env.terminal
            
            while not done:
                obs = env.get_observation()
                mask = env.get_mask()
                current = env.current
                
                if current == agent_seat:
                    # Agent's turn - use average policy
                    action_probs = self.agent0.policy(obs, mask, use_average=True)
                    legal = np.where(mask > 0)[0]
                    action = np.random.choice(legal, p=action_probs[legal])
                else:
                    # Opponent's turn - use fixed policy
                    action_probs = opponent_policy(obs, mask)
                    legal = np.where(mask > 0)[0]
                    action = np.random.choice(legal, p=action_probs[legal])
                
                env.step(action)
                done = env.terminal
            
            # Get agent's reward
            rewards = env.get_rewards()
            agent_reward = rewards[agent_seat]
            total_reward += agent_reward
        
        return total_reward / episodes
    
    # ===== Exploitability Evaluation =====

    def eval_profile(self):
        """
        Compute exploitability for the current policy profile.
        """

        # Loop over all 24 possible initial card distributions
        # (only first 3 cards matter, 2x private 1x public)
        # (are last 3 cards per deck as pop is used in env)
        unique_decks = [
            ['Q','K','K','J','J','Q'],
            ['Q','Q','K','J','J','K'],
            ['Q','K','K','J','Q','J'],
            ['Q','Q','K','J','K','J'],
            ['J','K','K','Q','Q','J'],
            ['J','J','K','Q','Q','K'],
            ['J','K','K','Q','J','Q'],
            ['J','J','K','Q','K','Q'],
            ['J','Q','Q','K','K','J'],
            ['J','J','Q','K','K','Q'],
            ['J','Q','Q','K','J','K'],
            ['J','J','Q','K','Q','K'],
            ['J','Q','Q','J','K','K'],
            ['J','K','K','J','Q','Q'],
            ['J','J','J','Q','K','K'],
            ['J','K','K','Q','J','J'],
            ['Q','Q','K','K','J','J'],
            ['J','J','K','K','Q','Q'],
            ['J','Q','K','J','Q','K'],
            ['J','K','Q','J','K','Q'],
            ['J','Q','K','Q','J','K'],
            ['J','K','Q','Q','K','J'],
            ['J','Q','K','K','J','Q'],
            ['J','K','Q','K','Q','J']
        ]
        unique_deck_probs = [
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/30, 1/30, 1/30, 1/30, 1/30, 1/30,
            1/15, 1/15, 1/15, 1/15, 1/15, 1/15
        ]
        exploitability = 0
        for deck, prob in zip(unique_decks, unique_deck_probs):
            # Evaluate best response for both players
            env = LeducEnv(deck)
            v_br0 = self.recursive_evaluate_best_response(env, br_player=0, other_player=self.agent1)
            env.reset(deck)
            v_br1 = self.recursive_evaluate_best_response(env, br_player=1, other_player=self.agent0)
            # Evaluate policy values
            env.reset(deck)
            v_pi0, v_pi1 = self.recursive_evaluate_policy_value(env, self.agent0, self.agent1)

            exploitability += prob * ((v_br0 - v_pi0) + (v_br1 - v_pi1))

        return exploitability
        
    def recursive_evaluate_best_response(self, env, br_player, other_player):
        obs, mask, done = env.last()
        if done:
            return env.get_rewards()[br_player]

        current = env.current
        if current == br_player:
            best = -np.inf
            for a in env.legal_actions():
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                val = self.recursive_evaluate_best_response(env_copy, br_player, other_player)
                best = max(best, val)
            return best
        else:
            # Normalize the other player's policy over legal actions
            raw = other_player.policy(obs, mask, use_average=True)
            pi = self._normalize_over_legal(raw, mask)

            ev = 0.0
            for a in env.legal_actions():
                p = float(pi[a])
                if p == 0.0:
                    continue
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                ev += p * self.recursive_evaluate_best_response(env_copy, br_player, other_player)
            return ev

        
    def recursive_evaluate_policy_value(self, env, agent0, agent1):
        obs, mask, done = env.last()
        if done:
            r0, r1 = env.get_rewards()
            return r0, r1

        current = env.current
        if current == 0:
            raw = agent0.policy(obs, mask, use_average=True)
        else:
            raw = agent1.policy(obs, mask, use_average=True)

        pi = self._normalize_over_legal(raw, mask)

        ev0 = 0.0
        ev1 = 0.0
        for a in env.legal_actions():
            p = float(pi[a])
            if p == 0.0:
                continue
            env_copy = copy.deepcopy(env)
            env_copy.step(a)
            v0, v1 = self.recursive_evaluate_policy_value(env_copy, agent0, agent1)
            ev0 += p * v0
            ev1 += p * v1
        return ev0, ev1



def plot_all_metrics(results, title="DREAM Training Progress", outfile=None):
    """
    Create a comprehensive plot with multiple subplots:
    1. Exploitability over iterations
    2. Performance vs fixed opponents
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    iters = results['iterations']
    
    # Subplot 1: Exploitability
    axes[0].plot(iters, results['exploitability'], marker='o', linewidth=2, color='red', label='Exploitability')
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


if __name__ == "__main__":
    t0 = time.perf_counter()
    exp = Experiment()
    results = exp.run(iters=4000, eval_every=50)
    
    # Plot all metrics
    plot_all_metrics(results, outfile='DREAM_training')
    
    # Also create individual exploitability plot
    plot_exploitability(results['iterations'], results['exploitability'], outfile='DREAM_exploitability')
    
    t1 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"Training completed in {t1 - t0:.2f} seconds")
    print(f"{'='*60}")
    print(f"\nFinal Results:")
    print(f"  Exploitability:     {results['exploitability'][-1]:.6f}")
    print(f"  vs Random:          {results['vs_random'][-1]:.3f}")
    print(f"  vs Always-Call:     {results['vs_always_call'][-1]:.3f}")
    print(f"  vs Always-Raise:    {results['vs_always_raise'][-1]:.3f}")
    print(f"{'='*60}")