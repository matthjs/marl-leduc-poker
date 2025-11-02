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


    def run(self, seed=42, iters=2000, eval_every=50, trajs_per_iter=64, batch_size=4096):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, and exploitabilities.
        """
        # Reused rollout envs (one per seat)
        roll_env0 = LeducEnv()
        roll_env1 = LeducEnv()

        iterations = []
        exploits = []

        # Initial evaluation before training
        iterations.append(0)
        exploits.append(self.eval_profile())

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
                print(f"\n[Iter {it:04d}] Exploitability: {exploits[-1]:.6f}")
                
        return iterations, exploits
        

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
        """
        Recursively compute best response value for a player.
        """
        obs, mask, done = env.last()
        if done:
            # Return terminal reward for best response player
            return env.get_rewards()[br_player]
        
        current = env.current

        if current == br_player:
            # Maximize br_player reward
            best = -np.inf
            for a in env.legal_actions():
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                val = self.recursive_evaluate_best_response(env_copy, br_player, other_player)
                best = max(best, val)
            return best
        else:
            # Follow the other player's policy
            # Use average policy for evaluation (use_average=True)
            action_probs = other_player.policy(obs, mask, use_average=True)
            ev = 0
            for a in env.legal_actions():
                prob = action_probs[a]
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                ev += prob * self.recursive_evaluate_best_response(env_copy, br_player, other_player)
            return ev
        
    def recursive_evaluate_policy_value(self, env, agent0, agent1):
        """
        Recursively compute expected value of the current policy for both players.
        """
        obs, mask, done = env.last()
        if done:
            return env.get_rewards()[0], env.get_rewards()[1]

        current = env.current
        if current == 0:
            # Use average policy for evaluation (use_average=True)
            action_probs = agent0.policy(obs, mask, use_average=True)
        else:
            action_probs = agent1.policy(obs, mask, use_average=True)

        ev0, ev1 = 0, 0
        for a in env.legal_actions():
            prob = action_probs[a]
            env_copy = copy.deepcopy(env)
            env_copy.step(a)
            v0, v1 = self.recursive_evaluate_policy_value(env_copy, agent0, agent1)
            ev0 += prob * v0
            ev1 += prob * v1
        return ev0, ev1


def plot_exploitability(iters, exploitabilities, title="Exploitability over training (DREAM)", outfile=None):
    # Plot exploitability vs iterations
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
    iters, exploits = exp.run()
    # Plot exploitability curves
    plot_exploitability(iters, exploits, outfile='DREAM_test')
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.6f} s")