from algorithms.dream.agent import DreamAgent
from environment.leduc_env import LeducEnv
import itertools
import numpy as np
import torch
import copy
import time
import matplotlib.pyplot as plt

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
        self.agent1.adv_net.load_state_dict(self.agent0.adv_net.state_dict())
        self.agent1.baseline.load_state_dict(self.agent0.baseline.state_dict())


    def run(self, seed=42, iters=5000, eval_every=50, trajs_per_iter=64, batch_size=4096):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, nodes touched, and exploitabilities.
        """
        # Reused rollout envs (one per seat)
        roll_env0 = LeducEnv()
        roll_env1 = LeducEnv()

        iterations = []
        nodes_touched = []
        exploits = []

        # Initial evaluation before training
        iterations.append(0)
        nodes_touched.append(0)
        exploits.append(self.eval_profile())

        for it in range(1, iters+1):
            # Perform one training iteration
            # Decay exploration
            self.agent0.eps = max(0.02, 0.10 * (1.0 - it / 2000.0))
            self.agent1.eps = max(0.02, 0.10 * (1.0 - it / 2000.0))
            
            for _ in range(trajs_per_iter):
                self.agent0.outcome_sampling_traj(roll_env0, player_i=0, opponent=self.agent1)
                self.agent1.outcome_sampling_traj(roll_env1, player_i=1, opponent=self.agent0)
            
            m0 = self.agent0.train_step(batch_size=batch_size)
            m1 = self.agent1.train_step(batch_size=batch_size)

       
            if (it) % eval_every == 0:
                # Record progress every eval_every iterations
                iterations.append(it)
                exploits.append(self.eval_profile())
                nodes_touched.append(self.agent0.nodes_touched + self.agent1.nodes_touched)
        return iterations, nodes_touched, exploits
        

    def eval_profile(self):
        """
        Compute exploitability for the current policy profile.
        """

        # Generate all 90 possible initial card distributions
        deck = ["J","J","Q","Q","K","K"]
        unique_decks = set(itertools.permutations(deck))

        exploitability = 0
        for deck in unique_decks:
            # Evaluate best response for both players
            env = LeducEnv(list(deck))
            v_br0 = self.recursive_evaluate_best_respons(env, br_player=0, other_player=self.agent1)
            env.reset(list(deck))
            v_br1 = self.recursive_evaluate_best_respons(env, br_player=1, other_player=self.agent0)
            exploitability += v_br0 + (-v_br1)

        # Average over all initial states
        exploitability /= len(unique_decks)
        return exploitability
        
    def recursive_evaluate_best_respons(self, env, br_player, other_player):
        """
        Recursively compute best response value for a player.
        """
        obs, mask, done = env.last()
        if done:
            # Return terminal reward for player 0
            return env.get_rewards()[0]
        
        current = env.current

        if current == br_player:
            # Maximize for player 0, minimize for player 1
            if br_player == 0:
                best = -np.inf
                for a in env.legal_actions():
                    env_copy = copy.deepcopy(env)
                    env_copy.step(a)
                    val = self.recursive_evaluate_best_respons(env_copy, br_player, other_player)
                    best = max(best, val)
                return best
            else:
                worst = np.inf
                for a in env.legal_actions():
                    env_copy = copy.deepcopy(env)
                    env_copy.step(a)
                    val = self.recursive_evaluate_best_respons(env_copy, br_player, other_player)
                    worst = min(worst, val)
                return worst
        else:
            # Follow the other player's policy
            action_probs = other_player.policy(obs, mask)
            ev = 0
            for a in env.legal_actions():
                prob = action_probs[a]
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                ev += prob * self.recursive_evaluate_best_respons(env_copy, br_player, other_player)
            return ev

def plot_exploitability(iters, nodes_touched, exploitabilities, title="Exploitability over training (DREAM)", outfile=None):
    # Plot exploitability vs iterations
    plt.figure(figsize=(8, 4.5))
    plt.plot(iters, exploitabilities, marker='o', linewidth=1)
    plt.xlabel("Training iterations")
    plt.ylabel("Exploitability (player-0 units)")
    plt.title(title + " (vs iterations)")
    plt.grid(True)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile + "_iterations.png", dpi=200)
    plt.show()

    # Plot exploitability vs nodes touched
    plt.figure(figsize=(8, 4.5))
    plt.plot(nodes_touched, exploitabilities, marker='o', linewidth=1, color='orange')
    plt.xlabel("Nodes touched")
    plt.ylabel("Exploitability (player-0 units)")
    plt.title(title + " (vs nodes touched)")
    plt.grid(True)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile + "_nodes.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    t0 = time.perf_counter()
    exp = Experiment()
    iters, nodes_touched, exploits = exp.run()
    # Plot exploitability curves
    plot_exploitability(iters, nodes_touched, exploits, outfile='DREAM_test')
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.6f} s")
