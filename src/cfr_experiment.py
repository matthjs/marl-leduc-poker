from algorithms.cfragent import CFRAgent
from environment.leduc_env import LeducEnv
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self):
        # Initialize CFR agent
        self.agent = CFRAgent()

    def run(self, iterations=3000, eval_every=50):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, nodes touched, and exploitabilities.
        """
        iters = []
        nodes_touched = []
        exploits = []

        # Initial evaluation before training
        iters.append(0)
        nodes_touched.append(0)
        exploits.append(self.eval_profile())

        for i in range(iterations):
            # Perform one training iteration
            self.agent.train_iteration()
       
            if (i + 1) % eval_every == 0:
                # Record progress every eval_every iterations
                iters.append(i+1)
                nodes_touched.append(self.agent.nodes_touched)
                exploits.append(self.eval_profile())
        return iters, nodes_touched, exploits

    def eval_profile(self):
        """
        Compute exploitability for the current policy profile.
        """
        pi_0, pi_1 = self.agent.get_average_strategy_separated()
        
        # Generate all 90 possible initial card distributions
        deck = ["J","J","Q","Q","K","K"]
        unique_decks = set(itertools.permutations(deck))

        exploitability = 0
        for deck in unique_decks:
            # Evaluate best response for both players
            env = LeducEnv(list(deck))
            v_br0 = self.recursive_evaluate_best_respons(env, br_player=0, other_policy=pi_1)
            env.reset(list(deck))
            v_br1 = self.recursive_evaluate_best_respons(env, br_player=1, other_policy=pi_0)
            exploitability += v_br0 + (-v_br1)

        # Average over all initial states
        exploitability /= len(unique_decks)
        return exploitability
        
    def recursive_evaluate_best_respons(self, env, br_player, other_policy):
        """
        Recursively compute best response value for a player.
        """
        obs, mask, done = env.last()
        if done:
            # Return terminal reward for player 0
            return env.get_rewards()[0]
        
        info_set = self.agent.get_information_set(obs, env.stage)
        current = env.current

        if current == br_player:
            # Maximize for player 0, minimize for player 1
            if br_player == 0:
                best = -np.inf
                for a in env.legal_actions():
                    env_copy = self.agent.copy_env(env)
                    env_copy.step(a)
                    val = self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
                    best = max(best, val)
                return best
            else:
                worst = np.inf
                for a in env.legal_actions():
                    env_copy = self.agent.copy_env(env)
                    env_copy.step(a)
                    val = self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
                    worst = min(worst, val)
                return worst
        else:
            # Follow the other player's policy
            if info_set in other_policy:
                action_probs = other_policy.get(info_set)
            else:
                # Default to uniform if info_set not in policy
                action_probs = mask * (1 / np.sum(mask))
            ev = 0
            for a in env.legal_actions():
                prob = action_probs[a]
                env_copy = self.agent.copy_env(env)
                env_copy.step(a)
                ev += prob * self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
            return ev

def plot_exploitability(iters, nodes_touched, exploitabilities, title="Exploitability over training", outfile=None):
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
    # Run training and evaluation
    iters, nodes_touched, exploits = exp.run()
    # Plot exploitability curves
    plot_exploitability(iters, nodes_touched, exploits, outfile='test')
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.6f} s")
