from algorithms.cfragent import CFRAgent
from environment.leduc_env import LeducEnv
import itertools
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class Experiment2:
    def __init__(self):
        # Initialize CFR agent
        self.agent = CFRAgent()

    def run(self, iterations=4000, eval_every=50):
        """
        Run training iterations and evaluate exploitability periodically.
        Returns iteration numbers, and exploitabilities.
        """
        iters = []
        exploits = []

        # Initial evaluation before training
        iters.append(0)
        exploits.append(self.eval_profile())

        for i in tqdm(range(iterations), ascii=True, desc="CFR_P"):
            # Perform one training iteration
            self.agent.train_iteration()
       
            if (i + 1) % eval_every == 0:
                # Record progress every eval_every iterations
                iters.append(i+1)
                exploits.append(self.eval_profile())
        return iters, exploits

    def eval_profile(self):
        """
        Compute exploitability for the current policy profile.
        """
        pi_0, pi_1 = self.agent.get_average_strategy_separated()
        
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
            v_br0 = self.recursive_evaluate_best_respons(env, br_player=0, other_policy=pi_1)
            env.reset(deck)
            v_br1 = self.recursive_evaluate_best_respons(env, br_player=1, other_policy=pi_0)
            # Evaluate policy values
            env.reset(deck)
            v_pi0, v_pi1 = self.recursive_evaluate_policy_value(env, pi_0=pi_0, pi_1=pi_1)

            exploitability += prob * ((v_br0 - v_pi0) + (v_br1 - v_pi1))

        return exploitability
        
    def recursive_evaluate_best_respons(self, env, br_player, other_policy):
        """
        Recursively compute best response value for a player.
        """
        obs, mask, done = env.last()
        if done:
            # Return terminal reward for best response player
            return env.get_rewards()[br_player]
        
        info_set = self.agent.get_information_set(obs, env.stage)
        current = env.current

        if current == br_player:
            # Maximize br_player reward
            best = -np.inf
            for a in env.legal_actions():
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                val = self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
                best = max(best, val)
            return best
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
                env_copy = copy.deepcopy(env)
                env_copy.step(a)
                ev += prob * self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
            return ev

    def recursive_evaluate_policy_value(self, env, pi_0, pi_1):
        """
        Recursively compute policy values for both player.
        """
        obs, mask, done = env.last()
        # Return terminal rewards
        if done:
            return env.get_rewards()[0], env.get_rewards()[1]

        current = env.current
        info_set = self.agent.get_information_set(obs, env.stage)

        if current == 0:
            if info_set in pi_0:
                action_probs = pi_0[info_set]
            else:
                # Default to uniform if info_set not in policy
                action_probs = mask * (1 / np.sum(mask))
        else:
            if info_set in pi_1:
                action_probs = pi_1[info_set]
            else:
                # Default to uniform if info_set not in policy
                action_probs = mask * (1 / np.sum(mask))

        ev0, ev1 = 0, 0
        for a in env.legal_actions():
            prob = action_probs[a]
            env_copy = copy.deepcopy(env)
            env_copy.step(a)
            v0, v1 = self.recursive_evaluate_policy_value(env_copy, pi_0, pi_1)
            ev0 += prob * v0
            ev1 += prob * v1
        return ev0, ev1

def plot_exploitability(iters, exploitabilities, title="Exploitability over training (CFR)", outfile=None):
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
    exp = Experiment2()
    # Run training and evaluation
    iters, exploits = exp.run()
    # Plot exploitability curves
    plot_exploitability(iters, exploits, outfile='CFR2POOPOO_test')
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.6f} s")