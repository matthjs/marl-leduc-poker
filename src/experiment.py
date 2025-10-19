from algorithms.cfragent import CFRAgent
from environment.leduc_env import LeducEnv
import itertools
import numpy as np
import time

class Experiment:
    def __init__(self):
        """
        """
        self.agent = CFRAgent()

    def run(self, iterations=10000, eval_every=1000):
        """
        Run the experiment — train (optional) and evaluate (optional).
        Returns a dictionary of metrics and timings.
        """
        for i in range(iterations):
            self.agent.train_iteration()
       
            if (i + 1) % eval_every == 0:
                self.eval_profile()

    def eval_profile(self):
        """
        Evaluate current exploitability.
        """
        pi_0, pi_1 = self.agent.get_average_strategy_separated()
        
        # generate all 90 possible root states
        deck = ["J","J","Q","Q","K","K"]
        unique_decks = set(itertools.permutations(deck))

        exploitability = 0
        for deck in unique_decks:
            env = LeducEnv(list(deck))
            v_br0 = self.recursive_evaluate_best_respons(env, br_player=0, other_policy=pi_1)
            env.reset(list(deck))
            v_br1 = self.recursive_evaluate_best_respons(env, br_player=1, other_policy=pi_0)
            exploitability += v_br0 + (-v_br1)

        exploitability /= len(unique_decks)
        print(exploitability)
        
    def recursive_evaluate_best_respons(self, env, br_player, other_policy):
        """
        
        """
        obs, mask, done = env.last()
        if done:
            return env.get_rewards()[0]
        
        info_set = self.agent.get_information_set(obs, env.stage)
        current = env.current

        if current == br_player:
            # Since we return the reward for player 0 at terminal nodes:
            # Player 0 wants to maximize
            # Player 1 wants to minimize
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
            if info_set in other_policy:
                action_probs = other_policy.get(info_set)
            else:
                action_probs = mask * (1 / np.sum(mask))
            ev = 0
            for a in env.legal_actions():
                prob = action_probs[a]
                env_copy = self.agent.copy_env(env)
                env_copy.step(a)
                ev += prob * self.recursive_evaluate_best_respons(env_copy, br_player, other_policy)
            return ev


if __name__ == "__main__":
    t0 = time.perf_counter()
    exp = Experiment()
    exp.run()
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.6f} s")
    