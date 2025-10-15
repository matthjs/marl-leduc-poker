from typing import Tuple
import numpy as np
from collections import defaultdict
import random
from src.environment.leduc_env import LeducEnv
from src.environment.leduc_env_usage import decode_observation


class CFRAgent:
    """
    Counterfactual Regret Minimization Plus implementation for Leduc Poker.
    Uses Regret Matching+ at each information set.
    """
   
    def __init__(self, cfrplus: bool = True):
        # Cumulative regrets for each information set and action
        self.regret_sum = defaultdict(lambda: np.zeros(4))
        # Strategy sum for computing average strategy
        self.strategy_sum = defaultdict(lambda: np.zeros(4))
        # Current iteration
        self.iteration = 0
        self.cfrplus = cfrplus
   
    def get_information_set(self, obs: np.ndarray, stage: int) -> Tuple[int, int, int, int, int]:
        """
        Create a unique key for the information set based on observation.
        obs format: [private_card(3), public_card(3), own_stake(14), other_stake(14)]
        """
        private_card = np.argmax(obs[0:3]) if np.any(obs[0:3]) else -1
        public_card = np.argmax(obs[3:6]) if np.any(obs[3:6]) else -1
        own_stake = np.argmax(obs[6:20]) + 1 if np.any(obs[6:20]) else 0
        other_stake = np.argmax(obs[20:34]) + 1 if np.any(obs[20:34]) else 0
       
        return (private_card, public_card, own_stake, other_stake, stage)
   
    def policy(self, info_set: Tuple[int, int, int, int, int], mask: np.ndarray) -> np.ndarray:
        """
        Get current strategy using Regret Matching+.
        Only considers legal actions (mask == 1).
        """
        regrets = self.regret_sum[info_set].copy()
       
        # Apply action mask - set illegal actions to very negative
        regrets = regrets * mask - (1 - mask) * 1e9
       
        # Regret Matching+: use max(regret, 0)
        positive_regrets = np.maximum(regrets, 0)
       
        sum_positive_regrets = np.sum(positive_regrets[mask == 1])
       
        if sum_positive_regrets > 0:
            strategy = positive_regrets / sum_positive_regrets
        else:
            # Uniform over legal actions
            strategy = mask / np.sum(mask)
       
        return strategy
   
    def get_action(self, obs: np.ndarray, mask: np.ndarray, stage: int) -> int:
        """Sample an action according to current strategy."""
        info_set = self.get_information_set(obs, stage)
        policy = self.policy(info_set, mask)   # get strategy
       
        legal_actions = np.where(mask == 1)[0]
        action_probs = policy[legal_actions]
        action_probs = action_probs / action_probs.sum()  # Normalize
       
        return np.random.choice(legal_actions, p=action_probs)
   
    def cfr_recursive(
        self, env: LeducEnv, player: int, reach_prob_0: float, reach_prob_1: float
    ) -> float:
        """
        Recursive CFR traversal.
        Returns the expected utility for the current player.
        """
        obs, mask, done = env.last()
       
        if done:
            rewards = env.get_rewards()
            return rewards[player]
       
        current_player = env.current
        stage = env.stage
        info_set = self.get_information_set(obs, stage)
        strategy = self.policy(info_set, mask)
       
        # If it's our turn, we need to compute counterfactual values
        if current_player == player:
            action_utilities = np.zeros(4)
           
            # Try each legal action
            for action in range(4):
                if mask[action] == 0:
                    continue
               
                # Create a copy of the environment
                env_copy = self.copy_env(env)
                env_copy.step(action)
               
                # Recursively compute utility
                action_utilities[action] = self.cfr_recursive(
                    env_copy, player,
                    reach_prob_0 * (strategy[action] if player == 0 else 1),
                    reach_prob_1 * (strategy[action] if player == 1 else 1)
                )
           
            # Node utility is expected value over our strategy
            node_utility = np.sum(strategy * action_utilities)
           
            # Compute counterfactual regrets
            counterfactual_reach = reach_prob_1 if player == 0 else reach_prob_0
           
            for action in range(4):
                if mask[action] == 0:
                    continue
               
                regret = action_utilities[action] - node_utility
                self.regret_sum[info_set][action] += counterfactual_reach * regret
           
            # Update strategy sum for average strategy (weighted by our reach prob)
            our_reach = reach_prob_0 if player == 0 else reach_prob_1
            self.strategy_sum[info_set] += our_reach * strategy
           
            return node_utility
       
        else:
            # Opponent's turn - sample an action according to their strategy
            action = self.sample_action(strategy, mask)
           
            env_copy = self.copy_env(env)
            env_copy.step(action)
           
            return self.cfr_recursive(
                env_copy, player,
                reach_prob_0 * (strategy[action] if current_player == 0 else 1),
                reach_prob_1 * (strategy[action] if current_player == 1 else 1)
            )
   
    def sample_action(self, strategy: np.ndarray, mask: np.ndarray) -> int:
        """Sample action from strategy (only legal actions)."""
        legal_actions = np.where(mask == 1)[0]
        action_probs = strategy[legal_actions]
        action_probs = action_probs / action_probs.sum()
        return np.random.choice(legal_actions, p=action_probs)
   
    def copy_env(self, env: LeducEnv) -> LeducEnv:
        """Create a deep copy of the environment."""
        new_env = LeducEnv()
        new_env.private = env.private.copy()
        new_env.stakes = env.stakes.copy()
        new_env.raises = env.raises.copy()
        new_env.public = env.public
        new_env.deck = env.deck.copy()
        new_env.to_call = env.to_call
        new_env.stage = env.stage
        new_env.current = env.current
        new_env.terminal = env.terminal
        new_env.illegal = env.illegal
        new_env.wrong_doer = env.wrong_doer
        new_env.winner = env.winner
        new_env.history = env.history.copy()
        return new_env
   
    def train_iteration(self) -> None:
        """Run one iteration of CFR."""
        env = LeducEnv()
       
        # Train for player 0
        self.cfr_recursive(env, player=0, reach_prob_0=1.0, reach_prob_1=1.0)
       
        # Reset and train for player 1
        env.reset()
        self.cfr_recursive(env, player=1, reach_prob_0=1.0, reach_prob_1=1.0)
       
        self.iteration += 1
       
        # CFR+ specific: floor regrets at 0
        if self.cfrplus:
            for info_set in self.regret_sum:
                self.regret_sum[info_set] = np.maximum(self.regret_sum[info_set], 0)
   
    def get_average_strategy(self, info_set: Tuple[int, int, int, int, int], mask: np.ndarray) -> np.ndarray:
        """Get the average strategy over all iterations."""
        strategy_sum = self.strategy_sum[info_set].copy()
       
        # Apply mask
        strategy_sum = strategy_sum * mask
       
        sum_strategy = np.sum(strategy_sum[mask == 1])
       
        if sum_strategy > 0:
            avg_strategy = strategy_sum / sum_strategy
        else:
            # Uniform over legal actions
            avg_strategy = mask / np.sum(mask)
       
        return avg_strategy


