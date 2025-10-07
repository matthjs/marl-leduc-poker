import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class CFRPlusSolver:
    """
    CFR+ Solver for Leduc Poker
    """
    
    def __init__(self, env):
        self.env = env
        self.regrets = defaultdict(lambda: np.zeros(4))  # Regrets for 4 actions
        self.strategy_sum = defaultdict(lambda: np.zeros(4))  # Cumulative strategies
        self.current_strategy = defaultdict(lambda: np.ones(4) / 4)  # Current strategy
        
    def get_infoset_key(self, player):
        """Create information set key: (private_card, public_card, betting_history, player)"""
        private_card = self.env.private[player]
        public_card = self.env.public if self.env.public else 'None'
        history = tuple(self.env.history)
        
        return (private_card, public_card, history, player)
    
    def get_strategy(self, infoset):
        """Get current strategy using regret matching+"""
        regrets = self.regrets[infoset]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            return positive_regrets / sum_positive
        else:
            return np.ones(len(regrets)) / len(regrets)
    
    def update_strategy(self, infoset, strategy, iteration):
        """Update cumulative strategy (weighted by iteration)"""
        self.strategy_sum[infoset] += strategy * iteration
    
    def get_average_strategy(self, infoset):
        """Get average strategy over all iterations"""
        strategy_sum = self.strategy_sum[infoset]
        total = np.sum(strategy_sum)
        
        if total > 0:
            return strategy_sum / total
        else:
            return np.ones(len(strategy_sum)) / len(strategy_sum)
    
    def train(self, iterations=10000, sampling_method='external'):
        """
        Train CFR+ with specified sampling method
        
        Args:
            iterations: Number of training iterations
            sampling_method: 'external' (faster, more robust) or 'full' (exact but slower)
        """
        exploitability = []
        
        for t in range(1, iterations + 1):
            if t % 1000 == 0:
                print(f"Iteration {t}/{iterations}")
            
            # Traverse game tree for both players
            for player in [0, 1]:
                self.env.reset()
                
                if sampling_method == 'external':
                    self._external_sampling_cfr(player, t)
                else:
                    self._full_cfr_traversal(player, 1.0, 1.0, t)
            
            # Track progress
            if t % 100 == 0:
                exp = self.calculate_exploitability()
                exploitability.append((t, exp))
                if t % 1000 == 0:
                    print(f"  Exploitability: {exp:.6f}")
        
        return exploitability
    
    def _external_sampling_cfr(self, player, iteration):
        """
        External sampling CFR - more practical and robust
        Samples chance outcomes and opponent actions
        """
        self.env.reset()
        
        # Handle public card chance node with sampling
        if self.env.stage == 1 and self.env.public is None and self.env.deck:
            self.env.public = random.choice(self.env.deck)
            self.env.deck = [c for c in self.env.deck if c != self.env.public]
        
        return self._external_traversal(player, 1.0, iteration)
    
    def _external_traversal(self, player, reach_prob, iteration):
        """External sampling traversal"""
        if self.env.terminal:
            rewards = self.env.get_rewards()
            return rewards[player]
        
        # Handle chance nodes with sampling
        if self.env.stage == 1 and self.env.public is None and not self.env.terminal:
            if self.env.deck:
                card = random.choice(self.env.deck)
                self.env.public = card
                self.env.deck = [c for c in self.env.deck if c != card]
            else:
                self.env.terminal = True
                self.env.determine_winner()
                rewards = self.env.get_rewards()
                return rewards[player]
        
        current_player = self.env.current
        infoset = self.get_infoset_key(current_player)
        
        # Get current strategy
        strategy = self.get_strategy(infoset)
        self.current_strategy[infoset] = strategy
        
        if current_player == player:
            # Current player - use CFR update
            node_value = 0
            action_values = np.zeros(4)
            legal_actions = self.env.legal_actions()
            
            for action in legal_actions:
                # Save state before taking action
                env_state = self._save_state()
                
                # Take action
                self.env.step(action)
                
                # Recursive call
                action_value = self._external_traversal(player, reach_prob, iteration)
                action_values[action] = action_value
                
                # Restore state
                self._load_state(env_state)
                
                node_value += strategy[action] * action_value
            
            # Update regrets
            for action in legal_actions:
                regret = action_values[action] - node_value
                # CFR+ update: max(0, current_regret + new_regret)
                self.regrets[infoset][action] = max(self.regrets[infoset][action] + regret, 0)
            
            # Update cumulative strategy
            self.update_strategy(infoset, strategy, iteration)
            
            return node_value
            
        else:
            # Opponent - sample from their strategy
            legal_actions = self.env.legal_actions()
            if not legal_actions:
                return 0
                
            # Mask strategy to legal actions
            strategy_masked = np.zeros(4)
            strategy_masked[legal_actions] = strategy[legal_actions]
            if np.sum(strategy_masked) > 0:
                strategy_masked /= np.sum(strategy_masked)
            else:
                # Uniform if no positive strategy
                strategy_masked[legal_actions] = 1.0 / len(legal_actions)
            
            # Sample opponent action
            action = np.random.choice(4, p=strategy_masked)
            self.env.step(action)
            
            # Continue traversal
            return self._external_traversal(player, reach_prob, iteration)
    
    def _full_cfr_traversal(self, player, reach_p0, reach_p1, iteration):
        """
        Full CFR traversal - exact but more computationally expensive
        Traverses all actions and chance outcomes
        """
        # Terminal node
        if self.env.terminal:
            rewards = self.env.get_rewards()
            return rewards[player]
        
        # Chance node - public card dealing
        if self.env.stage == 1 and self.env.public is None and not self.env.terminal:
            if not self.env.deck:
                self.env.terminal = True
                self.env.determine_winner()
                rewards = self.env.get_rewards()
                return rewards[player]
            
            # Average over all possible public cards
            node_utility = 0
            original_deck = self.env.deck.copy()
            
            for card in original_deck:
                env_state = self._save_state()
                self.env.public = card
                self.env.deck = [c for c in original_deck if c != card]
                
                utility = self._full_cfr_traversal(player, reach_p0, reach_p1, iteration)
                node_utility += utility / len(original_deck)
                
                self._load_state(env_state)
            
            return node_utility
        
        current_player = self.env.current
        infoset = self.get_infoset_key(current_player)
        
        strategy = self.get_strategy(infoset)
        self.current_strategy[infoset] = strategy
        
        node_utility = 0
        action_utilities = np.zeros(4)
        legal_actions = self.env.legal_actions()
        
        for action in legal_actions:
            env_state = self._save_state()
            self.env.step(action)
            
            # Calculate new reach probabilities
            if current_player == 0:
                new_reach_p0 = reach_p0 * strategy[action]
                new_reach_p1 = reach_p1
            else:
                new_reach_p0 = reach_p0
                new_reach_p1 = reach_p1 * strategy[action]
            
            action_utility = self._full_cfr_traversal(player, new_reach_p0, new_reach_p1, iteration)
            action_utilities[action] = action_utility
            
            self._load_state(env_state)
            node_utility += strategy[action] * action_utility
        
        # Update regrets and strategy
        if current_player == player:
            for action in legal_actions:
                regret = action_utilities[action] - node_utility
                self.regrets[infoset][action] = max(self.regrets[infoset][action] + regret, 0)
        
        self.update_strategy(infoset, strategy, iteration)
        return node_utility
    
    def _save_state(self):
        """Save complete environment state for recursion"""
        return {
            'private': self.env.private.copy(),
            'stakes': self.env.stakes.copy(),
            'raises': self.env.raises.copy(),
            'public': self.env.public,
            'deck': self.env.deck.copy(),
            'to_call': self.env.to_call,
            'stage': self.env.stage,
            'current': self.env.current,
            'terminal': self.env.terminal,
            'history': self.env.history.copy()
        }
    
    def _load_state(self, state):
        """Load environment state"""
        self.env.private = state['private'].copy()
        self.env.stakes = state['stakes'].copy()
        self.env.raises = state['raises'].copy()
        self.env.public = state['public']
        self.env.deck = state['deck'].copy()
        self.env.to_call = state['to_call']
        self.env.stage = state['stage']
        self.env.current = state['current']
        self.env.terminal = state['terminal']
        self.env.history = state['history'].copy()
    
    def calculate_exploitability(self):
        """Calculate Nash equilibrium exploitability (simplified)"""
        total_positive_regret = 0
        for infoset, regrets in self.regrets.items():
            total_positive_regret += np.sum(np.maximum(regrets, 0))
        return total_positive_regret / len(self.regrets) if self.regrets else 0
    
    def get_action(self, observation, legal_actions_mask, use_average_strategy=True):
        """
        Get action for current state
        
        Args:
            observation: Current game observation
            legal_actions_mask: Mask of legal actions
            use_average_strategy: Whether to use average strategy (for evaluation)
                                  or current strategy (for training)
        """
        player_hand = self._decode_card(observation[0:3])
        community_card = self._decode_card(observation[3:6])
        history = tuple(self.env.history)
        
        infoset_key = (player_hand, community_card, history, self.env.current)
        
        if use_average_strategy:
            strategy = self.get_average_strategy(infoset_key)
        else:
            strategy = self.current_strategy.get(infoset_key, np.ones(4) / 4)
        
        # Mask illegal actions
        masked_strategy = strategy * legal_actions_mask
        if np.sum(masked_strategy) > 0:
            action_probs = masked_strategy / np.sum(masked_strategy)
        else:
            action_probs = legal_actions_mask / np.sum(legal_actions_mask)
        
        return np.random.choice(4, p=action_probs), action_probs
    
    def _decode_card(self, one_hot):
        """Decode one-hot encoded card"""
        cards = ['J', 'Q', 'K']
        if np.any(one_hot):
            idx = np.argmax(one_hot)
            if idx < len(cards):
                return cards[idx]
        return 'None'

