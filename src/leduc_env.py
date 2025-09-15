import numpy as np
import random

class LeducEnv:
    """
    Two-player Leduc Hold'em environment
    """
    ACTIONS = ["Call", "Raise", "Fold", "Check"]
    RANKS   = ["J", "Q", "K"]
    AGENTS = ["player_0","player_1"]

    
    def __init__(self):
        self.reset()
    
    def reset(self):
        deck = ["J","J","Q","Q","K","K"]
        random.shuffle(deck)
        self.private = {0: deck.pop(), 1: deck.pop()}
        self.stakes = {0: 1, 1: 2}
        self.raises = {0: 1, 1: 1}
        self.public = None
        self.deck = deck

        self.to_call = True
        self.stage = 0 # 0 = preflop, 1 = postflop
        self.current = 0 # current player to act
        self.terminal = False
        self.winner = None
        self.history = []

        #aux
        self.stacks_active = False
        self.stacks = {0: 0, 1: 0}


    def last(self):
        return self.get_observation(), self.get_mask(), self.get_rewards(), self.terminal
        
    def step(self, action):
        if action != None:
            assert action in self.legal_actions()

        player = self.current
        opponent = 1 - player
        self.history.append(action)

        if action == 0:  # Call
            self.stakes[player] = self.stakes[opponent]
            self.to_call = False
        
        if action == 1:  # Raise or Bet
            self.raises[self.current] = 0
            if self.to_call:
                self.stakes[player] = self.stakes[opponent]
            self.stakes[player] += (2 + self.stage * 2)
            self.to_call = True
        
        if action == 2:  # Fold
            self.terminal = True
            self.winner = opponent

        if action == 3:  # Check
            pass

        new_obs = self.get_observation()
        self.next_turn()
        reward = self.get_rewards()
        self.current = 1 - self.current
        
        self.possible_deal_public_card()
        return new_obs, reward
        

    def next_turn(self):
        if len(self.history) >= 2:
            a1 = self.history[-2]
            a2 = self.history[-1]
            # betting round is over when:
            # 1. both players check
            # 2. one player raises and the next calls
            # 3. one player calls and the next checks (ONLY first round because of small blind)
            if not self.to_call and ((a1 == a2 == 3) or (a1 == 1 and a2 == 0) or (a1 == 0 and a2 == 3)):
                self.history = []
                if self.stage == 0:
                    self.stage = 1
                    self.raises = {0: 1, 1: 1}
                    return
                else:
                    self.terminal = True
                    self.determine_winner()
                    return

    def legal_actions(self):
        if self.terminal:
            return []
        if self.to_call:
            acts = [0, 2]               # Call, Fold
            if self.raises[self.current] == 1:
                acts.append(1)          # Raise possible
            return acts
        else:
            acts = [3]                  # Check
            if self.raises[self.current] == 1:
                acts.append(1)          # Bet/Raise
            return acts
    
    def possible_deal_public_card(self):
        if self.stage == 1 and self.public is None and not self.terminal:
            self.public = self.deck.pop()

    def rank_one_hot(self, rank):
        return np.array([1 if r == rank else 0 for r in self.RANKS], dtype=np.float32)
    
    def stakes_one_hot(self, stake):
        return np.eye(14, dtype=int)[stake-1] if 1 <= stake <= 14 else np.zeros(14, dtype=np.float32)

    def get_rewards(self):
        if self.winner == None:
            return 0.0
        if self.winner == self.current:
            return self.stakes[self.current] / 2
        else:
            return -self.stakes[self.current] / 2
    
    def get_observation(self):
        private_card = self.rank_one_hot(self.private[self.current])
        public_card = self.rank_one_hot(self.public)
        own_stake = self.stakes_one_hot(self.stakes[self.current])
        other_stake = self.stakes_one_hot(self.stakes[1-self.current])
        return np.concatenate((private_card, public_card, own_stake, other_stake))
    
    def get_mask(self):
        mask = self.legal_actions()
        return np.isin(np.arange(4), mask).astype(int)
    
    def determine_winner(self):
        # check for pair
        if self.private[0] == self.public:
            self.winner = 0
            if self.stacks_active:
                self.update_stacks()
            return
        if self.private[1] == self.public:
            self.winner = 1
            if self.stacks_active:
                self.update_stacks()
            return
        
        # compare high card
        rank_order = {"J": 1, "Q": 2, "K": 3}
        r0 = rank_order[self.private[0]]
        r1 = rank_order[self.private[1]]

        if r0 > r1:
            self.winner = 0
            if self.stacks_active:
                self.update_stacks()
            return
        if r1 > r0:
            self.winner = 1
            if self.stacks_active:
                self.update_stacks()
            return
        
        # tie
        self.winner = None
        
    #aux
    def set_stacks(self, amount):
        self.stacks_active = True
        self.stacks[0] = amount
        self.stacks[1] = amount

    def get_stacks(self):
        return self.stacks.copy()
    
    def update_stacks(self):
        if self.winner == 0:
            self.stacks[0] += self.stakes[1]
            self.stacks[1] -= self.stakes[1]
        if self.winner == 1:
            self.stacks[1] += self.stakes[0]
            self.stacks[0] -= self.stakes[0]
        if self.stacks[0] < 0 or self.stacks[1] < 0:
            self.stacks_active = False
    

if __name__ == "__main__":
    env = LeducEnv()
    while True:
        env.last()
        action = int(input("Enter a action: "))
        env.step(action)