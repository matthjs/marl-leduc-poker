import numpy as np
import random


def stakes_one_hot(stake):
    return np.eye(14, dtype=int)[stake-1] if 1 <= stake <= 14 else np.zeros(14, dtype=np.float32)


class LeducEnv:
    """
    Two-player Leduc Hold'em environment
    """
    RANKS   = ["J", "Q", "K"]

    def __init__(self):
        self.reset()
    
    def reset(self):
        deck = ["J","J","Q","Q","K","K"]
        random.shuffle(deck)
        self.private = {0: deck.pop(), 1: deck.pop()}
        self.stakes = {0: 1, 1: 2} # Player 0 small blind, bets 1. Player 1 big blind, bets 2
        self.raises = {0: 1, 1: 1} # Players can only raise once per betting round as in Pettingzoo
        self.public = None
        self.deck = deck

        self.to_call = True
        self.stage = 0 # 0 = preflop, 1 = postflop
        self.current = 0 # current player to act
        self.terminal = False
        self.illegal = False
        self.wrong_doer = None
        self.winner = None
        self.history = [] # Required for betting round logic

        # aux, perhaps for evaluation
        self.stacks_active = False
        self.stacks = {0: 0, 1: 0}

    def last(self):
        return self.get_observation(), self.get_mask(), self.terminal
        
    def step(self, action):
        if action is not None and action not in self.legal_actions():
            self.terminal = True
            self.illegal = True
            self.wrong_doer = self.current
            return

        player = self.current
        opponent = 1 - player
        self.history.append(action)

        if action == 0: # Call
            self.stakes[player] = self.stakes[opponent]
            self.to_call = False
        
        if action == 1: # Raise or Bet
            self.raises[self.current] = 0
            if self.to_call:
                self.stakes[player] = self.stakes[opponent]
            # First betting round (pre-flop) raises are 2 chips over other player
            # Second betting round (post-flop) raises are 4 chips over other player
            # Equal to Pettingzoo
            self.stakes[player] += (2 + self.stage * 2)
            self.to_call = True
        
        if action == 2: # Fold
            self.terminal = True
            self.winner = opponent

        if action == 3: # Check
            pass

        new_obs = self.get_observation()
        self.next_turn()
        
        self.possible_deal_public_card()
        return new_obs
        

    def next_turn(self):
        self.current = 1 - self.current
        if len(self.history) >= 2:
            a1 = self.history[-2]
            a2 = self.history[-1]
            # betting round is over when:
            # 1. both players check
            # 2. one player raises and the next calls
            # 3. one player calls and the next checks (ONLY first round because of blind)
            if (a1 == a2 == 3) or (a1 == 1 and a2 == 0) or (a1 == 0 and a2 == 3):
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
        if self.to_call: # Call, Fold
            acts = [0, 2]
            if self.raises[self.current] == 1: # Raise possible
                acts.append(1)
            return acts
        else: # Check
            acts = [3]
            if self.raises[self.current] == 1: # Raise possible
                acts.append(1)
            return acts
    
    def possible_deal_public_card(self):
        if self.stage == 1 and self.public is None and not self.terminal:
            self.public = self.deck.pop()

    # Encode char of card to one hot array for observation
    def rank_one_hot(self, rank):
        return np.array([1 if r == rank else 0 for r in self.RANKS], dtype=np.float32)

    def get_rewards(self):
        if self.illegal:
            r = [0.0, 0.0]
            r[self.wrong_doer] = -1.0
            return r
        if self.winner is None:
            return [0.0, 0.0]
        else:
            loser = 1 - self.winner
            return [
                #Pettingzoo returns 'raised chips /2'
                #This returns the exact amount of won or lost chips
                self.stakes[loser] if self.winner == 0 else -self.stakes[loser],
                self.stakes[loser] if self.winner == 1 else -self.stakes[loser]
            ]
    
    def get_observation(self):
        private_card = self.rank_one_hot(self.private[self.current])
        public_card = self.rank_one_hot(self.public)
        own_stake = stakes_one_hot(self.stakes[self.current])
        other_stake = stakes_one_hot(self.stakes[1-self.current])
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
        action = int(input("Enter a action: "))
        env.step(action)