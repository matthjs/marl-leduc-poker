
import numpy as np

from src.environment.leduc_env import LeducEnv

def decode_one_hot_card(array):
    mapping = ['J', 'Q', 'K']
    index = None if not np.any(array) else int(np.argmax(array))
    return None if index is None else mapping[index]

def decode_one_hot_stake(array):
    mapping = list(range(1,15))
    index = None if not np.any(array) else int(np.argmax(array))
    return 0 if index is None else mapping[index]

def decode_observation(obs):
    player_hand = decode_one_hot_card(obs[0:3])
    community_card = decode_one_hot_card(obs[3:6])
    current_player_stake = decode_one_hot_stake(obs[6:20])
    other_player_stake = decode_one_hot_stake(obs[20:34])

    return player_hand, community_card, current_player_stake, other_player_stake

def print_obs(obs, v):
    if not v:
        return
    
    player_hand, community_card, current_player_stake, other_player_stake = decode_observation(obs)
        
    print(f"Current Player’s Hand: {player_hand}")
    print(f"Community Card: {community_card}")
    print(f"Current Player has commited {current_player_stake} chips to the pot")
    print(f"Other Player has commited {other_player_stake} chips to the pot")

def print_mask(mask, v):
    if not v:
        return
    
    action_mapping = ['Call(0)', 'Raise(1)', 'Fold(2)', 'Check(3)']
    legal_actions = [action_mapping[i] for i, val in enumerate(mask) if val == 1]
    print(f"Legal actions: {', '.join(legal_actions)}")

def print_action(action, v):
    if not v:
        return
    
    action_mapping = ['Call', 'Raise', 'Fold', 'Check']
    print(f"Chosen action: {action_mapping[action]}")
    print("-" * 40)

def main():
    verbose = True
    env = LeducEnv()

    marl_agents = ["player_0", "player_1"]

    games = 3
    for _ in range(games):
        done = False
        while not done:
            for agent in marl_agents:
                observation, mask, done = env.last()

                if done:
                    action = None
                else:
                    print(f"Current Player: {agent}")
                    print_obs(observation, verbose)
                    print_mask(mask, verbose)

                    action = int(input("Enter a action: "))
                    
                    print_action(action, verbose)

                next_obs = env.step(action)
                    
        reward_p0, reward_p1 = env.get_rewards()
        print(f"{marl_agents[0]} reward: {reward_p0}\n{marl_agents[1]} reward: {reward_p1}")
        print("-" * 40)
        # Reverse order
        marl_agents[0], marl_agents[1] = marl_agents[1], marl_agents[0]
        env.reset()


if __name__ == "__main__":
    main()
