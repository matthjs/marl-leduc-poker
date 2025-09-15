from leduc_env import LeducEnv
import numpy as np

def decode_one_hot(array):
    mapping = ['J', 'Q', 'K']
    index = None if not np.any(array) else int(np.argmax(array))
    return None if index is None else mapping[index]

def decode_observation(obs):
    player_hand = decode_one_hot(obs[0:3])
    community_card = decode_one_hot(obs[3:6])

    return player_hand, community_card

def print_obs(obs, v):
    if not v:
        return
    
    player_hand, community_card = decode_observation(obs)
        
    print(f"Current Player’s Hand: {player_hand}")
    print(f"Community Card: {community_card}")

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

def main():
    verbose = True
    env = LeducEnv()

    games = 3
    for _ in range(games):
        done = False
        while not done:
            for agent in env.AGENTS:
                observation, mask, reward, done = env.last()

                if done:
                    action = None
                else:
                    print(f"Current Player: {agent}")
                    print_obs(observation, verbose)
                    print_mask(mask, verbose)

                    action = int(input("Enter a action: "))
                    
                    print_action(action, verbose)

                next_obs, reward = env.step(action)
                print(f"Agent: {agent} received reward {reward}")
                print("-" * 40)
                    
        env.reset()


if __name__ == "__main__":
    main()
