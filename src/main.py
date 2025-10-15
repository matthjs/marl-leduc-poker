from matplotlib import pyplot as plt
import numpy as np
from environment.leduc_env import LeducEnv
from algorithms.cfragent import CFRAgent
from src.environment.leduc_env_usage import decode_observation


def train_cfr(iterations=10000, print_every=1000):
    """Train CFR+ agent."""
    agent = CFRAgent()
   
    print(f"Training CFR+ for {iterations} iterations...")
   
    for i in range(iterations):
        agent.train_iteration()
       
        if (i + 1) % print_every == 0:
            print(f"Completed iteration {i + 1}/{iterations}")
   
    print("Training complete!")
    return agent




def play_against_cfr(agent, num_games=10):
    """Play against the trained CFR agent."""
    wins = {0: 0, 1: 0, 'tie': 0}
    total_reward = {0: 0.0, 1: 0.0}
   
    for game in range(num_games):
        env = LeducEnv()
        done = False
       
        print(f"\n{'='*50}")
        print(f"Game {game + 1}/{num_games}")
        print(f"{'='*50}")
       
        while not done:
            obs, mask, done = env.last()
           
            if done:
                break
           
            current_player = env.current
           
            if current_player == 0:  # Human player
                player_hand, community, own_stake, other_stake = decode_observation(obs)
               
                print(f"\nYour hand: {player_hand}")
                print(f"Community card: {community if community else 'None'}")
                print(f"Your stake: {own_stake}, Opponent stake: {other_stake}")
                print(f"Stage: {'Pre-flop' if env.stage == 0 else 'Post-flop'}")
               
                legal_actions = [i for i in range(4) if mask[i] == 1]
                action_names = ['Call', 'Raise', 'Fold', 'Check']
                print(f"Legal actions: {[action_names[a] for a in legal_actions]}")
               
                action = int(input("Enter action (0=Call, 1=Raise, 2=Fold, 3=Check): "))
                while mask[action] == 0:
                    print("Illegal action! Try again.")
                    action = int(input("Enter action: "))
            else:  # CFR agent
                info_set = agent.get_information_set(obs, env.stage)
                strategy = agent.get_average_strategy(info_set, mask)
                action = agent.sample_action(strategy, mask)
                action_names = ['Call', 'Raise', 'Fold', 'Check']
                print(f"Opponent chose: {action_names[action]}")
           
            env.step(action)
       
        rewards = env.get_rewards()
        total_reward[0] += rewards[0]
        total_reward[1] += rewards[1]
       
        if rewards[0] > rewards[1]:
            wins[0] += 1
            print(f"\nYou won! Reward: {rewards[0]}")
        elif rewards[1] > rewards[0]:
            wins[1] += 1
            print(f"\nOpponent won! Reward: {rewards[1]}")
        else:
            wins['tie'] += 1
            print(f"\nTie!")
   
    print(f"\n{'='*50}")
    print(f"Results after {num_games} games:")
    print(f"Your wins: {wins[0]}")
    print(f"Opponent wins: {wins[1]}")
    print(f"Ties: {wins['tie']}")
    print(f"Your average reward: {total_reward[0]/num_games:.2f}")
    print(f"Opponent average reward: {total_reward[1]/num_games:.2f}")




if __name__ == "__main__":
    # Train the agent
    agent = train_cfr(iterations=10000, print_every=2000)
   
    # Play against it
    print("\n" + "="*50)
    print("Now you can play against the trained agent!")
    print("="*50)
    play_against_cfr(agent, num_games=3)

