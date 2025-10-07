from matplotlib import pyplot as plt
import numpy as np

from algorithms.cfgagent import CFRPlusSolver
from environment.leduc_env import LeducEnv

def test_cfr_plus():
    """Comprehensive test of CFR+ implementation"""
    env = LeducEnv()
    
    print("Training CFR+ for Leduc Poker...")
    
    solver = CFRPlusSolver(env)
    exploitability = solver.train(iterations=10000, sampling_method='full')
    
    # Plot convergence
    if exploitability:
        iterations, exp_values = zip(*exploitability)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, exp_values)
        plt.xlabel('Iterations')
        plt.ylabel('Exploitability')
        plt.title('CFR+ Convergence for Leduc Poker')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('cfr_plus_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test against various opponents
    print("\n" + "="*50)
    print("EVALUATION AGAINST DIFFERENT STRATEGIES")
    print("="*50)
    
    test_episodes = 500
    opponents = [
        ("Random", lambda obs, mask: np.random.choice(np.where(mask)[0])),
        ("Always_Call", lambda obs, mask: 0 if 0 in np.where(mask)[0] else np.random.choice(np.where(mask)[0])),
        ("Always_Raise", lambda obs, mask: 1 if 1 in np.where(mask)[0] else np.random.choice(np.where(mask)[0])),
    ]
    
    for opponent_name, opponent_strategy in opponents:
        wins = 0
        losses = 0
        ties = 0
        
        for episode in range(test_episodes):
            env.reset()
            current_opponent = 1  # CFR+ plays as player 0, opponent as player 1
            
            while not env.terminal:
                if env.current == 0:  # CFR+ agent's turn
                    obs = env.get_observation()
                    mask = env.get_mask()
                    action, _ = solver.get_action(obs, mask, use_average_strategy=True)
                    env.step(action)
                else:  # Opponent's turn
                    obs = env.get_observation()
                    mask = env.get_mask()
                    action = opponent_strategy(obs, mask)
                    env.step(action)
            
            rewards = env.get_rewards()
            if rewards[0] > 0:
                wins += 1
            elif rewards[0] < 0:
                losses += 1
            else:
                ties += 1
        
        win_rate = wins / test_episodes * 100
        print(f"VS {opponent_name:12}: {win_rate:5.1f}% win rate ({wins:3d} wins, {losses:3d} losses, {ties:3d} ties)")
    
    # Test self-play (should be close to 50% if near Nash equilibrium)
    print("\n" + "="*50)
    print("SELF-PLAY EVALUATION")
    print("="*50)
    
    wins_p0 = 0
    wins_p1 = 0
    ties = 0
    
    for episode in range(test_episodes):
        env.reset()
        
        while not env.terminal:
            obs = env.get_observation()
            mask = env.get_mask()
            action, _ = solver.get_action(obs, mask, use_average_strategy=True)
            env.step(action)
        
        rewards = env.get_rewards()
        if rewards[0] > 0:
            wins_p0 += 1
        elif rewards[1] > 0:
            wins_p1 += 1
        else:
            ties += 1
    
    print(f"Player 0 wins: {wins_p0/test_episodes*100:5.1f}%")
    print(f"Player 1 wins: {wins_p1/test_episodes*100:5.1f}%") 
    print(f"Ties:         {ties/test_episodes*100:5.1f}%")
    
    return solver

if __name__ == "__main__":
    solver = test_cfr_plus()
