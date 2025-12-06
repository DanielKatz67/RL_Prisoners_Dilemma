import numpy as np
from rpd_env import RPDEnv, COOPERATE, DEFECT
from policy_iteration import policy_iteration

def run_analysis():
    strategies = ['ALL-C', 'ALL-D', 'TFT', 'Imperfect-TFT']
    gammas = np.linspace(0.1, 0.99, 20)
    
    print("--- 1. Discount Factor Analysis ---")
    # Check when cooperation becomes optimal against TFT (Depth 1)
    env = RPDEnv(opponent_strategy='TFT', memory_depth=1)
    P, R = env.get_mdp()
    
    coop_gamma = None
    for gamma in gammas:
        policy, _ = policy_iteration(P, R, gamma=gamma, states=env.states)
        # Check if policy is all cooperate (0)
        if np.all(policy == COOPERATE):
            if coop_gamma is None:
                coop_gamma = gamma
            # print(f"Gamma {gamma:.2f}: All Cooperate")
        else:
            pass
            # print(f"Gamma {gamma:.2f}: Defect found")
            
    print(f"Cooperation becomes optimal against TFT at Gamma >= {coop_gamma:.2f}")

    print("\n--- 2. Memory Depth Analysis ---")
    gamma = 0.9
    for strategy in strategies:
        rewards = {}
        for depth in [1, 2]:
            env = RPDEnv(opponent_strategy=strategy, memory_depth=depth)
            P, R = env.get_mdp()
            policy, _ = policy_iteration(P, R, gamma=gamma, states=env.states)
            
            # Simulation
            total_reward = 0
            for _ in range(50):
                state_idx, _ = env.reset()
                for _ in range(50):
                    action = policy[state_idx]
                    next_state_idx, reward, _, _, _ = env.step(action)
                    total_reward += reward
                    state_idx = next_state_idx
            rewards[depth] = total_reward / 50
        
        print(f"Opponent: {strategy}")
        print(f"  Memory-1: {rewards[1]:.2f}")
        print(f"  Memory-2: {rewards[2]:.2f}")
        if rewards[2] > rewards[1]:
            print("  Memory-2 outperformed Memory-1")
        else:
            print("  Memory-2 did NOT outperform Memory-1")

    print("\n--- 3. Noise Analysis ---")
    gamma = 0.9
    for strategy in ['TFT', 'Imperfect-TFT']:
        env = RPDEnv(opponent_strategy=strategy, memory_depth=1)
        P, R = env.get_mdp()
        policy, _ = policy_iteration(P, R, gamma=gamma, states=env.states)
        
        print(f"Strategy: {strategy}")
        print("Optimal Policy:")
        for s_idx, action in enumerate(policy):
            state = env.idx_to_state[s_idx]
            action_str = "C" if action == COOPERATE else "D"
            print(f"  State {state}: {action_str}")

if __name__ == "__main__":
    run_analysis()
