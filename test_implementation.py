
import numpy as np
from rpd_env import RPDEnv, COOPERATE, DEFECT
from policy_iteration import policy_iteration

def test_mdp_generation():
    print("Testing MDP Generation...")
    env = RPDEnv(opponent_strategy='ALL-C', memory_depth=1)
    P, R = env.get_mdp()
    
    assert P.shape == (4, 2, 4)
    assert R.shape == (4, 2)
    
    # Check transitions for ALL-C
    # If I cooperate (0), opponent cooperates (0). Outcome (0,0) -> State 0
    # If I defect (1), opponent cooperates (0). Outcome (1,0) -> State 2
    
    # State 0: (C, C)
    # Action C -> Next State (C, C) = 0
    assert P[0, COOPERATE, 0] == 1.0
    # Action D -> Next State (D, C) = 2 (outcomes are (0,0), (0,1), (1,0), (1,1))
    # Let's check state mapping
    # states: [(0, 0), (0, 1), (1, 0), (1, 1)]
    # indices: 0, 1, 2, 3
    
    assert env.states[0] == (0, 0)
    assert env.states[2] == (1, 0)
    
    assert P[0, DEFECT, 2] == 1.0
    
    print("MDP Generation Passed!")

def test_policy_iteration():
    print("Testing Policy Iteration...")
    env = RPDEnv(opponent_strategy='TFT', memory_depth=1)
    P, R = env.get_mdp()
    
    policy, V = policy_iteration(P, R, gamma=0.9)
    
    print("Policy:", policy)
    print("Value Function:", V)
    print("Policy Iteration Passed!")

if __name__ == "__main__":
    test_mdp_generation()
    test_policy_iteration()
