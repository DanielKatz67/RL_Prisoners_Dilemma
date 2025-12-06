
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Actions
COOPERATE = 0
DEFECT = 1

# Payoff Matrix
# (My Action, Opponent Action) -> My Reward
PAYOFFS = {
    (COOPERATE, COOPERATE): 3,  # R
    (COOPERATE, DEFECT): 0,     # S
    (DEFECT, COOPERATE): 5,     # T
    (DEFECT, DEFECT): 1         # P
}

class RPDEnv(gym.Env):
    def __init__(self, opponent_strategy, memory_depth):
        super().__init__()
        self.opponent_strategy = opponent_strategy
        self.memory_depth = memory_depth
        
        self.action_space = spaces.Discrete(2) # 0: Cooperate, 1: Defect
        
        # Define State Space
        # Memory-1: (my_last, opp_last) -> 4 states
        # Memory-2: (my_last_2, opp_last_2, my_last, opp_last) -> 16 states
        
        self.states = self._generate_states()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}
        
        self.observation_space = spaces.Discrete(len(self.states))
        
        self.current_state = None
        self.reset()

    def _generate_states(self):
        actions = [COOPERATE, DEFECT]
        outcomes = [(a1, a2) for a1 in actions for a2 in actions]
        
        if self.memory_depth == 1:
            return outcomes
        elif self.memory_depth == 2:
            return [(o1, o2) for o1 in outcomes for o2 in outcomes]
        else:
            raise ValueError("Memory depth must be 1 or 2")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial State Definition
        # Memory-1: (C, C)
        # Memory-2: ((C, C), (C, C))
        
        if self.memory_depth == 1:
            self.current_state = (COOPERATE, COOPERATE)
        else:
            self.current_state = ((COOPERATE, COOPERATE), (COOPERATE, COOPERATE))
            
        return self.state_to_idx[self.current_state], {}

    def step(self, action):
        # Determine opponent's action based on current state (history)
        opponent_action = self._get_opponent_action(self.current_state)
        
        # Calculate reward
        reward = PAYOFFS[(action, opponent_action)]
        
        # Update state
        new_outcome = (action, opponent_action)
        
        if self.memory_depth == 1:
            next_state = new_outcome
        else:
            # Shift history: old (t-1) becomes (t-2), new outcome becomes (t-1)
            # current_state is ((my_t-2, opp_t-2), (my_t-1, opp_t-1))
            # next_state is ((my_t-1, opp_t-1), new_outcome)
            next_state = (self.current_state[1], new_outcome)
            
        self.current_state = next_state
        
        return self.state_to_idx[next_state], reward, False, False, {}

    def _get_opponent_action(self, state):
        # Extract relevant history for opponent decision
        # Opponent reacts to MY previous move(s)
        
        if self.memory_depth == 1:
            # state is (my_last, opp_last)
            my_last_move = state[0]
        else:
            # state is ((my_prev, opp_prev), (my_last, opp_last))
            my_last_move = state[1][0]
            
        if self.opponent_strategy == 'ALL-C':
            return COOPERATE
        elif self.opponent_strategy == 'ALL-D':
            return DEFECT
        elif self.opponent_strategy == 'TFT':
            return my_last_move # Copies my last move
        elif self.opponent_strategy == 'Imperfect-TFT':
            # 90% TFT, 10% Opposite
            if np.random.random() < 0.9:
                return my_last_move
            else:
                return 1 - my_last_move
        else:
            raise ValueError(f"Unknown opponent strategy: {self.opponent_strategy}")

    def get_mdp(self):
        """
        Returns:
        P: Transition Matrix of shape (n_states, n_actions, n_states)
           P[s, a, s'] = Probability of transitioning from s to s' given action a
        R: Reward Matrix of shape (n_states, n_actions)
           R[s, a] = Expected reward in state s taking action a
        """
        n_states = len(self.states)
        n_actions = 2
        
        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions))
        
        for s_idx in range(n_states):
            state = self.idx_to_state[s_idx]
            
            for action in [COOPERATE, DEFECT]:
                # Determine opponent's action probabilities
                # For deterministic opponents, prob is 1 for one action, 0 for other
                # For stochastic (Imperfect-TFT), prob is split
                
                opp_probs = self._get_opponent_action_probs(state)
                
                expected_reward = 0
                
                for opp_action, prob in opp_probs.items():
                    if prob == 0: continue
                    
                    # Calculate reward for this outcome
                    r = PAYOFFS[(action, opp_action)]
                    expected_reward += prob * r
                    
                    # Determine next state
                    new_outcome = (action, opp_action)
                    if self.memory_depth == 1:
                        next_state = new_outcome
                    else:
                        next_state = (state[1], new_outcome)
                        
                    next_s_idx = self.state_to_idx[next_state]
                    
                    # Add probability to transition matrix
                    P[s_idx, action, next_s_idx] += prob
                    
                R[s_idx, action] = expected_reward
                
        return P, R

    def _get_opponent_action_probs(self, state):
        if self.memory_depth == 1:
            my_last_move = state[0]
        else:
            my_last_move = state[1][0]
            
        if self.opponent_strategy == 'ALL-C':
            return {COOPERATE: 1.0, DEFECT: 0.0}
        elif self.opponent_strategy == 'ALL-D':
            return {COOPERATE: 0.0, DEFECT: 1.0}
        elif self.opponent_strategy == 'TFT':
            if my_last_move == COOPERATE:
                return {COOPERATE: 1.0, DEFECT: 0.0}
            else:
                return {COOPERATE: 0.0, DEFECT: 1.0}
        elif self.opponent_strategy == 'Imperfect-TFT':
            # 90% copy, 10% flip
            if my_last_move == COOPERATE:
                return {COOPERATE: 0.9, DEFECT: 0.1}
            else:
                return {COOPERATE: 0.1, DEFECT: 0.9}
        else:
            raise ValueError(f"Unknown opponent strategy: {self.opponent_strategy}")
