import numpy as np
import matplotlib.pyplot as plt

def policy_iteration(P, R, gamma=0.9, theta=1e-6, verbose=False, states=None):
    """
    Run Policy Iteration algorithm.
    
    Args:
        P: Transition matrix
        R: Reward matrix
        gamma: Discount factor
        theta: Convergence threshold
        verbose: Whether to print verbose output
        states: List of state names (optional, for plotting)
        
    Returns:
        policy: Optimal policy
        V: Optimal value function
    """
    n_states = P.shape[0]
    # Step 1: Initialize
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    iteration = 1

    if verbose:
        print(f"====================================================================================\n")
        print(f"******** Starting new policy iteration for gamma = {gamma} and theta = {theta} ********")
    
    while True:
        if verbose:
            print(f"\n=== Policy Iteration {iteration} ===")
            print(f"=== Policy: {policy}  ===\n")
            
        # Step 2: Policy Evaluation
        V = policy_evaluation(policy, P, R, V, gamma, theta, verbose)
        
        # Step 3: Policy Improvement
        new_policy = policy_improvement(V, P, R, gamma, verbose)
        
        if verbose and states is not None:
            print("=== Plots: ===")
            plot_values(states, V, title="Value Function per State")
            plot_policy(states, new_policy)
            
        if np.array_equal(new_policy, policy):
            if verbose:
                print("Policy converged.")
                print(f"BEST POLICY found for gamma = {gamma} and theta = {theta} is: {policy}\n")
            break
            
        policy = new_policy
        iteration += 1
        
    if verbose:
        print(f"====================================================================================\n")
    
    return policy, V

def policy_evaluation(policy, P, R, V_init, gamma, theta=1e-6, verbose=False):
    """
    Evaluate a policy given an environment's MDP.
    
    Args:
        policy: Array of shape (n_states,) representing the action to take in each state.
        P: Transition matrix (n_states, n_actions, n_states)
        R: Reward matrix (n_states, n_actions)
        V_init: Initial value function array
        gamma: Discount factor
        theta: Convergence threshold
        verbose: Whether to print verbose output
        
    Returns:
        V: Value function array of shape (n_states,)
    """
    n_states = P.shape[0]
    V = V_init.copy()
    iteration = 1
    
    if verbose:
        print("******** Starting new policy evaluation ************")
    
    while True:
        delta = 0   
        new_V = V.copy()
        for s in range(n_states):
            a = policy[s]
            
            # Bellman Expectation Equation for V_pi
            new_v = calculate_state_value(s, a, P, R, V, gamma)
            new_V[s] = new_v
            
            delta = max(delta, abs(V[s] - new_v))
            
        V = new_V
        
        if verbose:
            # Optional: Plotting logic if states are available globally or passed
            # For now, we skip plotting inside evaluation loop to avoid spam
            pass
            
        iteration += 1
        
        if delta < theta:
            break

    if verbose:
        print(f"Policy evaluation took {iteration} iterations.\n")
            
    return V

def policy_improvement(V, P, R, gamma, verbose):
    """
    Improve the policy given a value function.
    
    Args:
        V: Value function array (n_states,)
        P: Transition matrix
        R: Reward matrix
        gamma: Discount factor
        
    Returns:
        new_policy: Improved policy array
    """
    if verbose:
        print("******** Starting new policy improvement ************\n")

    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    new_policy = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        # Find action that maximizes Q(s, a)
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = calculate_state_value(s, a, P, R, V, gamma)
            
        best_action = np.argmax(action_values)
        new_policy[s] = best_action
        
    return new_policy

def calculate_state_value(s, a, P, R, V, gamma):
    """
    Calculate the value of taking action a in state s.
    V(s) = R(s, a) + gamma * sum(P(s'|s, a) * V(s'))
    """
    n_states = P.shape[0]
    expected_future_val = 0
    for next_s in range(n_states):
        expected_future_val += P[s, a, next_s] * V[next_s]
        
    return R[s, a] + gamma * expected_future_val

def plot_values(states, values, title="Value Function per State"):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, len(states))
    ax.set_ylim(0, max(1, max(values)) * 1.1)
    ax.axis('off')
    for i, (s, v) in enumerate(zip(states, values)):
        rect = plt.Rectangle((i, 0), 1, max(values), fill=True, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(i + 0.5, max(values)*0.65, str(s), ha='center', fontsize=12, fontweight='bold')
        try:
            ax.text(i + 0.5, max(values)*0.35, f"V={float(v):.2f}", ha='center', fontsize=11)
        except ValueError:
            ax.text(i + 0.5, max(values)*0.35, "V=ERR", ha='center', fontsize=11, color='red')
    ax.set_title(title, fontsize=14)
    plt.show()

def plot_policy(states, policy):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, len(states))
    ax.set_ylim(0, 1)
    ax.axis('off')

    for i, (s, a) in enumerate(zip(states, policy)):
        rect = plt.Rectangle((i, 0), 1, 1, fill=True, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.65, str(s), ha='center', fontsize=12, fontweight='bold')

        # Map action 0/1 to text or arrows if desired
        # 0 = Cooperate, 1 = Defect
        action_text = "C" if a == 0 else "D"
        ax.text(i + 0.5, 0.3, action_text, ha='center', fontsize=14, color='blue' if a==0 else 'red')

    ax.set_title("Policy per State", fontsize=14)
    plt.show()
