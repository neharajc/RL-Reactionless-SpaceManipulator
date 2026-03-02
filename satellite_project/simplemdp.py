import numpy as np

# -------------------------------
# Simple Markov Decision Process
# -------------------------------

# Number of states and actions
n_states = 4
n_actions = 2  # 0 = left, 1 = right

# Transition table: P[state, action] = next_state
P = np.array([
    [0, 1],  # state 0: left->0, right->1
    [0, 2],  # state 1: left->0, right->2
    [1, 3],  # state 2: left->1, right->3
    [3, 3],  # state 3: terminal, stays
])

# Reward table: reward for landing in a state
R = np.array([0, 0, 0, 1])

# Discount factor
gamma = 0.9


# -------------------------------
# Random policy (for simulation)
# -------------------------------
def random_policy(state):
    """Choose action randomly."""
    return np.random.choice([0, 1])


# -------------------------------
# Run one episode
# -------------------------------
def run_episode(policy):
    state = 0
    total_reward = 0
    steps = 0

    print("Episode trajectory:")

    while state != 3:  # until terminal state
        action = policy(state)
        next_state = P[state, action]
        reward = R[next_state]

        print(f" State {state} --action {action}--> State {next_state}, reward={reward}")

        # accumulate discounted reward
        total_reward += (gamma ** steps) * reward

        state = next_state
        steps += 1

        if steps > 20:  # safety stop
            break

    print(f"Total discounted reward = {total_reward:.3f}\n")
    return total_reward


# -------------------------------
# Value Iteration (Bellman backup)
# -------------------------------
def value_iteration(iterations=20):
    V = np.zeros(n_states)  # initialize values

    for it in range(iterations):
        V_new = np.zeros_like(V)
        for s in range(n_states):
            if s == 3:  # terminal state
                V_new[s] = 0
            else:
                Q_values = []
                for a in range(n_actions):
                    s_next = P[s, a]
                    r = R[s_next]
                    Q_values.append(r + gamma * V[s_next])
                V_new[s] = max(Q_values)  # greedy choice
        V = V_new
    return V


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("=== Running sample episodes with random policy ===")
    for _ in range(3):
        run_episode(random_policy)

    print("=== Value Iteration ===")
    V_opt = value_iteration()
    print("Optimal state values:", V_opt)
