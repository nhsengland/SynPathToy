import random
import numpy as np

def choose_q_action(q_state, EPSILON, q_table):
    """
    Choose an action based on the epsilon-greedy policy.
    """
    if random.random() < EPSILON:
        return random.choice([-3,-2, -1, 0, 1, 2, 3])
    return max([-3, -2, -1, 0, 1, 2, 3], key=lambda a: q_table[q_state][a])

def compute_reward(step_cost, patients):
    # Example: Non-linear penalty for high queue times, clinical penalties and system cost
    queue_penalty = sum((p.queue_time ** 2 for p in patients))  # Quadratic penalty
    clinical_penalty = sum(np.exp(p.outcomes['clinical_penalty'] / 50) for p in patients)  # Exponential penalty
    reward = -step_cost - 0.01 * queue_penalty - 0.1 * clinical_penalty
    return reward
