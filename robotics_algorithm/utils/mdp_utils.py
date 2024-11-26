import numpy as np


def make_greedy_policy(Q, num_actions):
    """
    Creates an greedy policy based on a given Q-function.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        num_action: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns the action
    """

    def policy_fn(state):
        action_prob = np.zeros(num_actions)
        best_action_idx = np.argmax(Q[state], axis=-1)
        action_prob[best_action_idx] = 1.0
        return action_prob

    return policy_fn
