import copy
import math
from collections import defaultdict

import numpy as np

from robotics_algorithm.env.base_env import MDPEnv


class ValueIteration:
    def __init__(self) -> None:
        pass

    def make_greedy_policy(self, Q, num_actions):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

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

    def run(self, env: MDPEnv, discount_factor: float = 0.99, diff_threshold: float = 0.01):
        print("ValueIteration: plan!!")

        states = env.state_space
        actions = env.action_space

        v_state = defaultdict(float)
        Q = defaultdict(lambda: np.zeros(env.action_space_size))

        # Iterate until convergence.
        # Value iteration operates on Bellman optimality equation
        # Q(s, a) <- Sum_s' p(s' | s) [reward + discount * V(s')]
        # V(s) = max_a Q(s, a)
        max_change = np.inf
        iter = 0
        while max_change > diff_threshold:
            iter += 1
            print("iter {}, max_change = {}".format(iter, max_change))

            v_state_new = copy.deepcopy(v_state)
            max_change = -np.inf
            for state in states:
                for action in actions:
                    results, probs = env.state_transition_func(state, action)
                    print(results, probs)

                    # calculate Q values
                    next_state_value = 0
                    for i, result in enumerate(results):
                        next_state, reward, term, trunc, info = result
                        next_state_value += probs[i] * (reward + discount_factor * v_state[next_state])

                    # update Q(s,a)
                    Q[state][action] = next_state_value

                # V(s) = max_a Q(s,a)
                value = Q[state].max(axis=-1)

                # Check maximum change incurred.
                if v_state[state] != value:
                    v_state_new[state] = value
                    max_change = max(max_change, math.fabs(value - v_state[state]))

            v_state = v_state_new

        policy = self.make_greedy_policy(Q, env.action_space_size)

        return Q, policy
