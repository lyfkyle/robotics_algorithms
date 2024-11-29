import copy
import math
from collections import defaultdict
from typing import Callable

import numpy as np

from robotics_algorithm.env.base_env import MDPEnv, SpaceType, EnvType


class PolicyIteration:
    def __init__(self, env: MDPEnv) -> None:
        assert env.state_space.type == SpaceType.DISCRETE.value
        assert env.action_space.type == SpaceType.DISCRETE.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env

    def make_greedy_policy(self, Q, num_actions):
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

    def run(self, discount_factor: float = 0.99) -> tuple[dict, Callable]:
        """Run algorithm.

        Args:
            discount_factor (float, optional): discount factor for future reward. Defaults to 0.99.

        Returns:
            Q (dict): Q function
            policy (Callable): policy constructed according to the Q function.
        """
        print("PolicyIteration: plan!!")

        # random init Q
        Q = defaultdict(lambda: np.zeros(self.env.action_space.size))

        iter = 0
        policy_converged = False
        while not policy_converged:
            print("PolicyIteration, iter {},".format(iter))
            prev_Q = copy.deepcopy(Q)
            prev_policy = self.make_greedy_policy(prev_Q, self.env.action_space.size)
            Q, policy_converged = self.policy_evaluation(self.env, Q, prev_policy, discount_factor)
            policy = self.policy_improvement(self.env, Q)
            iter += 1

        return Q, policy

    def policy_evaluation(self, env, Q, policy, discount_factor=0.99, max_steps=10, diff_threshold=0.01):
        # Value iteration operates on Bellman expectation equation
        # For state-action reward:
        # Q_pi(s, a) <- R(s, a) + discount * sum_s' [p(s' | s) * V_pi(s')]
        # For state-state reward:
        # Q_pi(s, a) = sum_s' [p(s' | s) * (R(s, s') +  discount * V_pi(s'))]
        # V_pi(s) = sum_a [pi(a | s) * Q_pi(s, a)]

        states = env.state_space.get_all()
        actions = env.action_space.get_all()

        v_state = defaultdict(float)

        for state in states:
            action_probs = policy(state)
            value = np.dot(action_probs, Q[state])  # calculate v_pi
            v_state[state] = value

        iter = 0
        max_change = np.inf
        while max_change > diff_threshold and iter < max_steps:
            v_state_new = copy.deepcopy(v_state)
            max_change = 0
            for state in states:
                term, trunc, _ = self.env.get_state_info(state)
                if term or trunc:
                    continue

                for action in actions:
                    new_states, probs = env.state_transition_func(state, action)

                    q_sa = 0
                    for i, new_state in enumerate(new_states):
                        reward = self.env.reward_func(state, action, new_state)
                        q_sa += probs[i] * (reward + discount_factor * v_state[new_state])

                    # update Q(s,a)
                    Q[state][action] = q_sa

                # Instead of using max, use action probability under the current policy
                action_probs = policy(state)
                value = np.dot(action_probs, Q[state])  # calculate v_pi

                # Check maximum change incurred.
                if v_state[state] != value:
                    v_state_new[state] = value
                    max_change = max(max_change, math.fabs(value - v_state[state]))

            v_state = v_state_new

            print("policy_evaluation, iter {}, max_change = {}".format(iter, max_change))
            iter += 1

        return Q, max_change < diff_threshold

    def policy_improvement(self, env, Q):
        policy = self.make_greedy_policy(Q, env.action_space.size)
        return policy
