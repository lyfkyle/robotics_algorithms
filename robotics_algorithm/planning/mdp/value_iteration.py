#! /usr/bin/env python

import numpy as np
import copy
import math
from collections import defaultdict


class ValueIteration:
    def __init__(self) -> None:
        pass

    def make_greedy_policy(self, Q, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns the action
        """

        def policy_fn(state):
            action_prob = np.zeros(nA)
            best_action_idx = np.argmax(Q[state], axis=-1)
            action_prob[best_action_idx] = 1.0
            return action_prob

        return policy_fn

    def plan(self, env, discount_factor=0.99, diff_threshold=0.01):
        print("ValueIteration: plan!!")

        states = env.states
        actions = env.actions

        # v_state_new = np.full((self.world_x, self.world_y, num_orientation), -np.inf)
        v_state = defaultdict(float)
        Q = defaultdict(lambda: np.zeros(env.action_space_size))
        # q = np.full((self.world_x, self.world_y, num_orientation, 3), -np.inf) # Q(s,a)

        max_change = np.inf
        iter = 0
        while max_change > diff_threshold:
            iter += 1
            print("iter {}, max_change = {}".format(iter, max_change))

            v_state_new = copy.deepcopy(v_state)
            max_change = -np.inf
            for state in states:
                for action in actions:
                    next_states, probs, done = env.transit_func(state, action)
                    reward = env.reward_func(state, action)  # r(s,a)

                    if done:
                        next_state_values = 0
                    else:
                        next_state_values = 0
                        for i, next_state in enumerate(next_states):
                            next_state_values += probs[i] * v_state[next_state]

                    # update Q(s,a)
                    Q[state][action] = reward + discount_factor * next_state_values

                # V(s) = max_a Q(s,a)
                value = Q[state].max(axis=-1)
                if v_state[state] != value:
                    v_state_new[state] = value
                    max_change = max(max_change, math.fabs(value - v_state[state]))

            v_state = v_state_new

        policy = self.make_greedy_policy(Q, env.action_space_size)

        return Q, policy


if __name__ == "__main__":
    from env.windy_gridworld import WindyGridWorld
    from env.cliff_walking import CliffWalking

    # env = WindyGridWorld()
    env = CliffWalking()
    vi = ValueIteration()
    Q, policy = vi.plan(env)

    state = env.reset()
    path = []
    while True:
        ## choose action according to epsilon-greedy policy
        action_probs = policy(state)
        action = np.random.choice(env.actions, p=action_probs)  # choose action
        next_state, reward, done, _ = env.step(action)

        path.append(state)
        state = next_state

        print(state)
        print(action)

        if done:
            break

    env.plot(path)
