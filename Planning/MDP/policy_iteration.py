#!/usr/bin/evn python

import sys
import os
import copy
import math
import numpy as np
from collections import defaultdict

class PolicyIteration():
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

    def plan(self, env, discount_factor=0.99):
        print("PolicyIteration: plan!!")

        states = env.states

        # random init Q
        Q = defaultdict(lambda: np.zeros(env.action_space_size))
        
        iter = 0
        policy_converged = False
        while not policy_converged:
            print("PolicyIteration, iter {},".format(iter))
            prev_Q = copy.deepcopy(Q)
            prev_policy = self.make_greedy_policy(prev_Q, env.action_space_size)
            Q, policy_converged = self.policy_evaluation(env, Q, prev_policy, discount_factor)
            policy = self.policy_improvement(env, Q)
            iter += 1
        
        return Q, policy

    def policy_evaluation(self, env, Q, policy, discount_factor=0.99, max_steps = 10, diff_threshold = 0.01):
        states = env.states
        actions = env.actions

        v_state = defaultdict(float)

        for state in states:
            action_probs = policy(state)
            value = np.dot(action_probs, Q[state]) # calculate v_pi  
            v_state[state] = value

        iter = 0
        max_change = np.inf
        while max_change > diff_threshold and iter < max_steps:
            v_state_new = copy.deepcopy(v_state)
            max_change = 0
            for state in states:
                for action in actions:
                    next_states, probs, done = env.transit_func(state, action)
                    reward = env.reward_func(state, action) # r(s,a)

                    if done:
                        next_state_values = 0
                    else:
                        next_state_values = 0
                        for i, next_state in enumerate(next_states):
                            next_state_values += probs[i] * v_state[next_state]
                            
                    # update Q(s,a)
                    Q[state][action] = reward + discount_factor * next_state_values

                action_probs = policy(state)
                value = np.dot(action_probs, Q[state]) # calculate v_pi
                if v_state[state] != value:
                    v_state_new[state] = value
                    max_change = max(max_change, math.fabs(value - v_state[state]))

            v_state = v_state_new

            print("policy_evaluation, iter {}, max_change = {}".format(iter, max_change))
            iter += 1

        return Q, max_change < diff_threshold

    def policy_improvement(self, env, Q):
        policy = self.make_greedy_policy(Q, env.action_space_size)
        return policy

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    from environment.windy_gridworld import WindyGridWorld
    from environment.cliff_walking import CliffWalking

    # env = WindyGridWorld()
    env = CliffWalking()
    pi = PolicyIteration()
    Q, policy = pi.plan(env)

    state = env.reset()
    path = []
    while True:
        ## choose action according to epsilon-greedy policy
        action_probs = policy(state)
        action = np.random.choice(env.actions, p = action_probs)  # choose action
        next_state, reward, done, _ = env.step(action)

        path.append(state)
        state = next_state

        if done:
            break
    
    env.plot(path)
