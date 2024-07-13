#!/usr/bin/evn python

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

class MCControlOnPolicy():
    def __init__(self):
        pass

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.
        
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        
        """

        def policy_fn(state):
            ## epsilon probability of picking a random action.
            action_prob = np.ones(nA, dtype = float) * epsilon / nA
            
            ## 1 - epsilon probability of picking the best action
            best_action = np.random.choice(np.flatnonzero(np.isclose(Q[state], Q[state].max())))
            action_prob[best_action] += (1 - epsilon)
            return action_prob

        return policy_fn

    def learn(self, env, num_episodes, discount_factor = 1.0, epsilon = 0.1):
        """
        Monte Carlo Control using Epsilon-Greedy policies.
        Finds an optimal epsilon-greedy policy.
        
        Args:
            env: OpenAI gym environment.
            num_episodes: Number of episodes to sample.
            discount_factor: Gamma discount factor.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.
        
        Returns:
            A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
        """
        # Keeps track of sum and count of returns for each state
        # to calculate an average. We could use an array to save all
        # returns (like in the book) but that's memory inefficient.
        returns_count = defaultdict(float)

        # for plotting
        self.episodes = []
        self.cumulative_rewards = []

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space_size))

        # The policy we're following
        policy = self.make_epsilon_greedy_policy(Q, epsilon, env.action_space_size)

        # Run 
        for i_episode in range(1, num_episodes + 1):
            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            trajectory = []
            cumulative_reward = 0
            state = env.reset()

            ## run a trajectory of length 50.
            # for t in range(5000):
            steps = 0
            while True:
                ## choose action according to epsilon-greedy policy
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p = action_probs)  # choose action
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward))

                steps += 1
                cumulative_reward += reward

                if done or steps >= 100000:
                    break

                state = next_state

            ## first-visit MC Prediction

            # Find all states the we've visited in this episode
            # We convert each state to a tuple so that we can use it as a dict key
            state_action_in_episode = set([(tuple(x[0]), x[1]) for x in trajectory]) # we use set because it is first-visit
            for state_action in state_action_in_episode:
                state, action = state_action
                ## calculate total return for that state:
                first_idx = next(i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action)
                ## calcualte total return
                total_return = sum([x[2] * (discount_factor ** i) for i, x in enumerate(trajectory[first_idx:])])
                ## calculate average return in incremental fashion
                returns_count[state_action] += 1.0
                Q[state][action] = Q[state][action] + (1 / returns_count[state_action]) * (total_return - Q[state][action])

            # improve policy based on updated Q function
            policy = self.make_epsilon_greedy_policy(Q, epsilon, env.action_space_size)

            # Print out which episode we're on, useful for debugging.
            if i_episode % 1 == 0:
                self.cumulative_rewards.append(cumulative_reward)
                self.episodes.append(i_episode)
                print("Episode {}/{}, reward : {}".format(i_episode, num_episodes, cumulative_reward))

        return Q, policy
    
    def get_learning_curve(self):
        return self.episodes, self.cumulative_rewards

if __name__ == "__main__":

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from environment.windy_gridworld import WindyGridWorld
    from environment.cliff_walking import CliffWalking

    env = WindyGridWorld()
    mc = MCControlOnPolicy()
    Q, policy = mc.learn(env, num_episodes = 400, epsilon = 0.1)
    episodes, learning_curve_1 = mc.get_learning_curve()
    plt.plot(episodes, learning_curve_1, label="mc")
    plt.show()



