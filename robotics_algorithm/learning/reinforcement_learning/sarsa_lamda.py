#!/usr/bin/evn python

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.windy_gridworld import WindyGridWorld
from environment.cliff_walking import CliffWalking

class SARSALamda():
    def __init__(self, lamda=0):
        self.lamda = lamda

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

    def learn(self, env, num_episodes, discount_factor = 1.0, alpha=0.5, epsilon = 0.1):
        """
        
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
        # for plotting
        self.episodes = []
        self.cumulative_rewards = []

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space_size))

        # Run 
        for i_episode in range(1, num_episodes + 1):
            # The policy we're following
            policy = self.make_epsilon_greedy_policy(Q, epsilon, env.action_space_size)

            # The first action
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)  # choose action

            # Sample a trajectory
            steps = 0
            cumulative_reward = 0
            trajectory = []
            while True:
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward))

                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)  # choose action

                steps += 1
                cumulative_reward += reward
                if steps % 1000 == 0:
                    print(state)
                    print(action_probs)
                    print(steps)
                    pass

                update_step = steps - self.lamda - 1
                if update_step >= 0:
                    # TD(lamda) Update
                    update_state, update_action = trajectory[update_step][0], trajectory[update_step][1]
                    total_return = sum([x[2] * (discount_factor ** i) for i, x in enumerate(trajectory[update_step:])])
                    td_target = total_return + discount_factor * Q[next_state][next_action]
                    td_delta = td_target - Q[update_state][update_action]
                    Q[update_state][update_action] += alpha * td_delta

                if done:
                    update_step += 1
                    update_step = max(update_step, 0)
                    while update_step < steps:
                        # TD(lamda) Update
                        update_state, update_action = trajectory[update_step][0], trajectory[update_step][1]
                        total_return = sum([x[2] * (discount_factor ** i) for i, x in enumerate(trajectory[update_step:])])
                        td_target = total_return + discount_factor * Q[next_state][next_action]
                        td_delta = td_target - Q[update_state][update_action]
                        Q[update_state][update_action] += alpha * td_delta

                        update_step += 1

                    break

                state = next_state
                action = next_action

            # Print out which episode we're on, useful for debugging.
            if i_episode % 1 == 0:
                self.episodes.append(i_episode)
                self.cumulative_rewards.append(cumulative_reward)
                print("Episode {}/{}, reward : {}".format(i_episode, num_episodes, cumulative_reward))

        plt.plot(self.episodes, self.cumulative_rewards)
        plt.ylabel('reward')
        plt.xlabel('episodes')
        plt.show()

        return Q, policy

    def get_learning_curve(self):
        return self.episodes, self.cumulative_rewards

if __name__ == "__main__":
    env = WindyGridWorld()
    sarsa_lamda = SARSALamda(2)
    Q, policy = sarsa_lamda.learn(env, num_episodes = 400, epsilon = 0.1)
    episodes, learning_curve_1 = sarsa_lamda.get_learning_curve()
    plt.plot(episodes, learning_curve_1, label="sarsa_lamda")
    plt.show()

