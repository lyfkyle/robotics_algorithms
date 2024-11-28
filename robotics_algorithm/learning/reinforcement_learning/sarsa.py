import numpy as np
from collections import defaultdict
from typing import Callable

from robotics_algorithm.env.base_env import MDPEnv, SpaceType, EnvType


class SARSA:
    def __init__(self, env: MDPEnv, max_episode_len: int = 5000) -> None:
        """
        Constructor.

        Args:
            env (MDPEnv): the env.
        """
        assert env.state_space.type == SpaceType.DISCRETE.value
        assert env.action_space.type == SpaceType.DISCRETE.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self._max_episode_len = max_episode_len

    def make_epsilon_greedy_policy(self, epsilon, num_actions):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            epsilon: The probability to select a random action . float between 0 and 1.
            num_actions: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length num_actions.

        """

        def policy_fn(state):
            # epsilon probability of picking a random action.
            action_prob = np.ones(num_actions, dtype=float) * epsilon / num_actions

            # (1 - epsilon) probability of picking the best action
            best_action = np.argmax(self.Q[state])
            action_prob[best_action] += 1 - epsilon
            return action_prob

        return policy_fn

    def run(
        self, num_episodes: int = 500, discount_factor: float = 0.95, epsilon: float = 0.1, alpha=0.05
    ) -> tuple[dict, Callable]:
        """
        Run the SARSA algorithm to learn the optimal policy.

        Args:
            num_episodes (int): Number of episodes to sample. Defaults to 500.
            discount_factor (float, optional): Gamma discount factor. Defaults to 0.95.
            epsilon (float, optional): Chance the sample a random action. Float betwen 0 and 1. Defaults to 0.1.
            alpha (float): TD Update rate.

        Returns:
            tuple[dict, Callable]: A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
        """
        # for plotting
        self.episodes = []
        self.cumulative_rewards = []

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.size))

        # The policy we're following
        policy = self.make_epsilon_greedy_policy(epsilon, self.env.action_space.size)

        # Iterate over each episode
        for i_episode in range(1, num_episodes + 1):
            # Initialize the state and action
            state, _ = self.env.reset()

            cumulative_reward = 0
            # Run the episode
            action_probs = policy(state)
            action = np.random.choice(self.env.action_space.get_all(), p=action_probs)  # choose action
            for steps in range(self._max_episode_len):
                # Take action, observe reward and next state
                next_state, reward, term, trunc, _ = self.env.step(action)
                cumulative_reward += reward

                # Next state and next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(self.env.action_space.get_all(), p=next_action_probs)

                # Perform TD Update
                # NOTE: we perform TD update even if term or trunc is true. Although Q function for terminal
                #       state does not make sense, this is needed so that the negative reward earned when transitioning
                #       to terminal state is learned.
                td_target = reward + discount_factor * self.Q[next_state][next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_delta

                # Print debug information every 1000 steps
                # if steps % 1000 == 0:
                #     print(state)
                #     print(action_probs)
                #     print(steps)

                # Update state and action
                state = next_state
                action = next_action

                # Check for termination
                if term or trunc:
                    break

            # Log progress after each episode
            if i_episode % 1 == 0:
                self.cumulative_rewards.append(cumulative_reward)
                self.episodes.append(i_episode)
                print("Episode {}/{}, reward : {}".format(i_episode, num_episodes, cumulative_reward))

        return self.Q, policy

    def get_learning_curve(self):
        return self.episodes, self.cumulative_rewards
