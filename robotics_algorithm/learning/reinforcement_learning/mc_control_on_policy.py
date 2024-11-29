
import numpy as np
from collections import defaultdict
from typing import Callable

from robotics_algorithm.env.base_env import MDPEnv, SpaceType, EnvType


class MCControlOnPolicy:
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

    def run(self, num_episodes: int = 500, discount_factor: float = 0.95, epsilon: float = 0.1) -> tuple[dict, Callable]:
        """
        Monte Carlo Control using Epsilon-Greedy policies.
        Finds an optimal epsilon-greedy policy.

        Args:
            num_episodes (int): Number of episodes to sample. Defaults to 500.
            discount_factor (float, optional): Gamma discount factor. Defaults to 0.95.
            epsilon (float, optional): Chance the sample a random action. Float between 0 and 1. Defaults to 0.1.

        Returns:
            tuple[dict, Callable]: A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns action probabilities.
        """
        # Keeps track of sum and count of returns for each state
        # to calculate an average. We could use an array to save all
        # returns (like in the book) but that's memory inefficient.
        state_action_visit_cnt = defaultdict(int)

        # for plotting
        self.episodes = []
        self.cumulative_rewards = []

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.size))

        # The policy we're following
        policy = self.make_epsilon_greedy_policy(epsilon, self.env.action_space.size)

        # Learn
        for i_episode in range(1, num_episodes + 1):
            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            trajectory = []
            cumulative_reward = 0
            state, _ = self.env.reset()

            # Run the episode
            for steps in range(self._max_episode_len):
                # choose action according to epsilon-greedy policy
                action_probs = policy(state)
                action = np.random.choice(self.env.action_space.get_all(), p=action_probs)  # choose action
                next_state, reward, term, trunc, _ = self.env.step(action)
                trajectory.append((state, action, reward))

                steps += 1
                cumulative_reward += reward

                if term or trunc:
                    break

                state = tuple(next_state)

            # every-visit MC Prediction
            # Find all states the we've visited in this episode
            # We convert each state to a tuple so that we can use it as a dict key
            state_action_in_episode = set(
                [(tuple(x[0]), x[1]) for x in trajectory]
            )  # we use set because it is first-visit
            for state_action in state_action_in_episode:
                state, action = state_action

                # calculate total return for that state:
                first_idx = next(i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action)
                # calculate total return
                total_return = sum([x[2] * (discount_factor**i) for i, x in enumerate(trajectory[first_idx:])])
                # calculate average return in incremental fashion
                state_action_visit_cnt[state_action] += 1
                self.Q[state][action] = self.Q[state][action] + (1.0 / state_action_visit_cnt[state_action]) * (
                    total_return - self.Q[state][action]
                )

            # improve policy based on updated Q function
            # NOTE: we do not need to call this because Q function value is already updated
            # policy = self.make_epsilon_greedy_policy(epsilon, self.env.action_space.size)

            # Print out which episode we're on, useful for debugging.
            if i_episode % 1 == 0:
                self.cumulative_rewards.append(cumulative_reward)
                self.episodes.append(i_episode)
                print("Episode {}/{}, reward : {}".format(i_episode, num_episodes, cumulative_reward))

        return self.Q, policy

    def get_learning_curve(self):
        return self.episodes, self.cumulative_rewards
