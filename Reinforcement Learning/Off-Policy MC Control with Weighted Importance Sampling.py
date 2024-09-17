import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype = float) / nA

    def policy_fn(observation):
        return A

    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype = float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A

    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor = 1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end = "")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        observation = env.reset()
        for t in range(100):
            probs = behavior_policy(observation)
            action = np.random.choice(np.arange(len(probs)), p = probs)  # choose action
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_observation

        ## Every-visit MC with weighted importance sampling

        # Sum of discounted returns
        total_reward = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            total_reward = discount_factor * total_reward + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (total_reward - Q[state][action])
            # If the action taken by the behavior policy is not the action
            # taken by the target policy the weight will be 0 and we can break

            ## Explanation: Here remember weight is calculated as a product of
            ## a seris of target[state][action] / behaviour[state][action]. Therefore, if
            ## any action is different for target policy and behaviour policy, the overall
            ## weight will be zero for this action and all actions before that.
            if action != np.argmax(target_policy(state)):
                break
            W = W * 1.0 / behavior_policy(observation)[action]

            # The policy is improved implicitly by changing the Q dictionary

    return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes = 500000, behavior_policy = random_policy)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title = "Optimal Value Function")
