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

def make_epsilon_greedy_policy(Q, epsilon, nA):
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

    def policy_fn(observation):
        ## epsilon probability of picking a random action.
        action_prob = np.ones(nA, dtype = float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        ## 1 - epsilon probability of picking the best action
        action_prob[best_action] += (1 - epsilon)
        return action_prob

    return policy_fn

def mc_prediction(policy, Q, discount_factor, num_episodes = 100000):
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end = "")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        observation = env.reset()
        ## run a trajectory of length 100.
        for t in range(100):
            ## choose action according to epsilon-greedy policy
            probs = policy(observation)
            action = np.random.choice(np.arange(len(probs)), p = probs)  # choose action
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_observation

        ## First visit MC Prediction

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        state_action_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state_action in state_action_in_episode:
            state, action = state_action
            ## calculate total return for that state:
            first_idx = next(i for i, x in enumerate(episode)
                                   if x[0] == state and x[1] == action)
            ## calcualte total return
            total_return = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_idx:])])
            ## calculate average return
            returns_sum[state_action] += total_return
            returns_count[state_action] += 1.0
            Q[state][action] = returns_sum[state_action] / returns_count[state_action]

def mc_control_epsilon_greedy(env, num_episodes, discount_factor = 1.0, epsilon = 0.1):
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
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    ## TODO
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end = "")
            sys.stdout.flush()

        mc_prediction(policy, Q, discount_factor)

        # The policy is improved implicitly by changing the Q dictionary
    '''

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end = "")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        observation = env.reset()
        ## run a trajectory of length 100.
        for t in range(100):
            ## choose action according to epsilon-greedy policy
            probs = policy(observation)
            action = np.random.choice(np.arange(len(probs)), p = probs)  # choose action
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_observation

        ## First visit MC Prediction

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        state_action_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state_action in state_action_in_episode:
            state, action = state_action
            ## calculate total return for that state:
            first_idx = next(i for i, x in enumerate(episode)
                                   if x[0] == state and x[1] == action)
            ## calcualte total return
            total_return = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_idx:])])
            ## calculate average return
            returns_sum[state_action] += total_return
            returns_count[state_action] += 1.0
            Q[state][action] = returns_sum[state_action] / returns_count[state_action]
    '''
    return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes = 5, epsilon = 0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title = "Optimal Value Function")
