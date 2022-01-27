from .mdp_env import MDP

class POMDP():
    def __init__(self) -> None:
        pass

    def transit_func(self, state, action):
        next_states = []
        probs = []
        done = False

        return next_states, probs, done
    
    def reward_func(self, state, action):
        reward = None

        return reward
    
    def observation_func(self, state):
        obs = []
        probs = []

        return obs, probs