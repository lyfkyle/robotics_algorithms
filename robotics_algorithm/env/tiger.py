
import numpy as np

from robotics_algorithm.env.base_env. import POMDPEnv

class Tiger(POMDPEnv):
    # action
    OPEN_LEFT = 0
    OPEN_RIGHT = 1
    LISTEN = 2

    # state
    TIGER_AT_LEFT = 0
    TIGER_AT_RIGHT = 1

    # observations
    HEARD_TIGER_LEFT = 0
    HEARD_TIGER_RIGHT = 1

    def __init__(self) -> None:
        super().__init__()

        self.states = [Tiger.TIGER_AT_LEFT, Tiger.TIGER_AT_RIGHT]
        self.actions = [Tiger.OPEN_LEFT, Tiger.OPEN_RIGHT, Tiger.LISTEN]
        self.observations = [Tiger.HEARD_TIGER_LEFT, Tiger.HEARD_TIGER_RIGHT]

        self.state_space_size = len(self.states)
        self.action_space_size = len(self.actions)
        self.observation_space_size = len(self.observations)

        self.reset()

    def reset(self):
        self.cur_state = np.random.choice(self.states)

        return self.cur_state

    def transit_func(self, state, action):
        next_states = []
        probs = []

        episode_over = False
        if action != Tiger.LISTEN:
            episode_over = True

        if not episode_over:
            if action == Tiger.LISTEN:
                # listen does not change state
                next_states.append(state)
                probs.append(1.0)

        return next_states, probs, episode_over

    def reward_func(self, state, action):
        reward = 0
        if action == Tiger.LISTEN:
            reward = -1
        elif action == Tiger.OPEN_LEFT:
            if state == Tiger.TIGER_AT_LEFT:
                reward = -100
            else:
                reward = 10
        else:
            if state == Tiger.TIGER_AT_LEFT:
                reward = 10
            else:
                reward = -100

        return reward

    def observation_func(self, state):
        obs = []
        probs = []

        if state == Tiger.TIGER_AT_LEFT:
            obs = [Tiger.HEARD_TIGER_LEFT, Tiger.HEARD_TIGER_RIGHT]
            probs = [0.85, 0.15]
        else: # state == Tiger.TIGER_AT_RIGHT
            obs = [Tiger.HEARD_TIGER_LEFT, Tiger.HEARD_TIGER_RIGHT]
            probs = [0.15, 0.85]

        return obs, probs

    def step(self, action):
        state = self.cur_state
        next_states, probs, episode_over = self.transit_func(state, action)
        if not episode_over:
            next_state_idx = np.random.choice(np.arange(len(next_states)), p = probs)  # choose next_state
            next_state = next_states[next_state_idx]
            self.cur_state = next_state
        else:
            next_state = state
        reward = self.reward_func(state, action)

        if not episode_over:
            observations, obs_probs = self.observation_func(next_state)
            obs_idx = np.random.choice(np.arange(len(observations)), p = obs_probs)  # choose observations
            obs = observations[obs_idx]
        else:
            obs = None

        return obs, reward, episode_over, None
