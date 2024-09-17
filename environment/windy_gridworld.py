#!/usr/bin/evn python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

GRID_HEIGHT = 7
GRID_WIDTH = 10

from .mdp_env import MDP

class WindyGridWorld(MDP):
    def __init__(self, start=(0,3), goal=(7,3), wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        """
        @param, start, the start position of agent
                goal, the goal position
                wind, the strength of upward wind in each horizontal position. 
                      The number decides how many grid cell the agent will be pushed upward.
        """
        MDP.__init__(self)

        self.states = [(i, j) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)]
        self.actions = [0, 1, 2, 3]
        self.action_space_size = len(self.actions)
        self.state_space_size = len(self.states)
        self.start = start
        self.cur_pos = start
        self.goal = goal
        self.wind = wind
        self.step_reward = -1
        self.goal_reward = 10

    def transit_func(self, state, action):
        # This environment has deterministic transition 
        i, j = state

        if state[0] == self.goal[0] and state[1] == self.goal[1]:
            episode_over = True
        else:
            episode_over = False

        next_states = []
        probs = []
        
        if not episode_over:
            if action == 0:
                next_pos = (i, min(j + 1 + self.wind[i], GRID_HEIGHT -1))
            elif action == 1:
                new_i = min(i + 1, GRID_WIDTH - 1)
                next_pos = (new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1))
            elif action == 2:
                next_pos = (i, max(0, min(j - 1 + self.wind[i], GRID_HEIGHT - 1)))
            elif action == 3:
                new_i = max(i - 1, 0) 
                next_pos = (new_i, min(j + self.wind[new_i], GRID_HEIGHT - 1))
            
            next_states.append(next_pos)
            probs.append(1.0)

        return next_states, probs, episode_over

    def reward_func(self, state, action):
        if state[0] == self.goal[0] and state[1] == self.goal[1]:
            reward = self.goal_reward
        else:
            reward = self.step_reward

        return reward

    def step(self, action):
        """
        @param, action, 0: up, 1: right, 2: down, 3: left

        @return, next_pos, reward, episode_over, info : tuple
            obs (tuple) :
                 Agent current position in the grid.
            reward (float) :
                 Reward is -1 at every step.
            episode_over (bool) :
                 True if the agent reaches the goal, False otherwise.
            info (dict) :
                 Contains no additional information.
        """
        state = self.cur_pos
        next_states, probs, episode_over = self.transit_func(state, action)
        if not episode_over:
            next_state_idx = np.random.choice(np.arange(len(next_states)), p = probs)  # choose next_state
            next_state = next_states[next_state_idx]
            self.cur_pos = next_state
        else:
            next_state = state
        reward = self.reward_func(state, action)

        return next_state, reward, episode_over, None
    
    def reset(self):
        self.cur_pos = self.start
        return self.cur_pos

    def plot(self, path=[]):
        _, ax = plt.subplots()
        self.gridworld = np.full((GRID_HEIGHT, GRID_WIDTH), 0)
        for state in path:
            self.gridworld[GRID_HEIGHT - state[1] - 1][state[0]] = 2

        self.colour_map = colors.ListedColormap(['white', 'black', 'yellow'])
        bounds = [0, 1, 2, 3]
        self.norm = colors.BoundaryNorm(bounds, self.colour_map.N)

        ax.imshow(self.gridworld, cmap=self.colour_map, norm=self.norm)
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(0.5, GRID_WIDTH, 1))
        ax.set_xticklabels(np.array([str(i) for i in range(GRID_WIDTH)]))
        ax.set_yticks(np.arange(0.5, GRID_HEIGHT, 1))
        ax.set_yticklabels(np.array([str(i) for i in range(GRID_HEIGHT)]))
        # ax.axis('off')
        plt.tick_params(axis='both', labelsize=5, length = 0)

        # fig.set_size_inches((8.5, 11), forward=False)
        plt.show()
