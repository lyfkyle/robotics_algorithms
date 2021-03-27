#!/usr/bin/evn python

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors

GRID_HEIGHT = 4
GRID_WIDTH = 9
OBSTACLES = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]

class CliffWalking():
    def __init__(self, start=(0,0), goal=(7, 0), obstacles=OBSTACLES):
        """
        @param, start, the start position of agent
                goal, the goal position
        """
        self.states = [(i, j) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)]
        self.actions = [0, 1, 2, 3]
        self.action_space_size = len(self.actions)
        self.state_space_size = len(self.states)
        self.start = start
        self.cur_pos = start
        self.goal = goal
        self.obstacles = obstacles
        self.step_reward = -1
        self.obstacle_reward = -100
        self.goal_reward = 0

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
        i, j = self.cur_pos
        goal_reached = False

        tmp = random.randint(1, 11) # random int from 1 to 10
        if action == 0:
            if tmp <= 8:
                next_pos = (i, min(j + 1, GRID_HEIGHT -1))
            elif tmp == 9:
                next_pos = (max(0, i - 1), min(j + 1, GRID_HEIGHT -1))
            else:
                next_pos = (min(i + 1, GRID_WIDTH - 1), min(j + 1, GRID_HEIGHT -1))
        elif action == 1:
            if tmp <= 8:
                next_pos = (min(i + 1, GRID_WIDTH - 1), j)
            elif tmp == 9:
                next_pos = (min(i + 1, GRID_WIDTH - 1), min(j + 1, GRID_HEIGHT -1))
            else:
                next_pos = (min(i + 1, GRID_WIDTH - 1), max(0, j - 1))
        elif action == 2:
            if tmp <= 8:
                next_pos = (i, max(0, j - 1))
            elif tmp == 9:
                next_pos = (max(0, i - 1), max(0, j - 1))
            else:
                next_pos = (min(i + 1, GRID_WIDTH - 1), max(0, j - 1))
        elif action == 3:
            if tmp <= 8:
                next_pos = (max(0, i - 1), j)
            elif tmp == 9:
                next_pos = (max(0, i - 1), min(j + 1, GRID_HEIGHT -1))
            else:
                next_pos = (max(0, i - 1), max(0, j - 1))
        if next_pos[0] == self.goal[0] and next_pos[1] == self.goal[1]:
            reward = self.goal_reward
            episode_over = True
            goal_reached = True
        elif next_pos in self.obstacles:
            reward = self.obstacle_reward
            episode_over = True
        else:
            reward = self.step_reward
            # reward = self.goal_reward - abs(self.goal[0] - next_pos[0]) - abs(self.goal[1] - next_pos[1])
            episode_over = False
        
        self.cur_pos = next_pos

        return next_pos, reward, episode_over, goal_reached
    
    def reset(self):
        self.cur_pos = self.start
        return self.cur_pos

    def plot(self, path=[]):
        fig, ax = plt.subplots()
        self.gridworld = np.full((GRID_HEIGHT, GRID_WIDTH), 0)
        for obstacle in self.obstacles:
            self.gridworld[GRID_HEIGHT - obstacle[1] - 1][obstacle[0]] = 1
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