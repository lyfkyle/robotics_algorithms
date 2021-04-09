#!/usr/bin/evn python

import numpy as np

GRID_HEIGHT = 7
GRID_WIDTH = 10

class WindyGridWorld():
    def __init__(self, start=(0,3), goal=(7,3), wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        """
        @param, start, the start position of agent
                goal, the goal position
                wind, the strength of upward wind in each horizontal position. 
                      The number decides how many grid cell the agent will be pushed upward.
        """
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

        if next_pos[0] == self.goal[0] and next_pos[1] == self.goal[1]:
            reward = self.goal_reward
            episode_over = True
        else:
            reward = self.step_reward
            episode_over = False
        
        self.cur_pos = next_pos

        return next_pos, reward, episode_over, None
    
    def reset(self):
        self.cur_pos = self.start
        return self.cur_pos
