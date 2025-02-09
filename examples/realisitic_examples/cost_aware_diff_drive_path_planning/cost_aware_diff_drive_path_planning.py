"""Example to plan a path for differential drive robot with added cost to encourage path planner to stay away from
obstacles.

It is often desired to have larger clearance from environment obstacles, therefore, we can add a cost value to each
map cell based on the distance to nearest obstacle. The distance is now weighted by this cost to encourage path planner
to find a path that stays away from obstacles.

In this example, we in addition use the original heuristics proposed by Hybrid A* to guide the path planner. The
heuristic is calculated as the maximum of two sub heuristic function, one ignoring obstacles but considering kinematics, and
another ignoring kinematics but considering obstacles.
"""

import time
import math
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

from robotics_algorithm.env.continuous_2d.diff_drive_2d_planning import DiffDrive2DPlanningWithCost
from robotics_algorithm.planning import HybridAStar
from robotics_algorithm.utils import math_utils

# Initialize environment
env = DiffDrive2DPlanningWithCost(discrete_action=True)  # use discrete action for Hybrid A*

# -------- Settings ------------
FIX_MAZE = True
COST_PENALTY = 3.0  # set this to 0 to disable cost penalty
OBST_CACHE_RES = 0.1


# -------- Helper Functions -------------
def heuristic_func(state, goal):
    """
    The heuristic function for Hybrid A*.

    Hybrid A star proposes to use the maximum of two heuristic functions:
    - A distance heuristic, which is the distance between state and goal considering kinematics but not obstacles.
    - An obstacle heuristic, which is the distance between state and goal considering obstacles but not kinematic.

    Args:
        state: A tuple of (x, y, theta) representing the state.
        goal: A tuple of (x, y, theta) representing the goal.

    Return:
        The maximum of the two heuristic values.
    """
    # Compute the distance heuristic, ignore obstacle but consider kinematics
    dist_h = distance_heuristic(state, goal)
    # Compute the obstacle heuristic, ignore kinematics but consider obstacles
    obst_h = obstacle_heuristic(state, goal)
    # Return the maximum of the two heuristic values
    return max(dist_h, obst_h)


def distance_heuristic(state, goal):
    # For car-like robots, this is often from Dubin or Reeds-shepp curve. For diff drive, we can simply use the distance
    # between v and goal in SE2 space.
    v_x, v_y, v_theta = state
    goal_x, goal_y, g_theta = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2) + 0.5 * math_utils.normalize_angle(v_theta - g_theta)


def obstacle_heuristic(state, goal):
    return obst_h_cache[get_obstacle_cache_indices(state[0], state[1])]


def get_obstacle_cache_indices(x, y):
    return (math.floor(x / OBST_CACHE_RES), math.floor(y / OBST_CACHE_RES))


def precompute_obstacle_heuristic(env: DiffDrive2DPlanningWithCost, goal):
    """
    Precompute the obstacle heuristic. The heuristic is computed using Breadth First Search. The heuristic is the
    minimum cost weighted distance to the goal state.

    Args:
        env (DiffDrive2DPlanningWithCost): The planning environment.
        goal (tuple): A tuple of (x, y, theta) representing the goal.

    Returns:
        A dictionary where the key is the state key and the value is the obstacle heuristic.
    """

    # element is a tuple of (x, y, previous value)
    dq = deque()

    # The cache of the obstacle heuristic
    obst_h_size = round(env.size / OBST_CACHE_RES)
    obst_h_cache = np.full((obst_h_size, obst_h_size), -1.0, dtype=np.float32)
    gx, gy = get_obstacle_cache_indices(goal[0], goal[1])
    # The heuristic value of the goal state is 0
    obst_h_cache[gx, gy] = 0

    # Initialize the queue with the 4 neighbors of the goal state
    dq.append((gx + 1, gy, 0))
    dq.append((gx - 1, gy, 0))
    dq.append((gx, gy + 1, 0))
    dq.append((gx, gy - 1, 0))

    while len(dq) > 0:
        # Get the next element from the queue
        x, y, prev_value = dq.popleft()

        # If the state is not within the environment, skip it
        if x < 0 or x >= obst_h_size or y < 0 or y >= obst_h_size:
            continue

        # if computed, skip
        if obst_h_cache[x, y] >= 0:
            continue

        state = (x * OBST_CACHE_RES, y * OBST_CACHE_RES, 0)
        # If the state is not valid, distance is infinity
        if not env.is_state_valid(state):
            obst_h_cache[x, y] = np.inf

        # else, calculate the cost weighted distance to the state
        else:
            cost_weighted_dist = OBST_CACHE_RES * (1.0 + COST_PENALTY * env.get_cost(state) / env.max_cost)
            cur_value = prev_value + cost_weighted_dist

            # Add the state to the cache
            obst_h_cache[x, y] = cur_value

        # Add the 4 neighbors of the state to the queue
        dq.append((x + 1, y, cur_value))
        dq.append((x - 1, y, cur_value))
        dq.append((x, y + 1, cur_value))
        dq.append((x, y - 1, cur_value))

    return obst_h_cache


def state_key_func(state):
    return env.calc_state_key(state)


# -------- Main Code ----------
env.reset()
env.cost_penalty = COST_PENALTY
start = env.start_state
goal = env.goal_state
env.render()

# initialize planner
planner = HybridAStar(env, heuristic_func, state_key_func)
obst_h_cache = precompute_obstacle_heuristic(env, env.goal_state)

# visualize obstacle heuristic
plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(obst_h_cache, cmap='hot', alpha=0.5, origin='lower', extent=[0, 10, 0, 10])
plt.title('Obstacle Heuristic')
plt.show()

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = planner.run(start, goal)
end_time = time.time()
print('TestHybridAStar, takes {} seconds'.format(end_time - start_time))

if not res:
    print('TestHybridAStar, no path is available!')
else:
    print('TestHybridAStar, found path of len {}'.format(shortest_path_len))
    # visualize path
    path = []
    for v in shortest_path:
        path.append(v)

    env.add_action_path(path)

env.render(title='path_with_cost_penalty')
