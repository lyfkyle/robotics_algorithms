import time
import math
import numpy as np
from collections import deque

from robotics_algorithm.env.continuous_world_2d import DiffDrive2DPlanningWithCost
from robotics_algorithm.planning import HybridAStar
from robotics_algorithm.utils import math_utils

# Initialize environment
env = DiffDrive2DPlanningWithCost(discrete_action=True)  # use discrete action for Hybrid A*

# -------- Settings ------------
FIX_MAZE = True
COST_PENALTY = 50.0


# -------- Helper Functions -------------
def heuristic_func(state, goal):
    """
    The heuristic function for Hybrid A*.

    Hybrid A star proposes to use the maximum of two heuristic functions:
    - A distance heuristic, which is the distance between state and goal considering kinematics but not obstacles.
    - An obstacle heuristic, which is the distance betwnne state and goal considering obstacles but not kinematic.

    Args:
        state: A tuple of (x, y, theta) representing the state.
        goal: A tuple of (x, y, theta) representing the goal.

    Return:
        The maximum of the two heuristic values.
    """
    # Compute the distance heuristic
    dist_h = distance_heuristic(state, goal)
    # Compute the obstacle heuristic
    obst_h = obstacle_heuristic(state, goal)
    # Return the maximum of the two heuristic values
    return max(dist_h, obst_h)


def distance_heuristic(state, goal):
    # simply the distance between v and goal
    v_x, v_y, v_theta = state
    goal_x, goal_y, g_theta = goal
    return math.sqrt((goal_x - v_x) ** 2 + (goal_y - v_y) ** 2) + 0.5 * math_utils.normalize_angle(v_theta - g_theta)


def obstacle_heuristic(state, goal):
    return obst_h_cache[get_obstacle_cache_key(state[0], state[1])]


def get_obstacle_cache_key(x, y):
    return (round(x / obst_cache_res), round(y / obst_cache_res))


def precompute_obstacle_heuristic(env: DiffDrive2DPlanningWithCost, goal):
    """
    Precompute the obstacle heuristic. The heuristic is computed using Dijkstra's algorithm. The heuristic is the minimum cost
    weighted distance to the goal state.

    Args:
        env (DiffDrive2DPlanningWithCost): The planning environment.
        goal (tuple): A tuple of (x, y, theta) representing the goal.

    Returns:
        A dictionary where the key is the state key and the value is the obstacle heuristic.
    """

    # The queue for Dijkstra's algorithm. The element is a tuple of (x, y, previous value)
    dq = deque()

    # The cache of the obstacle heuristic
    obst_h_cache = {}
    # The heuristic value of the goal state is 0
    obst_h_cache[get_obstacle_cache_key(goal[0], goal[1])] = 0

    # Initialize the queue with the 4 neighbors of the goal state
    gx, gy = goal[0], goal[1]
    dq.append((gx + obst_cache_res, gy, 0))
    dq.append((gx - obst_cache_res, gy, 0))
    dq.append((gx, gy + obst_cache_res, 0))
    dq.append((gx, gy - obst_cache_res, 0))

    while len(dq) > 0:
        # Get the next element from the queue
        x, y, prev_value = dq.popleft()

        # If the state is not within the environment, skip it
        if x < 0 or x > env.size or y < 0 or y > env.size:
            continue

        # Get the key of the state
        xy_key = get_obstacle_cache_key(x, y)

        # If the state is not valid, skip it
        if not env.is_state_valid((x, y)):
            obst_h_cache[xy_key] = float('inf')
            continue

        # If the state is not in the cache, add it
        if xy_key not in obst_h_cache:
            # Calculate the cost weighted distance to the state
            cost_weighted_dist = obst_cache_res * (1.0 + COST_PENALTY * env.get_cost((x, y)) / env.max_cost)
            cur_value = prev_value + cost_weighted_dist

            # Add the state to the cache
            obst_h_cache[xy_key] = cur_value

            # Add the 4 neighbors of the state to the queue
            dq.append((x + obst_cache_res, y, cur_value))
            dq.append((x - obst_cache_res, y, cur_value))
            dq.append((x, y + obst_cache_res, cur_value))
            dq.append((x, y - obst_cache_res, cur_value))

    return obst_h_cache


def state_key_func(state):
    return env.calc_state_key(state)


# -------- Main Code ----------
env.reset(random_env=not FIX_MAZE)
env.cost_penalty = COST_PENALTY
start = env.start_state
goal = env.goal_state
env.render()

# initialize planner
# a_star_planner = AStar(env, distance_heuristic, state_key_func)
planner = HybridAStar(env, heuristic_func, state_key_func)

obst_cache_res = 0.1
obst_h_cache = precompute_obstacle_heuristic(env, env.goal_state)

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

env.render()
