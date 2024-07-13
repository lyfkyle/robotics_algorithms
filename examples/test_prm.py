import time

from robotics_algorithm.env.grid_world_maze import GridWorldMaze
from robotics_algorithm.planning import ProbabilisticRoadmap

# Initialize environment
env = GridWorldMaze()

# -------- Settings ------------
FIX_MAZE = True


# -------- Helper Functions -------------
def sample_vertex():
    x, y = env.get_random_point()
    return x, y


def check_clear(vertex):
    x, y = vertex
    return env.maze[x, y] == env.FREE_SPACE


def check_link(v1, v2):
    local_path = compute_local_path(v1, v2)
    if local_path is None:
        return False, 0
    else:
        return True, len(local_path)


def compute_local_path(v1, v2):
    v1_x, v1_y = v1
    v2_x, v2_y = v2

    local_path = []
    path_exist = True

    # try x first then y
    if v1_x >= v2_x:
        for x in range(v1_x, v2_x - 1, -1):
            if env.maze[x, v1_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v1_y))
    else:
        for x in range(v1_x, v2_x + 1):
            if env.maze[x, v1_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v1_y))

    if v1_y >= v2_y:
        for y in range(v1_y - 1, v2_y - 1, -1):
            if env.maze[v2_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v2_x, y))
    else:
        for y in range(v1_y + 1, v2_y + 1):
            if env.maze[v2_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v2_x, y))

    if path_exist:
        return local_path

    path_exist = True
    local_path.clear()
    # try y first then x
    if v1_y >= v2_y:
        for y in range(v1_y, v2_y - 1, -1):
            if env.maze[v1_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v1_x, y))
    else:
        for y in range(v1_y, v2_y + 1):
            if env.maze[v1_x, y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((v1_x, y))

    if v1_x >= v2_x:
        for x in range(v1_x - 1, v2_x - 1, -1):
            if env.maze[x, v2_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v2_y))
    else:
        for x in range(v1_x + 1, v2_x + 1):
            if env.maze[x, v2_y] == env.OBSTACLE:
                path_exist = False
                break

            local_path.append((x, v2_y))

    if path_exist:
        return local_path

    return None


def compute_source():
    for x in reversed(range(env.size)):
        for y in range(env.size):
            if env.maze[x, y] == GridWorldMaze.FREE_SPACE:
                source = x, y
                return source


def compute_goal():
    for x in range(env.size):
        for y in reversed(range(env.size)):
            if env.maze[x, y] == GridWorldMaze.FREE_SPACE:
                goal = x, y
                return goal


# -------- Main Code ----------
# add default obstacles
env.add_default_obstacles()

# generate source and goal
# source_x, source_y = env.get_random_free_point()
# goal_x, goal_y = env.get_random_free_point()
# while goal_x == source_x and goal_y == source_y:
#     goal_x, goal_y = env.get_random_free_point()


# add source and goal to environment
source = compute_source()
goal = compute_goal()
# goal = (45, 0)
env.add_start(source)
env.add_goal(goal)
# path = compute_local_path(source, goal)
# env.add_path(path)
# env.plot()

# initialize planner
my_path_planner = ProbabilisticRoadmap(
    number_of_vertices=1000, K=10
)  # 1000 samples out of total 2500 vertex.

# offline portion of PRM
start_time = time.time()
my_path_planner.compute_roadmap(sample_vertex, check_clear, check_link)
end_time = time.time()
print("TestPRM, offline takes {} seconds".format(end_time - start_time))

# run path planner
start_time = time.time()
res, shortest_path, shortest_path_len = my_path_planner.get_path(
    source, goal, check_link
)
end_time = time.time()
print("TestPRM, online takes {} seconds".format(end_time - start_time))

# visualize roadmap
roadmap = my_path_planner.get_roadmap()
for vertex in roadmap:
    # for v in roadmap[vertex]:
    #     path = compute_local_path(vertex, v)
    #     if path is None:
    #         print("!!! this should not happen !!!")
    #         break
    #     else:
    #         env.add_path(path)
    if vertex != goal and vertex != source:
        env.add_point(vertex, env.WAYPOINT)

if not res:
    print("TestPRM, no path is available!")
else:
    # visualize path
    path = [source]
    v1 = source
    for v2 in shortest_path[1:]:
        local_path = compute_local_path(v1, v2)
        if local_path is None:
            print("!!! this should not happen !!!")
            break
        else:
            path += local_path[1:]  # ignore source of local path

        v1 = v2

    env.add_path(path)
    print("TestRRT, found path of len {}".format(len(path)))

env.plot()