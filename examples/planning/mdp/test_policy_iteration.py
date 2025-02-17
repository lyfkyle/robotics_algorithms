import numpy as np

from robotics_algorithm.env.windy_grid_world import WindyGridWorld
from robotics_algorithm.env.cliff_walking import CliffWalking
from robotics_algorithm.env.frozen_lake import FrozenLake
from robotics_algorithm.planning.mdp.policy_iteration import PolicyIteration

envs = [
    FrozenLake(dense_reward=True),
    CliffWalking(),
    WindyGridWorld()
]

for env in envs:
    state, _ = env.reset()
    env.render()

    # Plan
    planner = PolicyIteration(env)
    Q, policy = planner.run()

    print(Q)

    # Execute
    path = []
    while True:
        # choose action according to epsilon-greedy policy
        action_probs = policy(state)
        action = np.random.choice(env.action_space.get_all(), p=action_probs)  # choose action
        next_state, reward, term, trunc, info = env.step(action)
        env.render()

        print(state, action, reward)

        path.append(state)
        state = next_state

        if term or trunc:
            break

    # env.add_path(path)
    env.render()