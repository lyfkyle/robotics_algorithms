import numpy as np

from robotics_algorithm.env.classic_mdp.windy_grid_world import WindyGridWorld
from robotics_algorithm.env.classic_mdp.cliff_walking import CliffWalking
from robotics_algorithm.env.classic_mdp.frozen_lake import FrozenLake
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
    all_actions = env.action_space.get_all()
    while True:
        # choose action according to epsilon-greedy policy
        action_probs = policy(state)
        action_idx = np.random.choice(np.arange(len(all_actions)), p=action_probs)  # choose action
        action = all_actions[action_idx]

        next_state, reward, term, trunc, info = env.step(action)
        env.render()

        print(state, action, reward)

        path.append(state)
        state = next_state

        if term or trunc:
            break

    # env.add_path(path)
    env.render()