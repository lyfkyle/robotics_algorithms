import numpy as np

from robotics_algorithm.control.optimal_control.convex_mpc import ConvexMPC
from robotics_algorithm.env.cartpole_balance import CartPoleEnv


env = CartPoleEnv(quadratic_reward=True)
env.reset()
print('cur_state: ', env.cur_state)
env.render()

# initialize controller
controller = ConvexMPC(env, horizon=20)

# run controller
state = env.cur_state
path = [state]
while True:
    action = controller.run(state)
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward, term, trunc, info, env.step_cnt)
    env.render()

    state = next_state

    if term or trunc:
        break
