import numpy as np

from robotics_algorithm.env.planar_quadrotor_hover import PlanarQuadrotorHoverEnv
from robotics_algorithm.control.optimal_control.lqr import LQR

# discrete time model
env = PlanarQuadrotorHoverEnv(term_if_constraints_violated=False) # ! LQR can't deal with constraints
env.reset()
print('cur_state: ', env.cur_state)
env.render()

# initialize controller
controller = LQR(env, discrete_time=True)

# run controller
state = env.cur_state
path = [state]
while True:
    state_error = state - env.goal_state
    action = controller.run(state_error)
    action = env.goal_action + action

    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward, term, trunc)
    env.render()

    state = next_state

    if term or trunc:
        break


