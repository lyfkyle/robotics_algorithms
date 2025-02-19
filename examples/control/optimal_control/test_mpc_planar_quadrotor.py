import numpy as np

from robotics_algorithm.env.planar_quadrotor_hover import PlanarQuadrotorHoverEnv
from robotics_algorithm.control.optimal_control.convex_mpc import ConvexMPC

# discrete time model
env = PlanarQuadrotorHoverEnv(
    hover_pos=0.25, hover_height=1.0, quadratic_reward=True, term_if_constraints_violated=True
)
env.reset()
print('cur_state: ', env.cur_state)
env.render()

# initialize controller
controller = ConvexMPC(env, horizon=20)

# ! Add additional theta constraint so that linearization is always valid and thrust constraints are satisfied.
# ! Compared to LQR, we don't have to manually increase theta cost in Q any more.
x, u = controller.get_decision_variables()
state_low = env.state_space.low - env.goal_state + 1e-2
state_high = env.state_space.high - env.goal_state - 1e-2
action_low = env.action_space.low - env.goal_action + 1e-2
action_high = env.action_space.high - env.goal_action - 1e-2
constr = []
for t in range(controller.horizon + 1):
    constr += [x[t] <= state_high, x[t] >= state_low]
    constr += [u[t] <= action_high, u[t] >= action_low]

    constr += [
        x[t, 2] <= 0.174533,  # ~10 degree
        x[t, 2] >= -0.174533,  # ~10 degree
    ]  # We make theta tighter so that linearized dynamics is more valid

controller.add_constraints(constr)

# run controller
state = env.cur_state
path = [state]
while True:
    state_error = state - env.goal_state
    action_error = controller.run(state_error)
    action = env.goal_action + action_error

    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state, reward, term, trunc)

    env.render()

    state = next_state

    if term or trunc:
        break
