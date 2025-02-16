import numpy as np

from robotics_algorithm.env.cartpole_balance import CartPoleEnv
from robotics_algorithm.control.classical_control.pid import PID

# Initialize environment
# NOTE: we can use PID for cartpole because when it is near the upright position, the system behaves like LTI system.
#       refer to https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling for
#       the transfer function
# Here we only control theta
env = CartPoleEnv()
env.reset()

# These gains are tuned using Ziegler–Nichols method
controller = PID(env, goal_state=env.goal_state[2], Kp=16.0, Kd=1.8)  # we only control theta

# run controller
state = env.cur_state
while True:
    action = controller.run(state[2])  # we only control theta
    action *= -1  # post-process action, needed because action and state are not in the same space.

    next_state, reward, term, trunc, info = env.step(action)
    print(state, action, next_state, reward)

    env.render()
    state = next_state

    if term or trunc:
        break
