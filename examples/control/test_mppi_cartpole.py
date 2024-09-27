from robotics_algorithm.env.cartpole_balance import CartPoleEnv
from robotics_algorithm.control.mppi import MPPI

# Initialize environment
env = CartPoleEnv()
env.reset()

controller = MPPI(env, action_mean=[0], action_std=[5.0])

# run controller
state = env.cur_state
while True:
    action = controller.run(state)

    next_state, reward, term, trunc, info = env.step(action)
    print(state, action, next_state, reward)

    env.render()
    state = next_state

    if term or trunc:
        break
