from robotics_algorithm.env.one_d_world import DoubleIntegratorEnv
from robotics_algorithm.control.lqr import LQR

# Initialize environment
env = DoubleIntegratorEnv()

# -------- Settings ------------

# -------- Helper Functions -------------

# -------- Main Code ----------

env.reset()
print("cur_state: ", env.start_state)
env.render()

# initialize controller
controller = LQR(env)

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)
    next_state, reward, term, trunc, info = env.step(action)

    print(state, action, next_state)

    path.append(next_state)
    state = next_state

    if term or trunc:
        break

env.add_path(path)
env.render()
