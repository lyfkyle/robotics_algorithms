import matplotlib.pyplot as plt
import numpy as np

from robotics_algorithm.env.discrete_world_1d import DiscreteWorld1D
from robotics_algorithm.state_estimation.discrete_bayes_filter import DiscreteBayesFilter

# env
env = DiscreteWorld1D()
obs, _ = env.reset()
env.render()

# filter
filter = DiscreteBayesFilter(env)
filter.set_initial_state(env.cur_state)

# Step env with random actions
true_states = []
filter_states = []
obss = []
true_states.append(env.cur_state)
filter_states.append(filter.get_state())
obss.append(obs)
max_steps = 100
for i in range(max_steps):
    action = env.action_space.sample()
    new_obs, reward, term, trunc, info = env.step(action)

    filter.run(action, new_obs)

    true_states.append(env.cur_state)
    filter_states.append(filter.get_state())
    obss.append(new_obs)

    if term or trunc:
        break


# calculate RMSE
true_states = np.array(true_states)
filter_states = np.array(filter_states)
rmse = np.sqrt(np.mean((true_states - filter_states) ** 2, axis=0))
print("RMSE: {}".format(rmse))

# plot result
env_steps = len(true_states)
t = np.arange(env_steps)
fig = plt.figure()
plt.plot(t, [true_states[i][0] for i in range(env_steps)], "k", label="groundtruth")
plt.plot(t, [filter_states[i][0] for i in range(env_steps)], "b", label="predicted")
plt.plot(t, [obss[i][0] for i in range(env_steps)], "r", label="observed")
plt.ylabel("position")
plt.legend()
plt.show()