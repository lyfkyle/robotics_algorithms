import numpy as np

from robotics_algorithm.env.continuous_world_2d import DiffDrive2DEnvComplex
from robotics_algorithm.state_estimation.particle_filter import ParticleFilter


def spiral_velocity(spiral_radius, spiral_growth_rate, time, linear_velocity):
    """
    Calculate the linear velocity and angular velocity of a differential robot to drive in a spiral.

    Args:
        spiral_radius (float): The initial radius of the spiral.
        spiral_growth_rate (float): The rate at which the spiral's radius grows.
        time (float): The current time.
        linear_velocity (float): The desiredlinear velocity of the robot.

    Returns:
        tuple: The linear velocity and angular velocity of the robot.
    """
    # Calculate the current radius of the spiral
    current_radius = spiral_radius + spiral_growth_rate * time

    # Calculate the angular velocity
    angular_velocity = linear_velocity / current_radius

    return linear_velocity, angular_velocity


# Initialize environment
env = DiffDrive2DEnvComplex(action_dt=0.1, obs_noise_std=[0.1, 0.1, 0.1])
obs, _ = env.reset(empty=True)
# Manually override start state and start observation
env.start_state = [env.size / 2, env.size / 2, 0]
env.cur_state = [env.size / 2, env.size / 2, 0]
obs = [env.size / 2, env.size / 2, 0]
env.render()

# Initialize filter
filter = ParticleFilter(env, num_of_particles=250)
filter.set_initial_state(env.cur_state)

# Add initial state
# Step env with random actions
true_states = []
filter_states = []
obss = []
true_states.append(env.cur_state)
filter_states.append(filter.get_state())
obss.append(obs)

max_steps = 500
for i in range(max_steps):
    print(i)
    # action = [random.uniform(0.0, 0.5), random.uniform(0, 0.5)]
    action = spiral_velocity(1.0, 0.01, i * env.action_dt, 0.2)
    new_obs, reward, term, trunc, info = env.step(action)

    filter.run(action, new_obs)

    true_states.append(env.cur_state)
    filter_states.append(filter.get_state())
    obss.append(new_obs)

    if term or trunc:
        break

# print(true_states)
# print(filter_states)

# calculate RMSE
true_states = np.array(true_states)
filter_states = np.array(filter_states)
rmse = np.sqrt(np.mean((true_states - filter_states) ** 2, axis=0))
print("RMSE: {}".format(rmse))

env.add_state_path(true_states, id="groundtruth")
env.add_state_path(obss, id="observed")
env.add_state_path(filter_states, id="predicted")
env.render()
