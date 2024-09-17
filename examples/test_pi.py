from env.windy_gridworld import WindyGridWorld
from env.cliff_walking import CliffWalking

# env = WindyGridWorld()
env = CliffWalking()
pi = PolicyIteration()
Q, policy = pi.plan(env)

state = env.reset()
path = []
while True:
    ## choose action according to epsilon-greedy policy
    action_probs = policy(state)
    action = np.random.choice(env.actions, p=action_probs)  # choose action
    next_state, reward, done, _ = env.step(action)

    path.append(state)
    state = next_state

    if done:
        break

env.plot(path)
