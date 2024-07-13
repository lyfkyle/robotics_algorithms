import sys
import os
import numpy as np


class BeliefTreeSearch:
    def __init__(self) -> None:
        pass

    def belief_update(self, env, belief, action, obs):
        # bayesian filter
        new_belief = np.ones(env.state_space_size, dtype=np.float)
        for state_idx, state in enumerate(env.states):
            # Step 1: calculate prior
            prior = 0.0
            for prev_state_idx, prev_state in enumerate(env.states):
                next_states, next_state_probs, episode_over = env.transit_func(
                    prev_state, action
                )
                if episode_over:
                    continue

                if state in next_states:
                    idx = next_states.index(state)
                    probs = next_state_probs[idx]
                else:
                    probs = 0
                prior += probs * belief[prev_state_idx]

            # Step 2: calculate likelihood
            possible_obs, obs_probs = env.observation_func(state)
            idx = possible_obs.index(obs)
            obs_llh = obs_probs[idx]

            new_belief[state_idx] = prior * obs_llh

        new_belief /= new_belief.sum()

        # print(belief, action, obs, new_belief)

        return new_belief

    def get_p_obs_ba(self, env, belief, action, obs):
        p_obs_ba = 0
        for state_idx, state in enumerate(env.states):
            # Step 1: calculate prior
            prior = 0.0
            for prev_state_idx, prev_state in enumerate(env.states):
                next_states, next_state_probs, episode_over = env.transit_func(
                    prev_state, action
                )
                if episode_over:
                    continue

                if state in next_states:
                    idx = next_states.index(state)
                    probs = next_state_probs[idx]
                else:
                    probs = 0
                prior += probs * belief[prev_state_idx]

            # Step 2 calculate llh
            possible_obs, obs_probs = env.observation_func(state)
            idx = possible_obs.index(obs)
            obs_llh = obs_probs[idx]

            p_obs_ba += prior * obs_llh

        return p_obs_ba

    def get_belief_value(self, env, belief, cur_depth):
        q_b = np.zeros(env.action_space_size)

        if cur_depth <= self.max_depth:
            # sample action
            for action_idx, action in enumerate(env.actions):

                # reward of applying action
                r_ba = 0
                for state_idx, state in enumerate(env.states):
                    r_ba += belief[state_idx] * env.reward_func(state, action)

                # sample observation
                future_reward = 0
                for obs in env.observations:
                    # p(obs | b, a)
                    p_obs_ba = self.get_p_obs_ba(env, belief, action, obs)

                    # ignore observations that are not applicable
                    ## this can happen for example if the action terminates the process
                    if p_obs_ba < 1e-5:
                        continue

                    # belief update
                    new_belief = self.belief_update(env, belief, action, obs)

                    # calculate belief value
                    _, value = self.get_belief_value(env, new_belief, cur_depth + 1)

                    future_reward += p_obs_ba * value

                q_b[action_idx] = r_ba + self.discount_factor * future_reward

        belief_value = q_b.max()
        return q_b, belief_value

    def plan(self, env, discount_factor=0.99, max_depth=4):
        self.max_depth = max_depth
        self.discount_factor = discount_factor

        def policy_fn(belief):
            q, _ = self.get_belief_value(env, belief, 0)
            action_probs = np.zeros(env.action_space_size)
            action_probs[q.argmax()] = 1.0
            return action_probs

        return policy_fn


if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

    from env.tiger import Tiger

    env = Tiger()
    bts = BeliefTreeSearch()
    policy = bts.plan(env)

    _ = env.reset()
    path = []
    belief = (
        np.ones(env.state_space_size, dtype=np.float) / env.state_space_size
    )  # uniform belief over states
    while True:
        action_probs = policy(belief)
        # print(action_probs)
        action = np.random.choice(env.actions, p=action_probs)  # choose action
        obs, reward, done, _ = env.step(action)

        print(action)
        print(obs)
        print(reward)

        if done:
            break

        belief = bts.belief_update(env, belief, action, obs)
        print(belief)
