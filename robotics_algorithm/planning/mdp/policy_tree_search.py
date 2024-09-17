from typing import Callable

import numpy as np

from robotics_algorithm.env.base_env import MDPEnv


class PolicyTreeSearch:
    def __init__(self):
        pass

    def run(self, env: MDPEnv, max_depth: int = 5, discount_factor: float = 0.99) -> Callable:
        """Run algorithm.

        Args:
            env (MDPEnv): the env.
            max_depth (int, optional): maximum serach depth. Defaults to 5.
            discount_factor (float, optional): the discount factor for future reward. Defaults to 0.99.

        Returns:
            Callable: returns the policy. To be called online.
        """
        self.max_depth = max_depth
        self.discount_factor = discount_factor

        def policy_fn(state):
            q, _ = self._compute_state_value(env, state, 0)
            action_probs = np.zeros(env.action_space_size)
            action_probs[q.argmax()] = 1.0
            return action_probs

        return policy_fn

    def _compute_state_value(self, env, state, cur_depth):
        q = np.zeros(env.action_space_size)

        # If maximum depth is not reached, recurse deeper
        if cur_depth <= self.max_depth:
            for action_idx, action in enumerate(env.action_space):
                results, probs = env.state_transition_func(state, action)

                # calculate Q values
                q_sa = 0
                for i, result in enumerate(results):
                    next_state, reward, term, trunc, info = result

                    if not term and not trunc:
                        _, next_state_value = self._compute_state_value(env, next_state, cur_depth + 1)
                    else:
                        next_state_value = 0

                    q_sa += probs[i] * (reward + self.discount_factor * next_state_value)

                # update Q(s,a)
                q[action_idx] = q_sa

        # else we do nothing, hence q and state_value will be zero.
        else:
            pass

        state_value = q.max()
        return q, state_value
