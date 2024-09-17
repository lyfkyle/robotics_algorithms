from typing import Any

import numpy as np

from robotics_algorithm.env.base_env import MDPEnv, SpaceType, EnvType


class PolicyTreeSearch:
    def __init__(self, env: MDPEnv, max_depth: int = 5, discount_factor: float = 0.99):
        """Constructor.

        Args:
            env (MDPEnv): the env.
            max_depth (int, optional): maximum serach depth. Defaults to 5.
            discount_factor (float, optional): the discount factor for future reward. Defaults to 0.99.
        """
        assert env.state_space.type == SpaceType.DISCRETE.value
        assert env.action_space.type == SpaceType.DISCRETE.value
        assert env.observability == EnvType.FULLY_OBSERVABLE.value

        self.env = env
        self.max_depth = max_depth
        self.discount_factor = discount_factor

    def run(self, state: Any) -> Any:
        """Compute best action for the current state.

        Args:
            state (Any): the current state

        Returns:
            action (Any): the best action
        """
        q, _ = self._compute_state_value(self.env, state, 0)
        print(q)
        actions = self.env.action_space.get_all()
        return actions[q.argmax()]

    def _compute_state_value(self, env, state, cur_depth):
        q = np.zeros(env.action_space.size)

        # If maximum depth is not reached, recurse deeper
        if cur_depth <= self.max_depth:
            actions = env.action_space.get_all()
            for action_idx, action in enumerate(actions):
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
