import math
from collections import defaultdict
import time
import copy

from numpy.random import choice
import numpy as np

from robotics_algorithm.env.base_env import MDPEnv
from robotics_algorithm.utils.tree import Tree, TreeNode


class MCTS:
    """Monte Carlo Tree Search."""

    def __init__(
        self, env: MDPEnv, max_depth: int = 5, discount_factor: float = 0.99, c: float = 25, max_simulation_eps_len=100
    ):
        """Constructor.

        Args:
            env (MDPEnv): _description_
            max_depth (int, optional): _description_. Defaults to 5.
            discount_factor (float, optional): _description_. Defaults to 0.99.
            c (float, optional): the exploration-exploitation balancing factor. Defaults to 10.
            max_simulation_eps_len (float, optional): max episode length during simulation. Defaults to 100.
        """
        self.env = env
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.c = c
        self.max_simulation_eps_len = max_simulation_eps_len

    def run(self, state, max_planning_time=5.0) -> tuple:
        """Run algorithm.

        For MCTS, this should be called online.

        Args:
            state (state, optional): current state.
            max_planning_time (float, optional): maximum planning time

        Returns:
            best_action (tuple): best action found
        """
        self.total_state_value = defaultdict(float)
        self.state_visit_cnt = defaultdict(int)
        self.total_q_value = defaultdict(lambda: defaultdict(int))
        self.state_action_visit_cnt = defaultdict(lambda: defaultdict(int))

        state = copy.deepcopy(state)
        self.tree = Tree()
        root = self.tree.add_node(state=state)
        start_time = time.time()
        while time.time() - start_time < max_planning_time:
            # Select and expand until a valid child is sampled
            expanded_node = self.select_and_expand(root)
            total_return = self.simulate(expanded_node)
            self.backpropogate(expanded_node, total_return)

        print("Timeout!")
        # Retrieve the best action from estimated q values
        best_action_val = float("-inf")
        best_action = None
        for action in self.env.action_space:
            action_value = self._get_q_value(state, action)
            if action_value > best_action_val:
                best_action_val = action_value
                best_action = action
            print(action, action_value)

        return best_action

    def select_and_expand(self, node: TreeNode) -> TreeNode:
        """Perform both selection and expansion using Upper Confidence Bound (UCB).

        Args:
            node (TreeNode): the current node to expand from

        Returns:
            expanded node (TreeNode): the newly expanded node.
        """
        print("[MCTS]: select...")
        state = node.attr["state"]

        ucb_action_values = []
        for action in self.env.action_space:
            # if there is unexplored action, use it to sample a new state, return it
            if self.state_action_visit_cnt[state][action] == 0:
                result, prob = self._sample_step(state, action)
                new_state, reward, term, trunc, info = result

                new_node = self.tree.add_node(state=new_state, term=term)
                node.add_child(new_node, action=action, reward=reward, prob=prob)
                # self.tree.add_child(node, new_node, action=action, reward=reward, prob=prob)
                return new_node

            # else, calculate UCB value
            ucb_value = self._get_q_value(state, action) + self.c * math.sqrt(
                math.log(self.state_visit_cnt[state]) / self.state_action_visit_cnt[state][action]
            )
            ucb_action_values.append(ucb_value)
            # print(state, action, self._get_q_value(state, action), self.c * math.sqrt(
            #     math.log(self.state_visit_cnt[state]) / self.state_action_visit_cnt[state][action]
            # ))

        # Select action according to UCB
        # print(state, ucb_action_values)
        best_action_idx = np.argmax(np.array(ucb_action_values))
        best_action = self.env.action_space[best_action_idx]

        # Sample next state following best action
        result, prob = self._sample_step(state, best_action)
        new_state, reward, term, trunc, info = result

        # If a new state is encountered, return it
        # new_node = node.get_children(new_state)
        children_states = [c.attr["state"] for c in node.children]
        if new_state not in children_states:
            new_node = self.tree.add_node(state=new_state, term=term)
            self.tree.add_child(node, new_node, action=best_action, reward=reward, prob=prob)
            return new_node
        else:
            new_node = node.children[children_states.index(new_state)]

        # If new_node results in termination, return.
        if term or trunc:
            return new_node

        # else, we sampled an existing state, recurse deeper.
        return self.select_and_expand(new_node)

    def simulate(self, node: TreeNode) -> float:
        """Simulate using random policy

        Args:
            node (TreeNode): the current tree node to expand.

        Returns:
            return (float): total return of the simulated episode.
        """
        state = node.attr["state"]
        print(f"[MCTS]: Simulate from {state}")

        total_return = 0
        if node.attr["term"]:
            return total_return

        for _ in range(self.max_simulation_eps_len):
            random_action = np.random.choice(self.env.action_space)
            result, _ = self._sample_step(state, random_action)
            new_state, reward, term, trunc, info = result

            total_return += reward
            if term or trunc:
                break

            state = new_state

        return total_return

    def backpropogate(self, node: TreeNode, total_return: float) -> None:
        """Backpropogate return from leaf node to the root.

        Args:
            node (TreeNode): The leaf node.
            total_return (float): total return of the simulated episode.
        """
        print(f"[MCTS]: Backpropogate with total_return {total_return}")
        state = node.attr["state"]
        self.state_visit_cnt[state] += 1

        state_value = total_return
        parent_node = node.parent
        while parent_node is not None:
            state = node.attr["state"]
            parent_state = parent_node.attr["state"]
            print(f"[MCTS]: Backpropogate through {parent_state}")

            # # Increase state visit cnt
            self.state_visit_cnt[parent_state] += 1

            # Increase state action visit cnt
            action = parent_node.transition_attr(node)["action"]
            self.state_action_visit_cnt[parent_state][action] += 1

            # Increase state value, considering transition prob and discount factor.
            prob = parent_node.transition_attr(node)["prob"]
            reward = parent_node.transition_attr(node)["reward"]

            # Back up using Bellman optimality equation
            # Q(s, a) = p(s' | s, a) * (R(s, s') + lamda * V(s'))
            # V(s) = max Q(s, a)
            future_return = prob * (reward + self.discount_factor * state_value)
            self.total_q_value[parent_state][action] += future_return
            # print(state, parent_state, future_return, self._get_q_value(parent_state, action))

            state_value = np.array(list(self.total_q_value[parent_state])).max().item()
            node = parent_node
            parent_node = parent_node.parent

    def _sample_step(self, state, action):
        results, probs = self.env.state_transition_func(state, action)
        idx = choice(np.arange(len(results)), 1, p=probs)[
            0
        ]  # choose new state according to the transition probability.
        return results[idx], probs[idx]

    def _get_q_value(self, state, action):
        return self.total_q_value[state][action] / self.state_action_visit_cnt[state][action]
