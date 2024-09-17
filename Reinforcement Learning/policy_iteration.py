#!/usr/bin/evn python

import simple_problem
import copy
import math

def policy_evaluation(states, rewards, policy, trans_prob, diff_threshold = 0.01):
    dp = copy.deepcopy(rewards)
    dp1 = copy.deepcopy(rewards)

    cnt = 0
    while True:
        diff = 0
        for s in states:
            # skip terminal state
            if rewards[s[0]][s[1]] == 1 or rewards[s[0]][s[1]] == -1:
                continue

            # Value of state s according to policy
            if policy[s[0]][s[1]] == "up":
                val = trans_prob[0] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[1] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[2] * dp[s[0]][min(3, s[1] + 1)]  # up
            elif policy[s[0]][s[1]] == "left":
                val = trans_prob[0] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[1] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[2] * dp[max(0, s[0] - 1)][s[1]]  # left
            elif policy[s[0]][s[1]] == "down":
                val = trans_prob[0] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[1] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[2] * dp[s[0]][max(0, s[1] - 1)]  # down
            else:
                val = trans_prob[0] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[1] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[2] * dp[min(2, s[0] + 1)][s[1]]  # right
            dp1[s[0]][s[1]] = rewards[s[0]][s[1]] + val
            diff = max(diff, math.fabs(dp1[s[0]][s[1]] - dp[s[0]][s[1]]))
            #print(dp1[s[0]][s[1]])

        dp = dp1
        dp1 = copy.deepcopy(dp)
        cnt += 1

        if diff < diff_threshold:
            break

    return dp1

def policy_iteration_infinite(states, rewards, actions, trans_prob):
    policy = [["up" for x in range(4)] for y in range(3)]  # initial policy, all up
    #print(policy)

    while True:
        unchanged = True
        policy_val = policy_evaluation(states, rewards, policy, trans_prob)

        for s in states:
            # skip terminal state
            if rewards[s[0]][s[1]] == 1 or rewards[s[0]][s[1]] == -1:
                continue

            # Policy improvement by one-step lookahead
            max_val = float('-inf')
            up_val = trans_prob[0] * policy_val[max(0, s[0] - 1)][s[1]] + trans_prob[1] * policy_val[s[0]][max(0, s[1] - 1)] + trans_prob[2] * policy_val[s[0]][min(3, s[1] + 1)]  # up
            if up_val > max_val:
                max_val = up_val
                max_a = "up"
            left_val = trans_prob[0] * policy_val[s[0]][max(0, s[1] - 1)] + trans_prob[1] * policy_val[min(2, s[0] + 1)][s[1]] + trans_prob[2] * policy_val[max(0, s[0] - 1)][s[1]]  # left
            if left_val > max_val:
                max_val = left_val
                max_a = "left"
            down_val = trans_prob[0] * policy_val[min(2, s[0] + 1)][s[1]] + trans_prob[1] * policy_val[s[0]][min(3, s[1] + 1)] + trans_prob[2] * policy_val[s[0]][max(0, s[1] - 1)]  # down
            if down_val > max_val:
                max_val = down_val
                max_a = "down"
            right_val = trans_prob[0] * policy_val[s[0]][min(3, s[1] + 1)] + trans_prob[1] * policy_val[max(0, s[0] - 1)][s[1]] + trans_prob[2] * policy_val[min(2, s[0] + 1)][s[1]]  # right
            if right_val > max_val:
                max_val = right_val
                max_a = "right"

            if policy[s[0]][s[1]] != max_a:
                policy[s[0]][s[1]] = max_a
                unchanged = False

        if unchanged:
            break

    print(policy)

if __name__ == "__main__":
    policy_iteration_infinite(simple_problem.states, simple_problem.rewards, simple_problem.actions, simple_problem.trans_prob)
