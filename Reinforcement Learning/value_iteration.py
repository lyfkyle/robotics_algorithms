#! /usr/bin/env python

import simple_problem
import copy
import math

def value_iteration_finite(states, rewards, actions, trans_prob, T):
    dp = copy.deepcopy(rewards)
    dp1 = copy.deepcopy(rewards)

    for t in range(T):
        for s in states:
            # skip terminal state
            if rewards[s[0]][s[1]] == 1 or rewards[s[0]][s[1]] == -1:
                continue

            # Bellman Equation
            # for a in actions:
            max_val = max(trans_prob[0] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[1] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[2] * dp[s[0]][min(3, s[1] + 1)],  # up
                          trans_prob[0] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[1] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[2] * dp[max(0, s[0] - 1)][s[1]],  # left
                          trans_prob[0] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[1] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[2] * dp[s[0]][max(0, s[1] - 1)],  # down
                          trans_prob[0] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[1] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[2] * dp[min(2, s[0] + 1)][s[1]])  # right
            dp1[s[0]][s[1]] = rewards[s[0]][s[1]] + max_val
            #print(dp1[s[0]][s[1]])

        dp = dp1
        dp1 = copy.deepcopy(dp)

    print(dp1)

def value_iteration_infinite(states, rewards, actions, trans_prob, diff_threshold = 0.001):
    dp = copy.deepcopy(rewards)
    dp1 = copy.deepcopy(rewards)

    cnt = 0
    while True:
        diff = 0
        for s in states:
            # skip terminal state
            if rewards[s[0]][s[1]] == 1 or rewards[s[0]][s[1]] == -1:
                continue

            # Bellman Equation
            # for a in actions:
            max_val = max(trans_prob[0] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[1] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[2] * dp[s[0]][min(3, s[1] + 1)],  # up
                          trans_prob[0] * dp[s[0]][max(0, s[1] - 1)] + trans_prob[1] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[2] * dp[max(0, s[0] - 1)][s[1]],  # left
                          trans_prob[0] * dp[min(2, s[0] + 1)][s[1]] + trans_prob[1] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[2] * dp[s[0]][max(0, s[1] - 1)],  # down
                          trans_prob[0] * dp[s[0]][min(3, s[1] + 1)] + trans_prob[1] * dp[max(0, s[0] - 1)][s[1]] + trans_prob[2] * dp[min(2, s[0] + 1)][s[1]])  # right
            dp1[s[0]][s[1]] = rewards[s[0]][s[1]] + max_val
            diff = max(diff, math.fabs(dp1[s[0]][s[1]] - dp[s[0]][s[1]]))
            #print(dp1[s[0]][s[1]])

        dp = dp1
        dp1 = copy.deepcopy(dp)
        cnt += 1

        if diff < diff_threshold:
            break

    print(cnt)
    print(dp1)

if __name__ == "__main__":
    #value_iteration_finite(simple_problem.states, simple_problem.rewards, simple_problem.actions, simple_problem.trans_prob, 10)
    value_iteration_infinite(simple_problem.states, simple_problem.rewards, simple_problem.actions, simple_problem.trans_prob)
