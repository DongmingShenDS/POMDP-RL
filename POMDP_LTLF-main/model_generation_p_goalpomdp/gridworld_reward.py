#!/usr/bin/env python
# coding: utf-8
# current

# In[1]:
import numpy as np
from copy import deepcopy
from pomdp import constrained_pomdp
from itertools import product


# In[2]:
def probsum(iterable):
    sum = 0
    for i, j in enumerate(iterable):
        sum = sum + j
    return sum


# In[3]:
def man_dist(a, b):  # ?
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# In[4]:
def gridworld(m, n, flags, horizon, p, r, rew, f_loc, goal):
    """gridworld

    :param m:
    :param n:
    :param flags:
    :param horizon:
    :param p:
    :param r:
    :param rew:
    :param f_loc:
    :param goal:
    :return: pomdp object
    """
    "remove lines below in tessting"
    # m = 2
    # n = 2
    # flags = [0, 1]
    # horizon = 2
    # p = 1  # Transition uncertainity
    # f_loc = (1, 1)
    # r = 1  # Grid location uncertainity
    "get vars"
    n_flag = len(flags)
    f_dict = {flags[i]: i for i in range(n_flag)}
    "states: change (i, j) to ((i, j), o) where o in p_obstacle_locations"  # checked
    states = [((i, j), f) for i in range(m) for j in range(n) for f in flags]
    # ADDING 2*n_flag extra states
    for i in range(n_flag):
        states.append(((m - 1, n), flags[i]))  # GOALs (to claim rewards)
    for i in range(n_flag):
        states.append(((m - 1, n + 1), flags[i]))  # TERMINATEs (to stay in forever after GOAL)
    states = {t: states for t in range(1, horizon + 1)}

    "actions: no change"  # checked
    actions = {t: ['U', 'D', 'L', 'R'] for t in range(1, horizon + 1)}

    "P (one transition prob): same as before, only change {s} to {s, o} as the actual state"
    P = {}  # transition prob at t
    for s, o in states[1]:
        for a in actions[1]:
            P.update({(s, o, a): [[], []]})
    for s, o in states[1]:
        # GOAL: p=1 go to TERMINATE (with corresponding o)
        if s == (m - 1, n):
            for a in actions[1]:
                P[(s, o, a)][0] = [((m - 1, n + 1), o)]
                P[(s, o, a)][1] = [1.0]
            continue
        # TERMINATE: p=1 stay in TERMINATE (with corresponding o)
        if s == (m - 1, n + 1):
            for a in actions[1]:
                P[(s, o, a)][0] = [((m - 1, n + 1), o)]
                P[(s, o, a)][1] = [1.0]
            continue
        # if given a goal_location in the grid, direct transfer from goal to GOAL (with corresponding o)
        if goal != 'DFA' and s == goal:
            print("goal update")
            for a in actions[1]:
                P[(s, o, a)][0] = [((m - 1, n), o)]
                P[(s, o, a)][1] = [1.0]
            continue
        # NORMAL STATES with normal actions
        for a in actions[1]:
            if a != 'U' and s[0] + 1 < m:
                P[(s, o, a)][0].append(((s[0] + 1, s[1]), o))
                P[(s, o, a)][1].append(p if a == 'D' else (1 - p) / 2.0)
            if a != 'D' and s[0] - 1 >= 0:
                P[(s, o, a)][0].append(((s[0] - 1, s[1]), o))
                P[(s, o, a)][1].append(p if a == 'U' else (1 - p) / 2.0)
            if a != 'R' and s[1] - 1 >= 0:
                P[(s, o, a)][0].append(((s[0], s[1] - 1), o))
                P[(s, o, a)][1].append(p if a == 'L' else (1 - p) / 2.0)
            if a != 'L' and s[1] + 1 < n:
                P[(s, o, a)][0].append(((s[0], s[1] + 1), o))
                P[(s, o, a)][1].append(p if a == 'R' else (1 - p) / 2.0)
            # normal states 'stay'
            prob_sum = probsum(P[(s, o, a)][1])
            if prob_sum < 1:
                P[(s, o, a)][0].append((s, o))
                P[(s, o, a)][1].append(1 - prob_sum)

    "transitions: same as before, only change {s} to {s, o} as the actual state"  # checked
    trans_step = dict()
    for a in actions[1]:
        trans_step.update({a: np.zeros(((m * n + 2) * n_flag, (m * n + 2) * n_flag))})
        for s, o in states[1]:
            x, y = s[0], s[1]
            for i, (ns, no) in enumerate(P[(s, o, a)][0]):  # nx, ny, no => next state index
                nx, ny = ns[0], ns[1]
                # trans_step[action][state][next state] = prob of this transition under action
                trans_step[a][(n * x + y) * n_flag + f_dict[o]][(n * nx + ny) * n_flag + f_dict[no]] = \
                    P[(s, o, a)][1][i]
    transitions = {t: deepcopy(trans_step) for t in range(1, horizon)}

    "initial_dist: first gird distribution uniformly spread n_o (p_obstacle_locations), sum=1"  # checked
    initial_dist = np.zeros((m * n + 2) * n_flag)
    for i in range(n_flag):
        initial_dist[i] = 1 / n_flag

    "observation: same as states"
    observations = deepcopy(states)

    "constraints: no change"
    constraints = {}
    constraint_val = {}
    constraint_indices = []

    "rewards: reward at a location should spread its n_o in this setting"
    rew[(m - 1) * n + n] = 0  # GOAL
    rew[(m - 1) * n + (n + 1)] = 0  # TERMINATE
    rewards = {t: {a: np.zeros((m * n + 2) * n_flag, dtype=float) for a in actions[t]} for t in states}
    for t in states:
        for a in actions[t]:
            for i in range(m * n):  # m*n or m*n+2? although the last 2 doesn't matter...
                for o in range(n_flag):
                    rewards[t][a][i * n_flag + o] = rew[i]

    "observation probability"
    # In addition to the usual index observation (x,y), we have an extra component flag (f).
    # f = 0 if (x,y) != f_loc. f = Z if (x,y) == f_loc.
    obs_prob = np.zeros(((m * n + 2) * n_flag, (m * n + 2) * n_flag))
    for s, o in states[1]:
        # check surrounding, if movable
        x, y = s[0], s[1]
        count = 0
        sur = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (0 <= x + dx < m) and (0 <= y + dy < n) and (dx != 0 or dy != 0):  # exclude current state
                    sur.add((x + dx, y + dy))
                    count += 1
        # observe current state (with prob r)
        if (x, y) == f_loc:
            # if in flag grid, can observe the actual Z (f_dict[o]), otherwise just observe 0
            obs_prob[(n * x + y) * n_flag + f_dict[o]][(n * x + y) * n_flag + f_dict[o]] = r
        else:
            obs_prob[(n * x + y) * n_flag + f_dict[o]][(n * x + y) * n_flag + 0] = r
        # observe surrounding states (with prob sum = 1-r, uniform)
        for nx, ny in sur:
            # if in flag grid, can observe the actual Z (f_dict[o]), otherwise just observe 0
            if (x, y) == f_loc:
                obs_prob[(n * x + y) * n_flag + f_dict[o]][(n * nx + ny) * n_flag + f_dict[o]] = (1 - r) / count
            else:
                obs_prob[(n * x + y) * n_flag + f_dict[o]][(n * nx + ny) * n_flag + 0] = (1 - r) / count
        # GOAL only observes GOAL, TERMINATE only observes TERMINATE
        if (x == m-1 and y == n) or (x == m-1 and y == n+1):
            obs_prob[(n * x + y) * n_flag + f_dict[o]] = np.zeros((m * n + 2) * n_flag)
            obs_prob[(n * x + y) * n_flag + f_dict[o]][(n * x + y) * n_flag + f_dict[o]] = 1.0
    observation_probability = {t: deepcopy(obs_prob) for t in states}

    return constrained_pomdp(initial_dist,
                             states,
                             actions,
                             transitions,
                             observations,
                             observation_probability,
                             rewards,
                             constraints,
                             constraint_val,
                             constraint_indices,
                             horizon)
