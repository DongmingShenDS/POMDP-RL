#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from copy import deepcopy
from pomdp import constrained_pomdp


# In[2]:


def probsum(iterable):
    """ summation of items in iterable, return the sum (why not just use sum()?) """
    sum = 0
    for i, j in enumerate(iterable):
        sum = sum + j
    return sum


# In[3]:


def gridworld(m, n, horizon, p, rew):
    """ generate the complete gridworld with reward, construct the constrained_pomdp
    :param m: grid len1 m
    :param n: grid len2 n
    :param horizon: pomdp horizon / number of steps to look at
    :param p: non-stochastic move probability => given some action, (1-p) random move ???
    :param rew: reward: m*n np_array, specify which location has which reward (using 1D index when given)
    :return: a constrained_pomdp object
    """
    shape = (m, n)  # shape of gridworld (m x n)
    """ t*m*n
    states is a dictionary.
    For a given time t, states[t] is a list representing the state space at time t.
    state space s => gridworld index as tuple (x, y) => total = m*n.
    
    Let s = states[t][i], then => s[0]: x-axis index, s[1]: y-axis index
        s[0] - 1 => Up      U
        s[0] + 1 => Down    D
        s[1] - 1 => Left    L
        s[1] + 1 => Right   R
    """
    states = {t: ([(i, j) for i in range(shape[0]) for j in range(shape[1])]) for t in range(1, horizon + 1)}
    """ t*|actions|
    actions is a dictionary. 
    For a given time t, actions[t] is a list representing the action space at time t => ['U', 'D', 'L', 'R'].
    """
    actions = {t: ['U', 'D', 'L', 'R'] for t in range(1, horizon + 1)}

    """ m*n*|actions|
    transitions probability ? 
    format => (state, action): [[next state], [prob of this transition]]
    """
    P = {}  # init transition probability matrix
    for s in states[1]:  # s[0]: x-axis index, s[1]: y-axis index
        for a in actions[1]:
            P.update({(s, a): [[], []]})  # format => (state, action): [[?], [?]]

    for s in states[1]:  # m*n
        for a in actions[1]:  # 4
            """ IMPORTANT NOTE
            given action A, can never move to opposite direction ~A or move out the grid
            can only move to A with probability p, or other non-A with probability 1-p in total 
            """
            if a != 'U' and s[0] + 1 < m:  # if action != Up and can still go Down
                P[(s, a)][0].append((s[0] + 1, s[1]))  # next state = Down 1
                P[(s, a)][1].append(p if a == 'D' else (1 - p) / 2.0)  # prob=p if action=Down, =(1-p)/2 if not

            if a != 'D' and s[0] - 1 >= 0:  # if action != Down and can still go Up
                P[(s, a)][0].append((s[0] - 1, s[1]))  # next state = Up 1
                P[(s, a)][1].append(p if a == 'U' else (1 - p) / 2.0)  # prob=p if action=Up, =(1-p)/2 if not

            if a != 'R' and s[1] - 1 >= 0:  # if action != Right and can still go Left
                P[(s, a)][0].append((s[0], s[1] - 1))  # next state = Right 1
                P[(s, a)][1].append(p if a == 'L' else (1 - p) / 2.0)  # prob=p if action=Right, =(1-p)/2 if not

            if a != 'L' and s[1] + 1 < n:  # if action != Left and can still go Right
                P[(s, a)][0].append((s[0], s[1] + 1))  # next state = Left 1
                P[(s, a)][1].append(p if a == 'R' else (1 - p) / 2.0)  # prob=p if action=Left, =(1-p)/2 if not

            prob_sum = probsum(P[(s, a)][1])  # left-bottom corner, action=Up => Up=p, Left=(1-p)/2, Stand=(1-p)/2 ???

            if prob_sum < 1:
                P[(s, a)][0].append(s)  # next state = Stand
                P[(s, a)][1].append(1 - prob_sum)  # prob = what's left

    """ (m*n)*(m*n)*|actions|*t
    transitions is a dictionary. 
    For a given time t, transitions[t] is a dictionary => trans_step
    For a given action a, transitions[t][a] is a numpy array representing the transition prob matrix for action a.
    """
    trans_step = dict()
    for a in actions[1]:  # action: ['U', 'D', 'L', 'R']
        trans_step.update({a: np.zeros((m * n, m * n))})  # nparray => the transition prob matrix for action a
        for s in states[1]:  # m*n (grid axis x, y)
            x, y = s[0], s[1]  # s[0]: x-axis index, s[1]: y-axis index
            for i, ns in enumerate(P[(s, a)][0]):  # ns => next state corresponds to state(s) & action(a)
                nx, ny = ns[0], ns[1]  # next_x, next_y
                """trans_step[action][state][next state] = prob of this transition under action"""
                trans_step[a][n * x + y][n * nx + ny] = P[(s, a)][1][i]  # encoding (x,y) into 1D index
    transitions = {t: deepcopy(trans_step) for t in range(1, horizon)}  # copy trans_step t times => transition matrix

    """
    initial_dist
    """
    initial_dist = np.zeros(m * n)
    initial_dist[0] = 1.0

    """
    observations is a dictionary. 
    For a given time t, observations[t] is a list representing the observation space at time t.
    """
    observations = deepcopy(states)  # observations = states => t*m*n dictionary

    """
    constraints is a dictionary. 
    For a given time and constraint k, constraints[(t,k)] is a dictionary. 
    For a given action a, rewards[(t,k)][a] is a numpy array representing the instantaneous constraint-reward 
        associated with each state under action a.
    """
    constraints = {}
    constraint_val = {}
    constraint_indices = []

    """
    rewards is a dictionary. 
    For a given time t, rewards[t] is a dictionary. 
    For a given action a, rewards[t][a] is a numpy array representing the instantaneous reward associated with each 
        state under action a.
    """
    rewards = {t: {a: np.zeros(m * n, dtype=float) for a in actions[t]} for t in states}
    for t in states:  # rewards[t] is a dictionary
        for a in actions[t]:  # rewards[t][a] is a numpy array representing the instantaneous reward at s, a
            for i in range(m * n):
                rewards[t][a][i] = rew[i]  # reward from rew

    """
    observation_probability is a dictionary. 
    For a given time t, observation_probability[t] represents the observation kernel at time t. 
    The observation kernel is a two-dimensional numpy array with states on the rows and observations on the columns.
    """
    obs_prob = np.zeros((m * n, m * n))  # 2D kernel, (m*n)rows = states, (m*n)cols = observations
    for s in states[1]:
        x, y = s[0], s[1]
        count = 0
        sur = set()  # empty set
        for dx in [-1, 0, 1]:  # U, Stand, D
            for dy in [-1, 0, 1]:  # R, Stand, L
                if (0 <= x + dx < m) and (0 <= y + dy < n):  # if any U, D, L, R of current (x, y) still in the gird
                    sur.add((x + dx, y + dy))  # add obs to the set
                    count += 1
        for nx, ny in sur:
            """trans_step[state][next state] = prob of this observation"""
            obs_prob[n * x + y][n * nx + ny] = 1 / count  # encoding (x,y) into 1D index
    observation_probability = {t: deepcopy(obs_prob) for t in states}  # copy obs_prob t times => obs probability

    # construct the constrained_pomdp
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
                             horizon
                             )







