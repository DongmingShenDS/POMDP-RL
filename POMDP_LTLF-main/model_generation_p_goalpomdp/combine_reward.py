#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pomdp import constrained_pomdp
from itertools import product
from copy import deepcopy


# In[2]:
def xyzdfa_to_index(x, y, z, dfa, M=4, N=4, Z=2, DFA=7):
    """
    :param x:
    :param y:
    :param z: 0 or 1
    :param dfa: start with 1
    :return:
    """
    return (Z * (x * M + y) + z) * DFA + dfa - 1


def index_to_xyzdfa(index, M=4, N=4, Z=2, DFA=7):
    """
    :param index:
    :return:
    """
    dfa = (index + 1) % DFA
    z = ((index + 1) // DFA) % Z
    x = min((((index + 1) // DFA) // Z) // M, N - 1)
    y = ((index + 1 - dfa) // DFA - z) // Z - (x * M)
    return x, y, z, dfa


def combine(A, pomdp, labels, thres, goal, flags):
    N = A.n_qs
    n_flag = len(flags)

    "product states"
    prod_states = {t: list(product(pomdp.states[t], A.Q)) for t in pomdp.states}

    "product states"
    prod_actions = deepcopy(pomdp.actions)

    "product initial dist"
    prod_initial_dist = np.array([pomdp.initial_dist[i // N]
                                  if not i % N else 0.0 for i in range(len(prod_states[1]))])

    "product observations"
    prod_observations = deepcopy(pomdp.observations)

    "product observation probability"
    prod_observation_probability = {
        t: np.array([pomdp.observation_probability[t][i // N]
                     for i in range(len(prod_states[t]))]) for t in pomdp.observation_probability
    }

    "product rewards"
    prod_rewards = {
        t: {a: np.array([pomdp.rewards[t][a][i // N]
                         for i in range(len(prod_states[t]))]) for a in pomdp.actions[t]} for t in pomdp.rewards
    }

    "product constraint"
    prod_constraints = {
        (t, k): {a: np.array([pomdp.constraints[(t, k)][a][i // N]
                              for i in range(len(prod_states[t]))]) for a in pomdp.actions[t]
                 } for t, k in pomdp.constraints
    }
    prod_constraint_val = deepcopy(pomdp.constraint_val)
    prod_constraint_indices = deepcopy(pomdp.constraint_indices)

    "product horizon"
    prod_horizon = pomdp.horizon

    "product transition"
    prod_transitions = {
        t: {a: np.zeros((len(prod_states[t]), len(prod_states[t + 1])))
            for a in pomdp.actions[t]} for t in pomdp.transitions
    }
    for t in pomdp.transitions:
        for a in pomdp.actions[t]:
            for i in range(len(prod_states[t])):
                s, q = i // N, str(i % N + 1)
                q_n = A.T[q, labels[t][s]]
                q_ni = int(q_n) - 1
                for j in range(len(pomdp.states[t + 1])):
                    prod_transitions[t][a][i][N * j + q_ni] = pomdp.transitions[t][a][i // N][j]  # same add DFA GOAL
    if goal == "DFA":
        print("DFA as goal")
        for t in pomdp.transitions:
            for a in pomdp.actions[t]:
                for i in range(len(prod_states[t])):
                    zi = (i // N) % n_flag  # zi = i's corresponding flag
                    s, q = i // N, str(i % N + 1)  # s = grid state index, q = automaton index (from 1)
                    q_n = A.T[q, labels[t][s]]
                    q_ni = int(q_n) - 1
                    for j in range(len(pomdp.states[t + 1])):
                        zj = j % n_flag  # zj = j's corresponding flag
                        # 1st: if the next DFA state is accepted (satisfy specification)
                        # 2nd: next state corresponds to GOAL's grid
                        # 3rd: current state not correspond to GOAL's grid
                        # 4th: current state not correspond to TERMINATE's grid
                        # 5th: i's corresponding flag = j's corresponding flag (in same layer)
                        if q_n in A.acc \
                                and (j // n_flag) == (len(pomdp.states[t + 1]) - 2 * n_flag) // n_flag \
                                and ((i // N) // n_flag) != (len(pomdp.states[t + 1]) - 2 * n_flag) // n_flag \
                                and ((i // N) // n_flag) != (len(pomdp.states[t + 1]) - 1 * n_flag) // n_flag \
                                and zi == zj:
                            print("from: ", index_to_xyzdfa(i), "to: ", index_to_xyzdfa(N * j + q_ni + zi))
                            prod_transitions[t][a][i] = np.zeros(len(prod_states[t + 1]))
                            prod_transitions[t][a][i][N * j + q_ni] = 1

    "update constraint"
    if pomdp.constraint_indices:
        k = max(pomdp.constraint_indices) + 1
    else:
        k = 1  # same
    acc_indices = [int(x) - 1 for x in A.acc]
    constraint = {}
    for a in prod_actions[prod_horizon]:
        con = np.zeros(len(prod_states[prod_horizon]))
        for ind in acc_indices:
            for i in range(len(pomdp.states[prod_horizon])):
                # the only state with constraint_reward=1 is when in GOAL and DFA satisfied
                if i == len(pomdp.states[prod_horizon]) - 2 * n_flag:
                    for z in flags:
                        print(z, "index: ", index_to_xyzdfa(ind + i * N + z * N))
                        con[ind + i * N + z * N] = 1  # same change GOAL constraint reward
        constraint.update({a: con})
    prod_constraints.update({(prod_horizon, k): constraint})
    prod_constraint_val.update({k: thres})
    prod_constraint_indices.append(k)

    product_pomdp = constrained_pomdp(prod_initial_dist,
                                      prod_states,
                                      prod_actions,
                                      prod_transitions,
                                      prod_observations,
                                      prod_observation_probability,
                                      prod_rewards,
                                      prod_constraints,
                                      prod_constraint_val,
                                      prod_constraint_indices,
                                      prod_horizon)
    return product_pomdp

    temp = deepcopy(prod_transitions)
    if goal == "DFA":
        print("DFA as goal")
        for t in transitions:
            for a in actions[t]:
                for i in range(len(prod_states[t])):
                    zi = i % n_flag  # zi = i's corresponding flag
                    s, q = i // N, str(i % N + 1)  # s = grid state index, q = automaton index (from 1)
                    q_n = A.T[q, labels[t][s]]
                    q_ni = int(q_n) - 1
                    for j in range(len(states[t + 1])):
                        zj = j % n_flag  # zj = j's corresponding flag
                        # 1st: if the next DFA state is accepted (satisfy specification)
                        # 2nd: next state corresponds to GOAL's grid
                        # 3rd: current state not correspond to GOAL's grid
                        # 4th: current state not correspond to TERMINATE's grid
                        # 5th: i's corresponding flag = j's corresponding flag (in same layer)
                        if q_n in A.acc \
                                and (j // n_flag) == (len(states[t + 1]) - 2 * n_flag) // n_flag \
                                and ((i // N) // n_flag) != (len(states[t + 1]) - 2 * n_flag) // n_flag \
                                and ((i // N) // n_flag) != (len(states[t + 1]) - 1 * n_flag) // n_flag \
                                and zi == zj:
                            temp[t][a][i] = np.zeros(len(prod_states[t + 1]))
                            temp[t][a][i][N * j + q_ni] = 1
