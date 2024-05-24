import numpy as np
from pomdp import constrained_pomdp
from itertools import product
from copy import deepcopy


def combine(A, pomdp, labels, thres):
    n = A.n_qs

    prod_states = {t: list(product(pomdp.states[t], A.Q)) for t in pomdp.states}

    prod_actions = deepcopy(pomdp.actions)

    prod_initial_dist = np.array([pomdp.initial_dist[i // n] if not i % n else 0.0 for i in range(len(prod_states[1]))])

    prod_observations = deepcopy(pomdp.observations)

    prod_observation_probability = {
        t: np.array([pomdp.observation_probability[t][i // n] for i in range(len(prod_states[t]))]) for t in
        pomdp.observation_probability}

    prod_rewards = {
        t: {a: np.array([pomdp.rewards[t][a][i // n] for i in range(len(prod_states[t]))]) for a in pomdp.actions[t]}
        for t in pomdp.rewards}

    prod_constraints = {
        (t, k): {a: np.array([pomdp.constraints[(t, k)][a][i // n] for i in range(len(prod_states[t]))]) for a in
                 pomdp.actions[t]} for t, k in pomdp.constraints}

    prod_constraint_val = deepcopy(pomdp.constraint_val)

    prod_constraint_indices = deepcopy(pomdp.constraint_indices)

    prod_horizon = pomdp.horizon

    prod_transitions = {t: {a: np.zeros((len(prod_states[t]), len(prod_states[t + 1]))) for a in pomdp.actions[t]} for t
                        in pomdp.transitions}

    for t in pomdp.transitions:
        for a in pomdp.actions[t]:
            for i in range(len(prod_states[t])):
                s, q = i // n, str(i % n + 1)
                q_n = A.T[q, tuple(sorted(labels[t][s]))]
                q_ni = int(q_n) - 1
                for j in range(len(pomdp.states[t + 1])):
                    prod_transitions[t][a][i][n * j + q_ni] = pomdp.transitions[t][a][i // n][j]

    if pomdp.constraint_indices:
        k = max(pomdp.constraint_indices) + 1
    else:
        k = 1

    acc_indices = [int(x) - 1 for x in A.acc]
    constraint = {}

    for a in prod_actions[prod_horizon]:
        con = np.zeros(len(prod_states[prod_horizon]))
        for ind in acc_indices:
            start = ind
            for i in range(len(pomdp.states[prod_horizon])):
                con[start + i * n] = 1
        constraint.update({a: con})

    prod_constraints.update({(prod_horizon, k): constraint})

    prod_constraint_val.update({k: thres})

    prod_constraint_indices.append(k)

    product_pomdp = constrained_pomdp(prod_initial_dist, prod_states, prod_actions, prod_transitions, prod_observations,
                                      prod_observation_probability, prod_rewards, prod_constraints, prod_constraint_val,
                                      prod_constraint_indices, prod_horizon)

    return product_pomdp
