#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pomdp import constrained_pomdp
from itertools import product
from copy import deepcopy


# In[2]:


def combine(A, pomdp, labels, thres):
    """ update the constrained_pomdp with reward from LTLf (DFA in A)
    :param A: DFA automation object
    :param pomdp: the original constrained_pomdp object
    :param labels: horizon*(m*n) array specify the labels in the grid in time t
    :param thres:
    :return: the updated constrained_pomdp object
    """
    N = A.n_qs  # the number of automaton states, (i // N)=original grid state index, (i % N)=automation state index

    """ load and expand pomdp states
    states is a dictionary. 
    For a given time t, states[t] is a list representing the state space at time t.
    """
    # |pomdp.states| = horizon*(m*n) of (2) where 2 corresponds to (x,y)
    # |prod_states| = horizon*(N*m*n) of (3) where 3 corresponds to ((x,y),auto)
    #   => basically expand & copy for N times for each [t]
    # expand states by product with A.Q (automaton states), result dimension = |states| * N
    prod_states = {t: list(product(pomdp.states[t], A.Q)) for t in pomdp.states}  # expand states by A.Q (n_qs list)

    """ load pomdp actions
    actions is a dictionary. 
    ...
    """
    prod_actions = deepcopy(pomdp.actions)  # actions are the same

    """ load and expand pomdp initial_dist
    initial_dist
    """
    # |pomdp.initial_dist| = m*n (# of grids from pomdp)
    # |prod_initial_dist| = |pomdp.initial_dist|*N
    # pomdp.initial_dist[i // N] = 1 only when i = 0,1,2 (corresponds to the first grid state)
    #   => generally where grid_state = 1
    # i % N only when i = 0,N,2N,... (corresponds to the first automaton state)
    prod_initial_dist = np.array([pomdp.initial_dist[i // N]
                                  if not i % N else 0.0 for i in range(len(prod_states[1]))])

    """ load pomdp observations
    observations is a dictionary. 
    ...
    """
    prod_observations = deepcopy(pomdp.observations)  # observations are the same

    """ load and expand pomdp observation_probability
    observation_probability is a dictionary. 
    For a given time t, observation_probability[t] represents the observation kernel at time t. 
    The observation kernel is a two-dimensional numpy array with states on the rows and observations on the columns.
    """
    # |pomdp.observation_probability| = horizon*|dict|
    # |pomdp.observation_probability[t]| = |dict| = (m*n)states * (m*n)observations
    # pomdp.observation_probability[t][i // N] => (corresponds to grid_state) (every N automaton_states = 1 grid_state)
    #   => basically expand & copy for N times for each [t]
    # |prod_observation_probability[t]| = |pomdp.observation_probability[t]|*N = horizon*(N*m*n)*(m*n)
    prod_observation_probability = {
        t: np.array([pomdp.observation_probability[t][i // N]
                     for i in range(len(prod_states[t]))]) for t in pomdp.observation_probability
    }

    """ load and expand pomdp rewards
    rewards is a dictionary. 
    For a given time t, rewards[t] is a dictionary. 
    For a given action a, rewards[t][a] is a numpy array representing the instantaneous reward associated with each 
        state under action a.
    """
    # |pomdp.rewards| = horizon*|actions|*(m*n)
    # |pomdp.rewards[t][a]| = (m*n)
    # pomdp.rewards[t][a][i // N] => (corresponds to grid_state) (every N automaton_states = 1 grid_state)
    #   => basically expand & copy for N times for each [t][a]
    # |prod_rewards| = horizon*|actions|*(N*m*n)
    prod_rewards = {
        t: {a: np.array([pomdp.rewards[t][a][i // N]
                         for i in range(len(prod_states[t]))]) for a in pomdp.actions[t]} for t in pomdp.rewards
    }

    """ load and expand pomdp constraints
    constraints is a dictionary. (similar to rewards)
    For a given time and constraint k, constraints[(t,k)] is a dictionary. 
    For a given action a, constraints[(t,k)][a] is a numpy array representing the instantaneous constraint-reward 
        associated with each state under action a.
    """
    # |pomdp.constraints| = ?
    # |pomdp.constraints[(t, k)][a]| = ?
    # pomdp.constraints[(t, k)][a][i // N => (corresponds to grid_state) (every N automaton_states = 1 grid_state)
    # |prod_constraints| = ?
    prod_constraints = {
        (t, k): {a: np.array([pomdp.constraints[(t, k)][a][i // N]
                              for i in range(len(prod_states[t]))]) for a in pomdp.actions[t]
                 } for t, k in pomdp.constraints
    }
    prod_constraint_val = deepcopy(pomdp.constraint_val)  # prod_constraint_val not change
    prod_constraint_indices = deepcopy(pomdp.constraint_indices)  # constraint_indices not change

    """ load pomdp horizon
    horizon
    """
    prod_horizon = pomdp.horizon

    """
    transitions is a dictionary. 
    For a given time t, transitions[t] is a dictionary. 
    For a given action a, transitions[t][a] is a numpy array representing the transition prob matrix for action a.
    """
    # |pomdp.transitions| = (horizon-1)*|actions|*(m*n)*(m*n)
    prod_transitions = {t: {a: np.zeros((len(prod_states[t]), len(prod_states[t + 1])))
                            for a in pomdp.actions[t]} for t in pomdp.transitions
                        }
    # |prod_transitions| = (horizon-1)*|actions|*(N*m*n)*(N*m*n)
    # |prod_transitions[t]| = |dict| = |actions|*(N*m*n)*(N*m*n)
    # |prod_transitions[t][a]| = |np_array| = (N*m*n)*(N*m*n)
    for t in pomdp.transitions:
        for a in pomdp.actions[t]:
            for i in range(len(prod_states[t])):  # len(prod_states[t]) = N*m*n
                s, q = i // N, str(i % N + 1)  # s = grid state index, q = automaton index (from 1)
                # |label[t]| = (m*n) grid correspond to which location has which label
                # q_n => get the destination states corresponds to {(orig_state, ap): dest_state, ...} in A.T
                q_n = A.T[q, labels[t][s]]  # ? DFA transition
                q_ni = int(q_n) - 1
                for j in range(len(pomdp.states[t + 1])):
                    prod_transitions[t][a][i][N * j + q_ni] = pomdp.transitions[t][a][i // N][j]

    # append the LTLf constraints after the original pomdp constraints
    if pomdp.constraint_indices:
        k = max(pomdp.constraint_indices) + 1
    else:
        k = 1
    # list of index of the accepting states for the LTLf = int(accepting states) - 1
    acc_indices = [int(x) - 1 for x in A.acc]  # A.acc => list of accepting states
    """ constraint EQ8? """
    constraint = {}
    for a in prod_actions[prod_horizon]:  # the actions in the last horizon
        con = np.zeros(len(prod_states[prod_horizon]))  # (N*m*n)
        for ind in acc_indices:
            start = ind
            for i in range(len(pomdp.states[prod_horizon])):  # (m*n)
                con[start + i * N] = 1
        # |constraint| = |actions|*(N*m*n)
        constraint.update({a: con})  # update as {action: constraint} dict

    prod_constraints.update({(prod_horizon, k): constraint})  # update into prod_constraints at k
    prod_constraint_val.update({k: thres})  # update threshold at k
    prod_constraint_indices.append(k)  # update k

    # construct the final constrained_pomdp
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
                                      prod_horizon
                                      )
    return product_pomdp
