import numpy as np
from copy import deepcopy
from pomdp import constrained_pomdp
from itertools import product
import itertools
import random


def probsum(iterable):
    sum = 0
    for i, j in enumerate(iterable):
        sum = sum + j
    return sum


def man_dist(a, b):  # ?
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def transition_list(a, s, m, n, p):
    P = [[], []]
    if a != 'U' and s[0] + 1 < m:
        P[0].append((s[0] + 1, s[1]))
        P[1].append(p if a == 'D' else (1 - p) / 2.0)
    if a != 'D' and s[0] - 1 >= 0:
        P[0].append((s[0] - 1, s[1]))
        P[1].append(p if a == 'U' else (1 - p) / 2.0)
    if a != 'R' and s[1] - 1 >= 0:
        P[0].append((s[0], s[1] - 1))
        P[1].append(p if a == 'L' else (1 - p) / 2.0)
    if a != 'L' and s[1] + 1 < n:
        P[0].append((s[0], s[1] + 1))
        P[1].append(p if a == 'R' else (1 - p) / 2.0)
    # normal states 'stay'
    prob_sum = probsum(P[1])
    if prob_sum < 1:
        P[0].append((s))
        P[1].append(1 - prob_sum)
    return P


def gridworld(m, n, horizon, p, r, n_agent, agent_pri, start, private, rew, rewards, switch, p_swi):
    """gridworld
    :param m: grid m
    :param n: grid n
    :param horizon: pomdp horizon
    :param p: transition stochasticity
    :param r: observation stochasticity
    :param n_agent: number of agent
    :param agent_pri: agent private information
    :param start: agent starting locations
    :param private: "goal" if private information = goals
    :param rew: "goal" if reward is at goals
    :param rewards: corresponds to agent_pri
    :param switch: True if possibly switch dst after reaching
    :param p_swi: probability of randomly choose next destination
    :return:
    """
    "states"
    # states[t][i] = ((loc_1, loc_2), dst_1, dst_2)
    locations = [(i, j) for i in range(m) for j in range(n)]
    agent_locations = [(l1, l2) for l1 in locations for l2 in locations]
    states = [(al, x1t, x2t) for al in agent_locations for x1t in agent_pri[0] for x2t in agent_pri[1]]
    states_horizon = {t: deepcopy(states) for t in range(1, horizon + 1)}

    "actions"
    # moves[0] for agent1 action space; moves[1] for agent2 action space
    moves = [['U', 'D', 'L', 'R'], ['U', 'D', 'L', 'R']]
    # prescription for each agent is a list of possible mapping from its private state to moves
    prescription = [[] for _ in range(n_agent)]
    # must use tuple(sorted(dict.items())) to make dict hashable, get back dict by dict(prescription[i][j])
    for i in range(n_agent):
        prescription[i] = [tuple(sorted(dict(zip(agent_pri[i], j)).items()))
                           for j in itertools.product(moves[i], repeat=len(agent_pri[i]))]
    actions = [(p1, p2) for p1 in prescription[0] for p2 in prescription[1]]
    actions_horizon = {t: deepcopy(actions) for t in range(1, horizon + 1)}

    "transition prob"
    P = {}
    for state in states:
        for action in actions:
            P[(state, action)] = [[], []]
    if not switch:  # when goal reached, stay in goal
        for al, x1t, x2t in states:
            state = al, x1t, x2t
            ag1_s, ag2_s = al[0], al[1]  # current locations
            ag1_pri, ag2_pri = x1t, x2t  # private information
            for action in actions:
                ag1_per = dict(action[0])  # get back dict
                ag2_per = dict(action[1])  # get back dict
                ag1_a, ag2_a = ag1_per[ag1_pri], ag2_per[ag2_pri]  # current actions
                # deal with agent1 and agent2 transition P separately
                ag1_p = transition_list(ag1_a, ag1_s, m, n, p)
                ag2_p = transition_list(ag2_a, ag2_s, m, n, p)
                ns = [((ag1_ns, ag2_ns), x1t, x2t) for ag1_ns in ag1_p[0] for ag2_ns in ag2_p[0]]  # next state
                p_ns = [(p1 * p2) for p1 in ag1_p[1] for p2 in ag2_p[1]]  # combined prob of next state
                P[(state, action)] = [ns, p_ns]  # P[(s, a)][0]=next_states, P[(s, a)][1]=probabilities
    else:  # when goal reached, with p switch to a random goal
        for al, x1t, x2t in states:
            state = al, x1t, x2t
            ag1_s, ag2_s = al[0], al[1]  # current locations
            ag1_pri, ag2_pri = x1t, x2t  # private information
            for action in actions:
                ag1_per = dict(action[0])  # get back dict
                ag2_per = dict(action[1])  # get back dict
                ag1_a, ag2_a = ag1_per[ag1_pri], ag2_per[ag2_pri]  # current actions
                # deal with agent1 and agent2 transition P separately
                ag1_p = transition_list(ag1_a, ag1_s, m, n, p)
                ag2_p = transition_list(ag2_a, ag2_s, m, n, p)
                # if ag1 is in ag1_pri (ag1_s==x1t), pick randomly the next ag1dst (given it has >1 ag1goals)
                if ag1_s == x1t and len(agent_pri[0]) > 1 and random.uniform(0, 1) <= p_swi:
                    ag1_dsts = agent_pri[0]
                    p1_dsts = [1.0 / len(agent_pri[0])] * len(agent_pri[0])
                else:
                    ag1_dsts = [x1t]
                    p1_dsts = [1.0]
                # if ag2 is in ag2_pri (ag2_s==x2t), pick randomly the next ag2dst (given it has >1 ag2goals)
                if ag2_s == x2t and len(agent_pri[1]) > 1 and random.uniform(0, 1) <= p_swi:
                    ag2_dsts = agent_pri[1]
                    p2_dsts = [1.0 / len(agent_pri[1])] * len(agent_pri[1])
                else:
                    ag2_dsts = [x2t]
                    p2_dsts = [1.0]
                # combine next state
                ns = [((ag1_ns, ag2_ns), ag1_dst, ag2_dst)
                      for ag1_ns in ag1_p[0] for ag2_ns in ag2_p[0]
                      for ag1_dst in ag1_dsts for ag2_dst in ag2_dsts]
                # combine prob of next state
                p_ns = [(p1 * p2 * p1_dst * p2_dst)
                        for p1 in ag1_p[1] for p2 in ag2_p[1]
                        for p1_dst in p1_dsts for p2_dst in p2_dsts]
                P[(state, action)] = [ns, p_ns]  # P[(s, a)][0]=next_states, P[(s, a)][1]=probabilities

    "transitions Pr(𝑠′|𝑠,𝑎)"
    # transitions[action][state_i][next_state_i] gives the transition probability
    transitions = {a: np.zeros((len(states), len(states)), dtype=float) for a in actions}
    for a in actions:
        for s_i, s in enumerate(states):
            for i, ns in enumerate(P[(s, a)][0]):
                ns_i = states.index(ns)
                transitions[a][s_i][ns_i] = P[(s, a)][1][i]
    transitions_horizon = {t: deepcopy(transitions) for t in range(1, horizon)}

    "initial_dist"
    initial_dist = np.zeros(len(states), dtype=float)
    if private == "goal":
        count = len(agent_pri[0]) * len(agent_pri[1])  # distributed uniformly based on private information
        for i in range(len(states)):
            z = states[i][0]
            if z == (start[0], start[1]):  # i = index where two agents at corresponding starts
                initial_dist[i] = 1 / count

    "constraints"
    constraints = {}
    constraint_val = {}
    constraint_indices = []

    "rewards value"
    R = {}  # function of (S_t, Per_t)
    for action in actions:
        for state in states:
            R[(action, state)] = 0.0
    if rew == "goal" and rewards is None:
        for action in actions:
            for al, x1t, x2t in states:
                state = al, x1t, x2t
                ag1_s, ag2_s = al[0], al[1]
                ag1_pri, ag2_pri = x1t, x2t
                if ag1_s == ag1_pri:
                    R[(action, state)] += 1.0
                if ag2_s == ag2_pri:
                    R[(action, state)] += 1.0
    elif rew == "goal" and rewards is not None:
        for action in actions:
            for al, x1t, x2t in states:
                state = al, x1t, x2t
                ag1_s, ag2_s = al[0], al[1]
                ag1_pri, ag2_pri = x1t, x2t  # in some destination
                if ag1_s == ag1_pri:  # agent1
                    R[(action, state)] += rewards[0][agent_pri[0].index(ag1_pri)]
                if ag2_s == ag2_pri:  # agent2
                    R[(action, state)] += rewards[1][agent_pri[1].index(ag2_pri)]

    "rewards Pr(𝑟|𝑠,𝑎,𝑠′)"
    # rewards[action][state_i] gives the reward
    rewards = {a: np.zeros(len(states), dtype=float) for a in actions}
    for a in actions:
        for s_i, s in enumerate(states):
            rewards[a][s_i] = R[(a, s)]
    rewards_horizon = {t: deepcopy(rewards) for t in range(1, horizon + 1)}

    "observation space"
    # observations: Y_t = (Z_t) in format (loc1, loc2) => location only (for deterministic transition)
    observations = [ag_l for ag_l in agent_locations]
    observations_horizon = {t: deepcopy(observations) for t in range(1, horizon + 1)}

    "observation probability Pr(𝑜′|𝑠′,𝑎)"
    # observation_probability[state_i][observation_i] gives the prob of observing o under s (deterministic transition)
    observation_probability = np.zeros((len(states), len(observations)), dtype=float)
    for s_i, s in enumerate(states):
        obs, x1t, x2t = s # current observation = al = current agent locations
        for o_i, o in enumerate(observations):
            if o == obs:
                observation_probability[s_i][o_i] = 1.0
    observation_probability_horizon = {t: deepcopy(observation_probability) for t in range(1, horizon + 1)}

    return constrained_pomdp(initial_dist,
                             states_horizon,
                             actions_horizon,
                             transitions_horizon,
                             observations_horizon,
                             observation_probability_horizon,
                             rewards_horizon,
                             constraints,
                             constraint_val,
                             constraint_indices,
                             horizon)
