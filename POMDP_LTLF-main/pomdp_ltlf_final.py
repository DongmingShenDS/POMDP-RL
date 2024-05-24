import math
import numpy as np
import pomdp_py
from pomdp_py import sarsop
import random
import pickle
import matplotlib.pyplot as plt
import sys  # argv.py
import time as tictoc
import copy


class MyState(pomdp_py.State):
    """
    State
    """

    # states will cast into "s" + str(index)
    def __init__(self, state):
        if type(state) != MyState:
            self.state = state
        else:
            self.state = state.state

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if isinstance(other, MyState):
            return self.state == other.state
        return False

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return "MyState(%s)" % str(self.state)


class MyAction(pomdp_py.Action):
    """
    Action
    """

    # actions will cast into "a" + str(index)
    def __init__(self, action):
        if type(action) != MyAction:
            self.action = action
        else:
            self.action = action.action

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, MyAction):
            return self.action == other.action
        return False

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        action_original = self.action[1:]
        return "MyAction(%s)" % str(action_original)


class MyObservation(pomdp_py.Observation):
    """
    Observation
    """

    # observations will cast into "o" + str(index)
    def __init__(self, observation):
        if type(observation) != MyObservation:
            self.observation = observation
        else:
            self.observation = observation.observation

    def __hash__(self):
        return hash(self.observation)

    def __eq__(self, other):
        if isinstance(other, MyObservation):
            return self.observation == other.observation
        return False

    def __str__(self):
        return str(self.observation)

    def __repr__(self):
        return "MyObservation(%s)" % str(self.observation)


class ObservationModel(pomdp_py.ObservationModel):
    """
    ObservationModel: an ObservationModel models the distribution 𝑂(𝑠′,𝑎,𝑜)=Pr(𝑜|𝑠′,𝑎).
    """

    def __init__(self, observations, observation_probability, states):
        """
        Constructor for ObservationModel
        :param  observations: a list representing the observation space
        :param  observation_probability: the observation kernel - a 2D numpy array with states on the rows and
            observations on the columns. (independent of action?)
        :param  states: a list representing the state space
            should include it because need access states in observation_probability correct?
        """
        self.observations = observations
        self.observation_probability = observation_probability
        self.states = states

    def probability(self, observation, next_state, action):
        """
        Returns the probability of Pr(𝑜|𝑠′,𝑎).
        :param  observation (Observation) – the observation 𝑜
        :param  next_state (State) – the next state 𝑠′
        :param  action (Action) – the action 𝑎
        :return the probability Pr(𝑜|𝑠′,𝑎) - float
        """
        state_i = self.states.index(next_state)  # index of state
        observation_i = self.observations.index(observation)  # index of observation
        return self.observation_probability[state_i][observation_i]

    def sample(self, next_state, action):
        """
        Returns observation randomly sampled according to the distribution of this observation model.
        :param  next_state (State) – the next state 𝑠′
        :param  action (Action) – the action 𝑎
        :return the observation 𝑜
        """
        # given a next_state (row), the sum of observations in numpy array should sum to 1
        # example next_state's row: [0.1, 0.3, 0.4, 0.2]
        state_i = self.states.index(next_state)  # index of state
        observation_count = len(self.observations)
        rand_0_1 = random.uniform(0, 1)
        sum_prob_up = 0
        for i in range(observation_count):
            sum_prob_low = sum_prob_up
            sum_prob_up += self.observation_probability[state_i][i]
            if sum_prob_low < rand_0_1 < sum_prob_up:
                # need to return the observation at index i col in the numpy array
                return MyObservation(self.observations[i])

    def get_all_observations(self):
        """
        needed if using a solver that needs to enumerate over the observation space (e.g. value iteration)
        :return all observation 𝑜 in the problem
        """
        return [MyObservation(o) for o in self.observations]


class TransitionModel(pomdp_py.TransitionModel):
    """
    TransitionModel: models the distribution 𝑇(𝑠,𝑎,𝑠′)=Pr(𝑠′|𝑠,𝑎).
    """

    def __init__(self, states, transitions, actions):
        """
        Constructor for TransitionModel
        :param  states: a list representing the state space
        :param  transitions: a dictionary. For a given action a, transitions[a] is a numpy array representing the
            transition probability matrix for action a
        :param  actions: a list representing the action space
        """
        self.states = states
        self.transitions = transitions
        self.actions = actions

    def probability(self, next_state, state, action):
        """
        Returns the probability of Pr(𝑠′|𝑠,𝑎).
        :param  next_state (State) – the next state 𝑠′
        :param  state (State) – the (curr) state 𝑠
        :param  action (Action) – the action 𝑎
        :return the probability Pr(𝑠′|𝑠,𝑎) - float
        """
        # should return self.transitions[action][state_i][next_state_i]
        next_state_i = self.states.index(next_state)
        state_i = self.states.index(state)
        return self.transitions[action][state_i][next_state_i]

    def sample(self, state, action):
        """
        Returns next state randomly sampled according to the distribution of this transition model.
        :param  state (State) – the state 𝑠
        :param  action (Action) – the action 𝑎
        :return the next state 𝑠′
        """
        # should return s' according to the prob matrix at self.transitions[action]
        # given a action (key) and a state (row), the sum of next_states in array (col) should sum to 1
        state_i = self.states.index(state)
        state_count = len(self.states)
        rand_0_1 = random.uniform(0, 1)
        sum_prob_up = 0
        for i in range(state_count):
            sum_prob_low = sum_prob_up
            sum_prob_up += self.transitions[action][state_i][i]
            if sum_prob_low < rand_0_1 < sum_prob_up:
                return MyState(self.states[i])

    def get_all_states(self):
        """
        needed if using a solver that needs to enumerate over the observation space (e.g. value iteration)
        :return all state 𝑠 in the problem
        """
        return [MyState(s) for s in self.states]


class RewardModel(pomdp_py.RewardModel):
    """
    RewardModel: models the distribution Pr(𝑟|𝑠,𝑎,𝑠′) where 𝑟∈ℝ with argmax denoted as denoted as 𝑅(𝑠,𝑎,𝑠′)
    """

    def __init__(self, rewards, states, actions):
        """
        Constructor for RewardModel (what other parameters?)
        :param  rewards: a dictionary. For a given action a, rewards[a] is a numpy array representing the instantaneous
            reward associated with each state under action a.
        :param  states: a list representing the state space
        :param  actions: a list representing the action space
        """
        self.rewards = rewards
        self.states = states
        self.actions = actions

    def probability(self, reward, state, action, next_state):
        """
        Returns the probability of Pr(𝑟|𝑠,𝑎,𝑠′).
        :param  reward (float) – the reward 𝑟
        :param  state (State) – the state 𝑠
        :param  action (Action) – the action 𝑎
        :param  next_state (State) – the next state 𝑠′
        :return the probability Pr(𝑟|𝑠,𝑎,𝑠′) - float
        """
        # deterministic reward & depends only on state and action
        state_i = self.states.index(state)
        if reward == self.rewards[action][state_i]:
            return 1
        else:
            return 0

    def sample(self, state, action, next_state):
        """
        Returns reward randomly sampled according to the distribution of this reward model. This is required.
        :param  state (State) – the next state 𝑠
        :param  action (Action) – the action 𝑎
        :param  next_state (State) – the next state 𝑠′
        :return the reward 𝑟
        """
        # deterministic reward & depends only on state and action
        state_i = self.states.index(state)
        return self.rewards[action][state_i]


class PolicyModel(pomdp_py.RandomRollout):
    """
    PolicyModel: models the distribution PolicyModel models the distribution 𝜋(𝑎|𝑠).
    """

    def __init__(self, states, actions):
        """
        Constructor for PolicyModel
        :param  states: a list representing the state space
        :param  actions: a list representing the action space
        """
        self.states = states
        self.actions = actions

    def probability(self, action, state):
        """
        Returns the probability of 𝜋(𝑎|𝑠).
        :param  action (Action) – the action 𝑎
        :param  state (State) – the state 𝑠
        :return the probability 𝜋(𝑎|𝑠) - float
        """
        # uniformly sample -> equal probability for every actions (independent of state)
        return 1.0 / len(self.actions)

    def sample(self, state, **kwargs):
        """
        Returns action randomly (uniformly) sampled according to the distribution of this policy model.
        :param  state (State) – the next state 𝑠 (here independent of state, just uniformly sample)
        :return the action 𝑎
        """
        # randomly sample an action in actions
        return random.sample(self.actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        """
        Returns a set of all possible actions, if feasible.
        :return all action 𝑎 in the problem
        """
        return [MyAction(a) for a in self.actions]


class POMDP_LTLf(pomdp_py.POMDP):
    """
    POMDP_LTLf: a POMDP_LTLf instance = agent (Agent) + env (Environment).
    """

    def __init__(self,
                 initial_dist,
                 states,
                 actions,
                 observations,
                 observation_probability,
                 transitions,
                 rewards):
        """
        Constructor for POMDP_LTLf
        :param  initial_dist: numpy array of distribution (float) over states, sum=1 (not belief histogram yet)
        :param  states: states array (not of class MyState yet)
        :param  actions: actions array (not of class MyAction yet)
        :param  observations: observations array (not of class MyObservation yet)
        :param  observation_probability: the observation kernel - a 2D numpy array
        :param  transitions: a dictionary. For a given action a, transitions[a] is 2D a numpy array
        :param  rewards: a dictionary. For a given action a, rewards[a] is a 2D numpy array
        """
        # cast initial_dist from numpy array into a Histogram distribution initial_belief
        initial_belief = list_to_hist(initial_dist, states)
        print(initial_belief)
        # produce initial_true_state from initial_dist (randomly generate a state using the initial_dist)
        state_count = len(states)
        rand_0_1 = random.uniform(0, 1)
        sum_prob_up = 0
        for i in range(state_count):
            sum_prob_low = sum_prob_up
            sum_prob_up += initial_dist[i]
            if sum_prob_low < rand_0_1 < sum_prob_up:
                init_true_state = MyState(states[i])
        # cast transitions' key from (type) to MyAction
        transitions_final = {}
        for a in actions:
            transitions_final[a] = transitions[str(a)]
        # cast rewards' key from (type) to MyAction
        rewards_final = {}
        for a in actions:
            rewards_final[a] = rewards[str(a)]
        # create POMDP problem
        agent = pomdp_py.Agent(initial_belief,
                               PolicyModel(states, actions),
                               TransitionModel(states, transitions_final, actions),
                               ObservationModel(observations, observation_probability, states),
                               RewardModel(rewards_final, states, actions))
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(states, transitions_final, actions),
                                   RewardModel(rewards_final, states, actions))
        super().__init__(agent, env, name="POMDP_LTLf")


def hist_to_list(belief_hist, states_class):
    """
    Convert histogram to list
    :param  belief_hist: Histogram of belief (mapped with MyState)
    :param  states_class: MyState class array
    :return Returns list of belief (index are state index)
    """
    return [belief_hist[s] for s in states_class]


def list_to_hist(belief_arr, states_class):
    """
    Convert list to histogram
    :param  belief_arr: list of belief (index are state index)
    :param  states_class: MyState class array
    :return Returns Histogram of belief (mapped with MyState)
    """
    belief_dict = {}
    for i in range(len(belief_arr)):
        belief_dict[states_class[i]] = belief_arr[i]
    return pomdp_py.Histogram(belief_dict)


def get_next_belief(pomdp, cur_belief, action, observation, t):
    """
    Sample next belief from pomdp, cur_belief, action, observation, t
    :param  pomdp: pomdp problem from pickle
    :param  cur_belief: current belief as a list
    :param  action: current action
    :param  observation: current observation
    :param  t: time stamp
    :return Returns list of new belief (index are state index)
    """
    if t:
        state_size = len(pomdp.states[1])
    state_size_next = len(pomdp.states[1])
    next_belief = np.zeros(state_size_next)
    o_index = pomdp.observation_index[(1, observation)]
    for x_next in range(state_size_next):
        if t:
            obs_prob = pomdp.observation_probability[1][x_next][o_index]
            if obs_prob:
                for x in range(state_size):
                    trans_prob = pomdp.transitions[1][action][x][x_next]
                    next_belief[x_next] += cur_belief[x] * trans_prob * obs_prob
            else:
                next_belief[x_next] = 0
        else:
            obs_prob = pomdp.observation_probability[1][x_next][o_index]
            next_belief[x_next] = cur_belief[x_next] * obs_prob
    obs_prob = sum(next_belief)
    if obs_prob:
        return next_belief / obs_prob, obs_prob
    else:
        return np.ones(len(next_belief)) / len(next_belief), obs_prob


def trajectory_out(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, f_out, pomdp_grid_size):
    """
    return single trajectory in true form and indices form (t,s,o,a) , (t,index(s),index(o),index(a))
    only difference between trajectory(): this will print out the trajectory to f_out
    :param  pomdp: pomdp problem from pickle
    :param  policy: policy returned from solver
    :param  horizon: horizon to use in each trajectory
    :param  belief_dict: the dictionary to store belief to improve runtime
    :param  hit_test: TESTING ONLY
    :param  time_test: TESTING ONLY
    :param  belief_precision: decimal precision to use in the belief dict
    :param  f_out: output file
    :param  pomdp_grid_size: TODO
    :return Returns single trajectory in true form and indices form
    """
    traj = []
    traj_ind = []
    t = 1
    s_ind = random.choices(range(len(pomdp.states[t])),
                           weights=pomdp.initial_dist,
                           k=1)[0]
    s = pomdp.states[t][s_ind]  # get new state
    o_ind = random.choices(range(len(pomdp.observations[t])),
                           weights=pomdp.observation_probability[t][s_ind],
                           k=1)[0]
    o = pomdp.observations[t][o_ind]  # get new observation
    belief = get_next_belief(pomdp, pomdp.initial_dist, 0, o, 0)[0]  # get new belief
    a = alpha_vector_policy(policy, belief)  # get new action
    # f_out.writelines("state: {}, action: {}\n".format(s, a))  # not print observation
    f_out.writelines("state: {}, observation:{}, action: {}\n".format(s, o, a))
    traj.append((t, s, o, a))
    traj_ind.append((t, s_ind, o_ind, a))
    s_ind_next = random.choices(range(len(pomdp.states[1])),
                                weights=pomdp.transitions[t][a][s_ind],
                                k=1)[0]
    s_next = pomdp.states[1][s_ind_next]
    o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                weights=pomdp.observation_probability[1][s_ind_next],
                                k=1)[0]
    o_next = pomdp.observations[1][o_ind_next]
    # update belief & belief_dict (if not hit)
    belief_key = (tuple(np.round_(belief, decimals=belief_precision)), a, o_next)
    if belief_key in belief_dict:
        hit_test[0] += 1
        belief = belief_dict[belief_key]
    else:
        temp = get_next_belief(pomdp, belief, a, o_next, 1)[0]
        belief_dict[(tuple(np.round_(belief, decimals=belief_precision)), a, o_next)] = temp
        belief = temp
    s, o = s_next, o_next
    s_ind, o_ind = s_ind_next, o_ind_next
    for t in range(2, horizon):  # iterate until reaching horizon
        a = alpha_vector_policy(policy, belief)
        # f_out.writelines("state: {}, action: {}\n".format(s, a))  # not print observation
        f_out.writelines("state: {}, observation:{}, action: {}\n".format(s, o, a))
        # s[0] gives grid index from (0, 0) to (m-1, n-1). TERMINATE at (pomdp_grid_size[0], pomdp_grid_size[1])
        if s[0][0] == (pomdp_grid_size[0] - 1, pomdp_grid_size[1] + 1):  # s[0][0] or s[0]
            break  # break when TERMINATE reached
        traj.append((t, s, o, a))
        traj_ind.append((t, s_ind, o_ind, a))
        s_ind_next = random.choices(range(len(pomdp.states[1])),
                                    weights=pomdp.transitions[1][a][s_ind],
                                    k=1)[0]
        s_next = pomdp.states[1][s_ind_next]
        o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                    weights=pomdp.observation_probability[1][s_ind_next],
                                    k=1)[0]
        o_next = pomdp.observations[1][o_ind_next]
        # update belief & belief_dict (if not hit)
        belief_key = (tuple(np.round_(belief, decimals=belief_precision)), a, o_next)
        if belief_key in belief_dict:
            hit_test[0] += 1
            belief = belief_dict[belief_key]
        else:
            temp = get_next_belief(pomdp, belief, a, o_next, 1)[0]
            belief_dict[(tuple(np.round_(belief, decimals=belief_precision)), a, o_next)] = temp
            belief = temp
        s, o = s_next, o_next
        s_ind, o_ind = s_ind_next, o_ind_next
    a = alpha_vector_policy(policy, belief)
    # f_out.writelines("state: {}, action: {}\n".format(s, a))  # not print observation
    f_out.writelines("state: {}, observation:{}, action: {}\n".format(s, o, a))
    traj.append((t + 1, s, o, a))
    traj_ind.append((t + 1, s_ind, o_ind, a))
    return traj, traj_ind


def trajectory(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, pomdp_grid_size):
    """
    return single trajectory in true form and indices form (t,s,o,a) , (t,index(s),index(o),index(a))
    :param  pomdp: pomdp problem from pickle
    :param  policy: policy returned from solver
    :param  horizon: horizon to use in each trajectory
    :param  belief_dict: the dictionary to store belief to improve runtime
    :param  hit_test: TESTING ONLY
    :param  time_test: TESTING ONLY
    :param  belief_precision: decimal precision to use in the belief dict
    :param  pomdp_grid_size: TODO
    :return Returns single trajectory in true form and indices form
    """
    traj = []
    traj_ind = []
    t = 1
    s_ind = random.choices(range(len(pomdp.states[t])),
                           weights=pomdp.initial_dist,
                           k=1)[0]
    s = pomdp.states[t][s_ind]  # get new state
    o_ind = random.choices(range(len(pomdp.observations[t])),
                           weights=pomdp.observation_probability[t][s_ind],
                           k=1)[0]
    o = pomdp.observations[t][o_ind]  # get new observation
    belief = get_next_belief(pomdp, pomdp.initial_dist, 0, o, 0)[0]  # get new belief
    a = alpha_vector_policy(policy, belief)  # get new action
    traj.append((t, s, o, a))
    traj_ind.append((t, s_ind, o_ind, a))
    s_ind_next = random.choices(range(len(pomdp.states[1])),
                                weights=pomdp.transitions[t][a][s_ind],
                                k=1)[0]
    s_next = pomdp.states[1][s_ind_next]
    o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                weights=pomdp.observation_probability[1][s_ind_next],
                                k=1)[0]
    o_next = pomdp.observations[1][o_ind_next]
    # update belief & belief_dict (if not hit)
    belief_key = (tuple(np.round_(belief, decimals=belief_precision)), a, o_next)
    if belief_key in belief_dict:
        hit_test[0] += 1
        belief = belief_dict[belief_key]
    else:
        temp = get_next_belief(pomdp, belief, a, o_next, 1)[0]
        belief_dict[(tuple(np.round_(belief, decimals=belief_precision)), a, o_next)] = temp
        belief = temp
    s, o = s_next, o_next
    s_ind, o_ind = s_ind_next, o_ind_next
    for t in range(2, horizon):  # iterate until reaching horizon
        a = alpha_vector_policy(policy, belief)
        if s[0][0] == (pomdp_grid_size[0] - 1, pomdp_grid_size[1] + 1):  # s[0][0] or s[0]
            break  # break when TERMINATE reached
        traj.append((t, s, o, a))
        traj_ind.append((t, s_ind, o_ind, a))
        s_ind_next = random.choices(range(len(pomdp.states[1])),
                                    weights=pomdp.transitions[1][a][s_ind],
                                    k=1)[0]
        s_next = pomdp.states[1][s_ind_next]
        o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                    weights=pomdp.observation_probability[1][s_ind_next],
                                    k=1)[0]
        o_next = pomdp.observations[1][o_ind_next]
        # update belief & belief_dict (if not hit)
        belief_key = (tuple(np.round_(belief, decimals=belief_precision)), a, o_next)
        if belief_key in belief_dict:
            hit_test[0] += 1
            belief = belief_dict[belief_key]
        else:
            temp = get_next_belief(pomdp, belief, a, o_next, 1)[0]
            belief_dict[(tuple(np.round_(belief, decimals=belief_precision)), a, o_next)] = temp
            belief = temp
        s, o = s_next, o_next
        s_ind, o_ind = s_ind_next, o_ind_next
    a = alpha_vector_policy(policy, belief)
    traj.append((t + 1, s, o, a))
    traj_ind.append((t + 1, s_ind, o_ind, a))
    return traj, traj_ind


def evaluate_policy_mc(pomdp, policy, horizon, n_samples, disc, belief_dict, hit_test, time_test, belief_precision,
                       f_out, pomdp_grid_size):
    """
    EVAL(policy): run trajectory for n_samples times and get the avg reward & constraint
    :param  pomdp: pomdp problem from pickle
    :param  policy: policy returned from solver
    :param  horizon: horizon to use in each trajectory
    :param  n_samples: number of trajectories in the simulation
    :param  disc: discount factor
    :param  belief_dict: a dictionary to store belief to improve runtime
    :param  hit_test: TESTING ONLY
    :param  time_test: TESTING ONLY
    :param  belief_precision: decimal precision to use in the belief dict
    :param  f_out: output file
    :param  pomdp_grid_size: TODO
    :return reward: avg reward from simulation EVAL (discounted) - float
    :return constraint: avg constraint from simulation EVAL (discounted) - dict of float, key=constraint_indices
    """
    # NORMAL POMDP (M1 2 3)
    reward = 0
    constraint = {k: 0 for k in pomdp.constraint_indices}
    for iteration in range(n_samples):
        if iteration == 0:
            traj = trajectory_out(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, f_out, pomdp_grid_size)
        else:
            traj = trajectory(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, pomdp_grid_size)
        for t, s, o, a in traj[1]:
            reward += (disc ** (t - 1)) * pomdp.rewards[1][a][s]
            for k in pomdp.constraint_indices:
                if (pomdp.horizon, k) in pomdp.constraints:
                    constraint[k] += (disc ** (t - 1)) * pomdp.constraints[(pomdp.horizon, k)][a][s]
    reward /= (n_samples / (1 - disc))
    for k in pomdp.constraint_indices:
        constraint[k] /= (n_samples / (1 - disc))
    return reward, constraint

    # GOAL POMDP (M4)
    reward = 0
    constraint = {k: 0 for k in pomdp.constraint_indices}
    for iteration in range(n_samples):
        if iteration == 0:
            traj = trajectory_out(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, f_out, pomdp_grid_size)
        else:
            traj = trajectory(pomdp, policy, horizon, belief_dict, hit_test, time_test, belief_precision, pomdp_grid_size)
        for t, s, o, a in traj[1]:
            # reward += (disc ** (t - 1)) * pomdp.rewards[1][a][s]
            reward += pomdp.rewards[1][a][s]
            for k in pomdp.constraint_indices:
                if (pomdp.horizon, k) in pomdp.constraints:
                    # constraint[k] += (disc ** (t - 1)) * pomdp.constraints[(pomdp.horizon, k)][a][s]
                    constraint[k] += pomdp.constraints[(pomdp.horizon, k)][a][s]
    # reward /= (n_samples / (1 - disc))
    reward /= n_samples
    for k in pomdp.constraint_indices:
        # constraint[k] /= (n_samples / (1 - disc))
        constraint[k] /= (n_samples)
    return reward, constraint


def alpha_vector_policy(policy, belief_arr):
    """
    Return an action that is mapped by the agent belief, under this policy
    :param  policy: policy returned from solver
    :param  belief_arr: list of belief (index are state index)
    :return Returns an action (in format str as in the original pomdp read from pickle)
    """
    _, action = max(policy.alphas, key=lambda va: np.dot(belief_arr, va[0]))
    return str(action)[1:]  # argmax of alpha vectors, cast to str


def update_belief_arr(curr_belief_arr, states_class, real_action, real_observation_i, observation_probability,
                      transitions, normalize=True):
    """
    Update belief array, for sanity checking get_next_belief(), TESTING ONLY
    """
    new_belief_arr = np.zeros(len(curr_belief_arr))
    total_prob = 0
    next_state_space = states_class
    for state_i in range(len(next_state_space)):
        observation_prob = observation_probability[state_i][real_observation_i]
        transition_prob = 0
        for state_j in range(len(states_class)):
            transition_prob += transitions[str(real_action)][state_j][state_i] * curr_belief_arr[state_j]
        temp = observation_prob * transition_prob
        new_belief_arr[state_i] = temp
        total_prob += temp
    if normalize and total_prob > 0:
        for i in range(len(new_belief_arr)):
            new_belief_arr[i] /= total_prob
    return new_belief_arr


def expo_gradient(model_path, pomdpsol_path, output_path, display_path, discount, timeout, memory, precision, val_b,
                  total_k, num_it_simulation, time_step, eta, delta, lamb_arr, avg_reward_woc_arr, avg_constraint_arr,
                  belief_dict, belief_precision, global_hit, timer, total_rewards_arr, final_simulation, is_tuning,
                  pomdp_grid_size):
    """
    Run Exponentiated Gradient Method on a Constrained Product POMDP
    :param model_path: path to the model of the Constrained Product POMDP
    :param pomdpsol_path: path to the `pomdpsol` binary for sarsop solver (global path)
    :param output_path: path to output file
    :param display_path: path to graph
    :param discount: discount factor used in model - PARAMETER
    :param timeout: The time limit (seconds) to run the algorithm until termination
    :param memory: The memory size (mb) to run the algorithm until termination
    :param precision: solver runs until regret is less than this absolute `precision` - PARAMETER
    :param val_b: B value in algorithm - PARAMETER
    :param total_k: K value in algorithm - PARAMETER
    :param num_it_simulation: number of simulations to run in each EVAL() step - PARAMETER
    :param time_step: time step in each simulation - PARAMETER
    :param eta: η value in algorithm - PARAMETER
    :param delta: δ value in algorithm - PARAMETER
    :param lamb_arr: the array to store lambda - RETURN
    :param avg_reward_woc_arr: the array to store avg discounted reward from EVAL() - RETURN
    :param avg_constraint_arr: the array to store avg discounted constraint from EVAL() - RETURN
    :param belief_dict: the dictionary to store belief to improve runtime
    :param belief_precision: decimal precision to use in the belief dict - PARAMETER
    :param global_hit: TESTING ONLY
    :param timer: TESTING ONLY
    :param total_rewards_arr: TESTING ONLY
    :param final_simulation: True or False
    :param is_tuning: True (for tuning only) or Flase,
    :param pomdp_grid_size: pomdp grid size in format (m, n)
    """

    """ preparation """
    tic = tictoc.perf_counter()
    # load the class from pickle, open f_out
    constrained_pomdp = pickle.load(open(model_path, 'rb'))
    f_out = open(output_path, "w")

    # horizon
    horizon = constrained_pomdp.horizon

    # initial_dist -> initial_belief
    initial_dist = constrained_pomdp.initial_dist.tolist()

    # states, cast to string
    states_original = constrained_pomdp.states[1]
    states = ["s" + str(i) for i in range(len(states_original))]
    states_dict = {}
    for i in range(len(states)):
        states_dict[states[i]] = states_original[i]
    states_class = [MyState(s) for s in states]

    # action, cast to string, length > 1
    actions_original = constrained_pomdp.actions[1]
    actions = ["a" + str(a) for a in actions_original]
    actions_dict = {}
    for i in range(len(actions)):
        actions_dict[actions[i]] = actions_original[i]
    actions_class = [MyAction(a) for a in actions]

    # observations, cast to string
    observations_original = constrained_pomdp.observations[1]
    observations = ["o" + str(i) for i in range(len(observations_original))]
    observation_dict = {}
    for i in range(len(observations)):
        observation_dict[observations[i]] = observations_original[i]
    observations_class = [MyObservation(o) for o in observations]

    # observation_probability
    observation_probability = constrained_pomdp.observation_probability[1].tolist()

    # transition, map with new actions
    transitions = {}
    transitions_original = constrained_pomdp.transitions
    for a in actions_original:
        transitions["a" + str(a)] = transitions_original[1][a]
    # print(transitions)
    # constraints[(horizon, 1)]
    constraints_original = constrained_pomdp.constraints
    constraints = copy.deepcopy(constraints_original[(horizon, 1)])
    # print(constraints)
    # reward, map with new actions
    rewards_original = constrained_pomdp.rewards
    rewards = copy.deepcopy(rewards_original[horizon])
    # print(rewards)
    # exit(0)
    # constraint_rewards[a] = rewards[10][a] + lambda * constraints[(horizon, 1)][a]
    constraint_rewards = {}
    for a in actions_original:
        rewards["a" + str(a)] = rewards.pop(a)
        constraints["a" + str(a)] = constraints.pop(a)

    """ Exponentiated Gradient Method Body """
    # for tuning hyper-parameters only
    tune = ""
    if is_tuning:
        time_step = 500
        precision = 0.05
        total_k = 1
        tune = input("tuning for reward or constraint (type r or c): ")
        if tune == "r":
            lamb_new = 1
            lamb = 0
        elif tune == "c":
            lamb_new = 0
            lamb = 1
        else:
            print("invalid input, quitting")
            return
    else:
        lamb_new = 1  # TODO 1
        lamb = val_b / 10  # initialize lambda

    # constants initialization
    e = np.exp(1)  # e^1
    if time_step == 0:
        time_step = math.ceil(np.log(0.03) / np.log(discount))
    toc = tictoc.perf_counter()
    timer[0] += toc - tic

    # for k = 1, ..., K
    for k in range(total_k):
        tic = tictoc.perf_counter()
        # get new reward
        lamb_arr[k] = lamb
        for a in actions_original:
            constraint_rewards["a" + str(a)] = (lamb_new * rewards["a" + str(a)] + lamb * constraints["a" + str(a)]) \
                                               / (lamb_new + lamb)

        # create POMDP problem at iteration k
        pomdp_k = POMDP_LTLf(initial_dist,
                             states_class,
                             actions_class,
                             observations_class,
                             observation_probability,
                             transitions,
                             constraint_rewards)

        # solve POMDP problem at iteration k with sarsop, get policy_k
        policy = sarsop(pomdp_k.agent,
                        pomdpsol_path,
                        discount_factor=discount,
                        timeout=timeout,
                        memory=memory,
                        precision=precision,
                        remove_generated_files=False)
        toc = tictoc.perf_counter()
        timer[0] += toc - tic

        # simulation using the policy_k, get \hat{p_k} = avg constraint
        if tune == "r":
            return  # if tuning for reward, should quit here
        tic = tictoc.perf_counter()
        f_out.writelines("\n==========[Running Simulation at k={}, where lambda={}]==========\n".format(k + 1, lamb))
        reward_est, constraint_est = evaluate_policy_mc(constrained_pomdp, policy, time_step, num_it_simulation,
                                                        discount, belief_dict, global_hit, timer, belief_precision,
                                                        f_out, pomdp_grid_size)

        print("end of iteration k =", k + 1)
        # print("dict length", len(belief_dict))
        # print("hit count", global_hit[0])
        # print("hit rate", global_hit[0] / (time_step * num_it_simulation))
        global_hit[0] = 0

        # update lambda
        p_k = constraint_est[constrained_pomdp.constraint_indices[0]]  # get avg constraint from constraint_est dict
        total_reward = reward_est / (1 - discount) + lamb * p_k / (1 - discount)
        # print("total reward:", total_reward)
        total_rewards_arr[k] = total_reward
        power = e ** (eta * (-p_k + 1 - delta))
        # print("pk constraint:", p_k)
        # print("constraint:", (p_k - 1 + delta))
        avg_reward_woc_arr[k] = reward_est
        avg_constraint_arr[k] = p_k
        lamb = val_b * ((lamb * power) / (val_b - lamb + (lamb * power)))
        # print("new lambda:", lamb)
        if tune == "c":
            return  # if tuning for constraint, should quit here
        toc = tictoc.perf_counter()
        timer[1] += toc - tic

    """ Insert Final Results at TOP of the file """
    f_out.close()
    f_out = open(output_path, 'r+')
    old_lines = f_out.readlines()  # read old content
    f_out.seek(0)  # go back to the beginning of the file
    f_out.writelines("\n==========Time information==========\n")
    f_out.writelines("Total Solving & Loading Time: {}\n".format(timer[0]))
    f_out.writelines("Total Simulation Time: {}\n".format(timer[1]))
    f_out.writelines("==========Model information==========\n")
    f_out.writelines("model path: {}\n".format(model_path))
    f_out.writelines("\n==========Display Final Results==========\n")
    f_out.writelines("lambda array: {}\n".format(lamb_arr))
    lamb_running_avg = np.zeros(len(lamb_arr))
    for i in range(len(lamb_arr)):
        if i == 0:
            lamb_running_avg[i] = lamb_arr[i]
        else:
            lamb_running_avg[i] = np.mean(lamb_arr[0:i + 1])
    f_out.writelines("lambda running avg: {}\n".format(lamb_running_avg))
    f_out.writelines("avg reward array: {}\n".format(avg_reward_woc_arr))
    f_out.writelines("avg reward array mean: {}\n".format(avg_reward_woc_arr.mean()))
    f_out.writelines("avg constraint array: {}\n".format(avg_constraint_arr))
    f_out.writelines("avg constraint array mean: {}\n".format(avg_constraint_arr.mean()))
    f_out.writelines("total reward array (for choosing dict precision): {}\n\n\n\n\n".format(total_rewards_arr))
    print("time for finding action 50k*100simu*0.99:", timer[0] * 100 * 50 / (num_it_simulation * total_k))
    print("time for updating belief 50k*100simu*0.99:", timer[1] * 100 * 50 / (num_it_simulation * total_k))
    for line in old_lines:  # write old content after new
        f_out.write(line)
    f_out.close()

    """ Generate Graph """
    k_values = np.linspace(1, total_k, total_k)
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(k_values, lamb_arr, '-', color='blue', linewidth=2, label="lambda")
    lns2 = ax1.plot(k_values, lamb_running_avg, linestyle='dashed', color='red', label="lambda running avg")
    ax1.set_xlabel("Iteration Number k")
    ax1.set_ylabel('lambda value')
    ax2 = ax1.twinx()
    ax2.set_ylabel('reward & constraint')
    lns3 = ax2.plot(k_values, avg_reward_woc_arr, 'o-', color='orange', markevery=5, label="avg reward")
    lns4 = ax2.plot(k_values, avg_constraint_arr, 's-', color='green', markevery=5, label="avg constraint")
    lns = lns1 + lns2 + lns3 + lns4
    labs = [ln.get_label() for ln in lns]
    plt.legend(lns, labs, loc=0)
    plt.savefig(display_path)
    return


def main():
    # model_name = str(sys.argv[1])  # read from command line in server
    # argv2 = str(sys.argv[2])  # read from command line in server
    # if argv2 == "True":
    #     tune_param = True
    # else:
    #     tune_param = False
    tic_main = tictoc.perf_counter()
    # hyper-parameters dict
    model_params = {"case1_1": {"K": 50, "simu": 100, "eta": 2, "delta": 0.25, "discount": 0.99, "precision": 0.05,
                                "B": 5, "txt": "result/case1_1.txt", "png": "result/case1_1.png",
                                "belief_precision": 2, "time": 300, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "case1_2": {"K": 50, "simu": 100, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.05,          ### M1
                                "B": 8, "txt": "result/case1_2.txt", "png": "result/case1_2.png", "belief_precision": 2,
                                "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "case1_3": {"K": 50, "simu": 100, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.05,
                                "B": 8, "txt": "result/case1_3/test_output.txt",
                                "png": "result/case1_3/test_figure.png", "belief_precision": 2, "time": 300,
                                "mem": 1000},
                    "case2": {"K": 50, "simu": 100, "eta": 2, "delta": 0.25, "discount": 0.99, "precision": 0.05,
                              "B": 5, "txt": "result/case2/timed.txt", "png": "result/case2/timed.png",
                              "belief_precision": 2, "time": 300, "mem": 1000},
                    "case3": {"K": 50, "simu": 100, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.05,
                              "B": 6, "txt": "result/case3/timed3.txt", "png": "result/case3/timed3.png",
                              "belief_precision": 2, "time": 300, "mem": 1000},
                    "case4": {"K": 50, "simu": 100, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.05,
                              "B": 6, "txt": "result/case4/timed4.txt", "png": "result/case4/timed4.png",
                              "belief_precision": 2, "time": 300, "mem": 1000},
                    "case5": {"K": 50, "simu": 100, "eta": 2, "delta": 0.2, "discount": 0.99, "precision": 0.05,
                              "B": 10, "txt": "result/case5/timed5.txt", "png": "result/case5/timed5.png",
                              "belief_precision": 2, "time": 300, "mem": 1000},
                    "case6": {"K": 50, "simu": 100, "eta": 2, "delta": 0.2, "discount": 0.99, "precision": 0.05,
                              "B": 25, "txt": "result/case6/timed6.txt", "png": "result/case6/timed6.png",
                              "belief_precision": 2, "time": 300, "mem": 1000},
                    "Fa_multi1": {"K": 100, "simu": 100, "eta": 0.02, "delta": 0.15, "discount": 0.99,                      ### M3
                                  "precision": 1, "B": 20, "txt": "result/Fa_multi1.txt",
                                  "png": "result/Fa_multi1.png",
                                  "belief_precision": 8, "time": 30, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "sequence": {"K": 50, "simu": 50, "eta": 0.2, "delta": 0.25, "discount": 0.99, "precision": 0.05,
                                 "B": 10, "txt": "result/sequence/sequence1.txt",
                                 "png": "result/sequence/sequence1.png",
                                 "belief_precision": 8, "time": 300, "mem": 1000},
                    "case5new": {"K": 100, "simu": 200, "eta": 2, "delta": 0.2, "discount": 0.99, "precision": 0.05,        ### M2
                                 "B": 10, "txt": "result/case5new.txt", "png": "result/case5new.png",
                                 "belief_precision": 2, "time": 300, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "case6new": {"K": 50, "simu": 100, "eta": 2, "delta": 0.2, "discount": 0.99, "precision": 0.05,
                                 "B": 25, "txt": "result/case6new/new_timed6.txt",
                                 "png": "result/case6new/new_timed6.png", "belief_precision": 2, "time": 300,
                                 "mem": 1000},
                    "case_goal_1": {"K": 50, "simu": 10, "eta": 2, "delta": 0.25, "discount": 0.99, "precision": 0.05,
                                    "B": 100, "txt": "result/case_goal_1/1.txt",
                                    "png": "result/case_goal_1/1.png", "belief_precision": 8, "time": 300,
                                    "mem": 1000},
                    "p_sequence_gridgoal": {"K": 100, "simu": 100, "eta": 2, "delta": 0.1, "discount": 0.999,				### M4
                                            "precision": 0.05, "B": 100, "txt": "result/p_sequence_gridgoal.txt",
                                            "png": "result/p_sequence_gridgoal.png", "belief_precision": 8,
                                            "time": 300, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "p_sequence_dfagoal": {"K": 20, "simu": 100, "eta": 2, "delta": 0.1, "discount": 0.999,
                                           "precision": 0.05, "B": 100, "txt": "result/p_sequence_dfagoal/new.txt",
                                           "png": "result/p_sequence_dfagoal/new.png", "belief_precision": 8, "time": 300,
                                           "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "hit_back": {"K": 20, "simu": 10, "eta": 2, "delta": 0.1, "discount": 0.99,
                                 "precision": 0.05, "B": 100, "txt": "result/hit_back/1.txt",
                                 "png": "result/hit_back/1.png", "belief_precision": 8, "time": 300,
                                 "mem": 1000, "pomdp_grid_size": (5, 5)},
                    "toy22": {"K": 100, "simu": 200, "eta": 2, "delta": 0.1, "discount": 0.99,
                              "precision": 0.05, "B": 100, "txt": "result/toy22/1.txt",
                              "png": "result/toy22/1.png", "belief_precision": 8, "time": 300,
                              "mem": 1000, "pomdp_grid_size": (2, 2)},
                    "hit_back_7": {"K": 20, "simu": 50, "eta": 2, "delta": 0.1, "discount": 0.99,
                                   "precision": 0.05, "B": 100, "txt": "result/hit_back_7/1.txt",
                                   "png": "result/hit_back_7/1.png", "belief_precision": 8, "time": 300,
                                   "mem": 1000, "pomdp_grid_size": (5, 5)},
                    "hit_back_8": {"K": 20, "simu": 50, "eta": 2, "delta": 0.1, "discount": 0.99,
                                   "precision": 0.05, "B": 100, "txt": "result/hit_back_8/1.txt",
                                   "png": "result/hit_back_8/1.png", "belief_precision": 8, "time": 30,
                                   "mem": 1000, "pomdp_grid_size": (5, 5)},
                    "grid_goal_seq": {"K": 20, "simu": 100, "eta": 2, "delta": 0.1, "discount": 0.999,
                                            "precision": 0.05, "B": 100, "txt": "result/sequence_gridgoal/1.txt",
                                            "png": "result/sequence_gridgoal/1.png", "belief_precision": 8,
                                            "time": 300, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "ltlf_goal_seq": {"K": 20, "simu": 100, "eta": 2, "delta": 0.1, "discount": 0.999,
                                           "precision": 0.05, "B": 100, "txt": "result/sequence_dfagoal/1.txt",
                                           "png": "result/sequence_dfagoal/1.png", "belief_precision": 8,
                                           "time": 300,
                                           "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_5_5": {"K": 100, "simu": 200, "eta": 2, "delta": 0.5, "discount": 0.99, "precision": 0.5,
                                 "B": 5, "txt": "result/M2_5_5.txt", "png": "result/M2_5_5.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_5_7": {"K": 100, "simu": 200, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.5,
                                 "B": 5, "txt": "result/M2_5_7.txt", "png": "result/M2_5_7.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_5_9": {"K": 100, "simu": 200, "eta": 2, "delta": 0.1, "discount": 0.99, "precision": 0.5,
                                 "B": 5, "txt": "result/M2_5_9.txt", "png": "result/M2_5_9.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_50_5": {"K": 100, "simu": 200, "eta": 2, "delta": 0.5, "discount": 0.99, "precision": 0.5,
                                 "B": 50, "txt": "result/M2_50_5.txt", "png": "result/M2_50_5.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_50_7": {"K": 100, "simu": 200, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.5,
                                 "B": 50, "txt": "result/M2_50_7.txt", "png": "result/M2_50_7.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_50_9": {"K": 100, "simu": 200, "eta": 2, "delta": 0.1, "discount": 0.99, "precision": 0.5,
                                 "B": 50, "txt": "result/M2_50_9.txt", "png": "result/M2_50_9.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_500_5": {"K": 100, "simu": 200, "eta": 2, "delta": 0.5, "discount": 0.99, "precision": 0.5,
                                 "B": 500, "txt": "result/M2_500_5.txt", "png": "result/M2_500_5.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_500_7": {"K": 100, "simu": 200, "eta": 2, "delta": 0.3, "discount": 0.99, "precision": 0.5,
                                 "B": 500, "txt": "result/M2_500_7.txt", "png": "result/M2_500_7.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    "M2_500_9": {"K": 100, "simu": 200, "eta": 2, "delta": 0.1, "discount": 0.99, "precision": 0.5,
                                 "B": 500, "txt": "result/M2_500_9.txt", "png": "result/M2_500_9.png",
                                 "belief_precision": 2, "time": 10, "mem": 1000, "pomdp_grid_size": (4, 4)},
                    } # (B=5,50,100) (1-\delta = 0.5,0.7,0.9)

    # parameters for model & solver
    model_name = "case1_2"  # p_sequence_gridgoal, Fa_multi1, case5new, case1_2
    model_path = "models/" + model_name # or  case5new
    #pomdpsol_path = "/Users/dongmingshen/appl/src/pomdpsol"  # Path to the `pomdpsol` binary
    pomdpsol_path = "/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/appl/src/pomdpsol"  # Path to the `pomdpsol` binary
    print("==========MODEL: {}==========".format(model_path))

    # parameters for the algorithm
    total_k = model_params[model_name]["K"]
    lamb_arr = np.zeros(total_k)  # the array to store lambda
    avg_reward_woc_arr = np.zeros(total_k)  # the array to store avg discounted reward from EVAL()
    avg_constraint_arr = np.zeros(total_k)  # the array to store avg discounted constraint from EVAL()
    belief_dict = {}  # the dictionary to store belief to improve runtime

    # variables for testing only
    global_hit = [0]
    timer = [0, 0]
    total_rewards_arr = np.zeros(total_k)

    # run Exponentiated Gradient Method
    expo_gradient(model_path=model_path,
                  pomdpsol_path=pomdpsol_path,
                  output_path=model_params[model_name]["txt"],
                  display_path=model_params[model_name]["png"],
                  discount=model_params[model_name]["discount"],  # discount factor
                  timeout=model_params[model_name]["time"],  # (seconds) to run the algorithm until termination
                  memory=model_params[model_name]["mem"],  # (mb) to run the algorithm until termination 1gb
                  precision=model_params[model_name]["precision"],  # solver runs until this absolute `precision`
                  val_b=model_params[model_name]["B"],  # B value in algorithm
                  total_k=total_k,  # K value in algorithm (number of iterative updating)
                  num_it_simulation=model_params[model_name]["simu"],  # number of simulations in each EVAL() step
                  time_step=0,  # time step in each simulation = ceiling(ln(0.03)/ln(discount))
                  eta=model_params[model_name]["eta"],  # η value in algorithm (how sensitive)
                  delta=model_params[model_name]["delta"],  # δ value in algorithm
                  lamb_arr=lamb_arr,
                  avg_reward_woc_arr=avg_reward_woc_arr,
                  avg_constraint_arr=avg_constraint_arr,
                  belief_dict=belief_dict,
                  belief_precision=model_params[model_name]["belief_precision"],  # precision to use in the belief dict
                  global_hit=global_hit,
                  timer=timer,
                  total_rewards_arr=total_rewards_arr,
                  final_simulation=True,
                  is_tuning=False,
                  pomdp_grid_size=model_params[model_name]["pomdp_grid_size"])
    toc_main = tictoc.perf_counter()

    # print timer information, need to copy to the top of the file
    print("==========MODEL: {}==========".format(model_path))
    print("Total Simulation: K={}, simulation={}".format(total_k, model_params[model_name]["simu"]))
    print("Total Solving & Loading Time: {}".format(timer[0]))
    print("Total Simulation Time: {}".format(timer[1]))
    print("Total Time: {}".format(toc_main - tic_main))


if __name__ == '__main__':
    main()
