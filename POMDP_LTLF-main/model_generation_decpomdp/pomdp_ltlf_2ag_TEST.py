import math
import numpy as np
import pomdp_py
from pomdp_py import sarsop
import random
import pickle
import matplotlib.pyplot as plt
import sys  # argv.py
from copy import deepcopy
from itertools import compress, count
import time as tictoc
import copy
from gridworld_reward_middle import gridworld_middle
from combine import combine
from automation import DFA


class MyState(pomdp_py.State):
    """State
    """

    # states will cast into "s" + str(index)
    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    """Action
    """

    # actions will cast into "a" + str(index)
    def __init__(self, action, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    """Observation
    """

    # observations will cast into "o" + str(index)
    def __init__(self, observation, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class ObservationModel(pomdp_py.ObservationModel):  # TODO
    """ObservationModel: an ObservationModel models the distribution 𝑂(𝑠′,𝑎,𝑜)=Pr(𝑜|𝑠,'𝑎).
    """

    def __init__(self, observations, observation_probability, states):
        """Constructor for ObservationModel
        :param observations: observation space (list)
        :param observation_probability: the observation kernel (a 2D numpy array). observations[s_i][o_i]=Pr(𝑜|𝑠,𝑎)
        :param states: state space (list)
        """
        self.observations = observations
        self.observation_probability = observation_probability
        self.states = states

    def probability(self, observation, state, last_action):
        """Returns the probability of Pr(𝑜|𝑠,'𝑎).
        :param observation: the observation o
        :param state: the state s
        :param last_action: last action 'a – ignore
        :return the probability Pr(𝑜|𝑠,'𝑎)
        """
        state_i = int(str(state)[1:])  # state_i = self.states.index(state)
        observation_i = int(str(observation)[1:])  # observation_i = self.observations.index(observation)
        return self.observation_probability[state_i][observation_i]

    def sample(self, state, last_action):
        """Samples observation randomly according to the distribution of observation model.
        :param state: the state s
        :param last_action: last action 'a – ignore
        :return the observation 𝑜
        """
        # given a next_state (row), the sum of observations in numpy array should sum to 1  TODO
        state_i = self.states.index(state)  # index of state
        o_i = np.random.choice(np.arange(0, len(self.observations)), p=self.observation_probability[state_i])
        return MyObservation(self.observations[o_i])

    def get_all_observations(self):
        """Needed if using a solver that needs to enumerate over the observation space (e.g. value iteration)
        :return all observation 𝑜 in the problem
        """
        return [MyObservation(o) for o in self.observations]


class TransitionModel(pomdp_py.TransitionModel):  # TODO
    """TransitionModel: models the distribution 𝑇(𝑠,𝑎,𝑠′)=Pr(𝑠′|𝑠,𝑎).
    """

    def __init__(self, states, transitions, actions):
        """Constructor for TransitionModel
        :param states: state space (list)
        :param transitions: a dictionary with key a. transitions[a][s_i][ns_i]=Pr(𝑠′|𝑠,𝑎)
        :param actions: action space (list)
        """
        self.states = states
        self.transitions = transitions
        self.actions = actions

    def probability(self, next_state, state, action):
        """Returns the probability of Pr(𝑠′|𝑠,𝑎).
        :param  next_state: the next state 𝑠′
        :param  state: the (current) state 𝑠
        :param  action: the action 𝑎
        :return the probability Pr(𝑠′|𝑠,𝑎)
        """
        # should return self.transitions[action][state_i][next_state_i]
        next_state_i = int(str(next_state)[1:])  # next_state_i = self.states.index(next_state)
        state_i = int(str(state)[1:])  # state_i = self.states.index(state)
        return self.transitions[action][state_i][next_state_i]

    def sample(self, state, action):
        """Samples next state randomly according to the distribution of this transition model.
        :param  state: the (current) state 𝑠
        :param  action: the action 𝑎
        :return the next state 𝑠′
        """
        # should return s' according to the prob matrix at self.transitions[action]
        # given a action (key) and a state (row), the sum of next_states in array (col) should sum to 1
        state_i = self.states.index(state)  # index of state
        ns_i = np.random.choice(np.arange(0, len(self.states)), p=self.transitions[action][state_i])
        return MyState(self.states[ns_i])

    def get_all_states(self):
        """
        needed if using a solver that needs to enumerate over the observation space (e.g. value iteration)
        :return all state 𝑠 in the problem
        """
        return [MyState(s) for s in self.states]


class RewardModel(pomdp_py.RewardModel):
    """RewardModel: models the distribution Pr(𝑟|𝑠,𝑎,𝑠′) where 𝑟∈ℝ with argmax denoted as denoted as 𝑅(𝑠,𝑎,𝑠′)
    """

    def __init__(self, rewards, states, actions):
        """Constructor for RewardModel
        :param rewards: a dictionary with key a. rewards[a][s_i]=R (deterministic)
        :param states: state space (list)
        :param actions: action space (list)
        """
        self.rewards = rewards
        self.states = states
        self.actions = actions

    def probability(self, reward, state, action, next_state):
        """Returns the probability of Pr(𝑟|𝑠,𝑎,𝑠′).
        :param reward: the reward 𝑟
        :param state: the state 𝑠
        :param action: the action 𝑎
        :param next_state: the next state 𝑠′
        :return the probability Pr(𝑟|𝑠,𝑎,𝑠′)
        """
        # deterministic reward & depends only on state and action
        state_i = self.states.index(state)
        if reward == self.rewards[action][state_i]:
            return 1
        else:
            return 0

    def sample(self, state, action, next_state):
        """Samples reward randomly according to the distribution of this reward model. This is required.
        :param state: the next state 𝑠
        :param action: the action 𝑎
        :param next_state: the next state 𝑠′
        :return the reward 𝑟
        """
        # deterministic reward & depends only on state and action
        state_i = int(str(state)[1:])  # state_i = self.states.index(state)
        return self.rewards[action][state_i]


class PolicyModel(pomdp_py.RandomRollout):
    """PolicyModel: models the distribution PolicyModel models the distribution 𝜋(𝑎|𝑠).
    """

    def __init__(self, states, actions, *args, **kwargs):
        """Constructor for PolicyModel
        :param states: state space (list)
        :param actions: action space (list)
        """
        super().__init__(*args, **kwargs)
        self.states = states
        self.actions = actions

    def probability(self, action, state):
        """Returns the probability of 𝜋(𝑎|𝑠).
        :param action: the action 𝑎
        :param state: the state 𝑠
        :return the probability 𝜋(𝑎|𝑠)
        """
        # uniformly sample -> equal probability for every actions (independent of state)
        return 1.0 / len(self.actions)

    def sample(self, state, **kwargs):
        """Samples action randomly (uniformly) according to the distribution of this policy model.
        :param state: the state 𝑠 (here independent of state, just uniformly sample)
        :return the action 𝑎
        """
        # randomly sample an action in actions
        return random.sample(self.actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        """Returns a set of all possible actions, if feasible.
        :return all action 𝑎 in the problem
        """
        return [MyAction(a) for a in self.actions]


class POMDP_LTLf(pomdp_py.POMDP):
    """POMDP_LTLf: a POMDP_LTLf instance = agent (Agent) + env (Environment).
    """

    def __init__(self,
                 initial_dist,
                 states,
                 actions,
                 observations,
                 observation_probability,
                 transitions,
                 rewards):
        """Constructor for POMDP_LTLf
        :param initial_dist: numpy array of distribution (float) over s, sum=1 (not of belief histogram yet)
        :param states: states list (not MyState yet)
        :param actions: actions list (not MyAction yet)
        :param observations: observations list (not MyObservation yet)
        :param observation_probability: observations[s_i][o_i]=Pr(𝑜|𝑠)
        :param transitions: transitions[a][s_i][ns_i]=Pr(𝑠′|𝑠,𝑎)
        :param rewards: rewards[a][s_i]=𝑅(𝑠,𝑎)
        """

        "initial_belief"
        # cast initial_dist from numpy array into a Histogram distribution initial_belief
        initial_belief = list_to_hist(initial_dist, states)

        "init_true_state"
        # produce initial_true_state from initial_dist (randomly generate a state using the initial_dist)
        s_i = np.random.choice(np.arange(0, len(states)), p=initial_dist)
        init_true_state = MyState(states[s_i])

        "transitions_final"
        # cast transitions' key from (type) to MyAction
        transitions_final = {}
        for a in actions:
            transitions_final[a] = transitions[str(a)]

        "rewards_final"
        # cast rewards' key from (type) to MyAction
        rewards_final = {}
        for a in actions:
            rewards_final[a] = rewards[str(a)]

        "create POMDP problem = agent + env"
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
    """Convert histogram to list
    """
    return [belief_hist[s] for s in states_class]


def list_to_hist(belief_arr, states_class):
    """Convert list to histogram
    """
    belief_dict = {}
    for i in range(len(belief_arr)):
        belief_dict[states_class[i]] = belief_arr[i]
    return pomdp_py.Histogram(belief_dict)


def pomdpltlf_prepare(constrained_pomdp):
    """convert constraint pomdp to
    initial_dist, states, actions, observations, observation_probability, transitions, rewards, constraints
    """

    "constrained_pomdp: pickle from model_path"
    constrained_pomdp = constrained_pomdp

    "horizon"
    horizon = constrained_pomdp.horizon

    "initial_dist -> initial_belief"
    initial_dist = constrained_pomdp.initial_dist.tolist()

    "states, cast to string"
    states_original = constrained_pomdp.states[1]
    states = ["s" + str(i) for i in range(len(states_original))]
    states_dict = {}
    for i in range(len(states)):
        states_dict[states[i]] = states_original[i]
    states_class = [MyState(s) for s in states]
    print("# states :", len(states))

    "action, cast to string, length > 1"
    actions_original = constrained_pomdp.actions[1]
    actions = ["a" + str(i) for i in range(len(actions_original))]  # TODO
    actions_dict = {}
    for i in range(len(actions)):
        actions_dict[actions[i]] = actions_original[i]
    actions_class = [MyAction(a) for a in actions]
    print("# actions :", len(actions))

    "observations, cast to string"
    observations_original = constrained_pomdp.observations[1]
    observations = ["o" + str(i) for i in range(len(observations_original))]
    observation_dict = {}
    for i in range(len(observations)):
        observation_dict[observations[i]] = observations_original[i]
    observations_class = [MyObservation(o) for o in observations]
    print("# observations :", len(observations))

    "observation_probability"
    observation_probability = constrained_pomdp.observation_probability[1].tolist()

    "transitions"
    transitions = {}
    transitions_original = constrained_pomdp.transitions
    for ia, a in enumerate(actions_original):
        transitions[("a" + str(ia))] = transitions_original[1][a]

    "constraints[(horizon, 1)]"
    constraints_original = constrained_pomdp.constraints
    constraints = copy.deepcopy(constraints_original[(horizon, 1)])

    "reward"
    rewards_original = constrained_pomdp.rewards
    rewards = copy.deepcopy(rewards_original[horizon])

    "rewards & constraints"
    for ia, a in enumerate(actions_original):
        rewards["a" + str(ia)] = rewards.pop(a)
        constraints["a" + str(ia)] = constraints.pop(a)

    "create POMDP problem at iteration k"
    return initial_dist, states_class, actions_class, actions_original, observations_class, \
           observation_probability, transitions, rewards, constraints


def get_next_belief(pomdp, cur_belief, action, observation, t):
    """
    Sample next belief from pomdp, cur_belief, action, observation, t
    :param  pomdp: pomdp problem from pickle
    :param  cur_belief: current belief (list)
    :param  action: current action
    :param  observation: current observation
    :param  t: time stamp
    :return Returns new belief where index=s_i (list)
    """
    state_size = 0
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


def alpha_vector_policy(pomdp, policy, belief_arr):
    """Return an action that is mapped by the agent belief, under this policy
    """
    _, action = max(policy.alphas, key=lambda va: np.dot(belief_arr, va[0]))
    return pomdp.actions[1][int(str(action)[1:])]  # argmax of alpha vectors, cast to str


def trajectory_out(pomdp, policy, horizon, f_out=None):
    """Return single trajectory in true form and indices form (t,s,o,a), (t,index(s),index(o),index(a))
    """
    traj = []
    traj_ind = []
    t = 1
    s_ind = random.choices(range(len(pomdp.states[t])),
                           weights=pomdp.initial_dist, k=1)[0]
    s = pomdp.states[t][s_ind]  # get new state
    o_ind = random.choices(range(len(pomdp.observations[t])),
                           weights=pomdp.observation_probability[t][s_ind], k=1)[0]
    o = pomdp.observations[t][o_ind]  # get new observation
    belief = get_next_belief(pomdp, pomdp.initial_dist, 0, o, 0)[0]  # get new belief
    # DS: extra, print belief
    non_z_belief = list(compress(count(), belief))
    non_z_belief_sb = [str(pomdp.states[1][i]) + ": " + str(belief[i]) for i in non_z_belief]
    if f_out is not None:
        f_out.write('B: {')
        for sb in non_z_belief_sb:
            f_out.write(sb + ', ')
        f_out.write('}\n')
    a = alpha_vector_policy(pomdp, policy, belief)  # get new action
    r = pomdp.rewards[1][a][s_ind]
    if f_out is not None:
        f_out.writelines("S:{}, O:{}, A:{}, R:{}\n".format(s, o, a, r))
    traj.append((t, s, o, a))
    traj_ind.append((t, s_ind, o_ind, a))
    s_ind_next = random.choices(range(len(pomdp.states[1])),
                                weights=pomdp.transitions[t][a][s_ind], k=1)[0]
    s_next = pomdp.states[1][s_ind_next]
    o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                weights=pomdp.observation_probability[1][s_ind_next], k=1)[0]
    o_next = pomdp.observations[1][o_ind_next]
    belief = get_next_belief(pomdp, belief, a, o_next, 1)[0]
    # DS: extra, print belief
    non_z_belief = list(compress(count(), belief))
    non_z_belief_sb = [str(pomdp.states[1][i]) + ": " + str(belief[i]) for i in non_z_belief]
    if f_out is not None:
        f_out.write('B: {')
        for sb in non_z_belief_sb:
            f_out.write(sb + ', ')
        f_out.write('}\n')
    s, o = s_next, o_next
    s_ind, o_ind = s_ind_next, o_ind_next
    for t in range(2, horizon):  # iterate until reaching horizon
        a = alpha_vector_policy(pomdp, policy, belief)
        r = pomdp.rewards[1][a][s_ind]
        if f_out is not None:
            f_out.writelines("S:{}, O:{}, A:{}, R:{}\n".format(s, o, a, r))
        traj.append((t, s, o, a))
        traj_ind.append((t, s_ind, o_ind, a))
        s_ind_next = random.choices(range(len(pomdp.states[1])),
                                    weights=pomdp.transitions[1][a][s_ind], k=1)[0]
        s_next = pomdp.states[1][s_ind_next]
        o_ind_next = random.choices(range(len(pomdp.observations[1])),
                                    weights=pomdp.observation_probability[1][s_ind_next], k=1)[0]
        o_next = pomdp.observations[1][o_ind_next]
        belief = get_next_belief(pomdp, belief, a, o_next, 1)[0]
        # DS: extra, print belief
        non_z_belief = list(compress(count(), belief))
        non_z_belief_sb = [str(pomdp.states[1][i]) + ": " + str(belief[i]) for i in non_z_belief]
        if f_out is not None:
            f_out.write('B: {')
            for sb in non_z_belief_sb:
                f_out.write(sb + ', ')
            f_out.write('}\n')
        s, o = s_next, o_next
        s_ind, o_ind = s_ind_next, o_ind_next
    a = alpha_vector_policy(pomdp, policy, belief)
    r = pomdp.rewards[1][a][s_ind]
    if f_out is not None:
        f_out.writelines("S:{}, O:{}, A:{}, R:{}\n".format(s, o, a, r))
    traj.append((t + 1, s, o, a))
    traj_ind.append((t + 1, s_ind, o_ind, a))
    return traj, traj_ind


def near(location1, location2):
    """True if location1 and location2 are 1 step within each other
    locations are in format (x, y)
    """
    if location1[0] == location2[0]:
        if location2[1] - 1 <= location1[1] <= location2[1] + 1:
            return True
    if location1[1] == location2[1]:
        if location2[0] - 1 <= location1[0] <= location2[0] + 1:
            return True
    return False


def generate(m, n, horizon, p, r, n_agent, agent_pri, start, private, rew, rewards, switch, p_stay,
             collide, stay, specifications, thres=0.5):
    """Generate constraint_pomdp, replace save & load pickle for performance
    """
    # generate common-information pomdp from gridworld
    pomdp = gridworld_middle(m, n, stay, horizon,
                             p, r,
                             n_agent, agent_pri, start, private,
                             rew, rewards,
                             switch, p_stay)
    # DFA from specifications
    A = DFA(specifications, '/Users/shendongming/PycharmProjects/pythonProject/venv/lib/python3.9/site-packages')
    # create & assign labels
    label_v = [()] * (len(pomdp.states[1]))
    for s_i, s in enumerate(pomdp.states[1]):
        al, x1t, x2t = s
        # both agents in collide & collide
        if al[0] in collide and al[1] in collide and near(al[0], al[1]):
            label_v[s_i] += ('c',)
        # both agents in stay
        if al[0] in stay and al[1] in stay:
            label_v[s_i] += ('s',)
    print(label_v)
    labels = {t: deepcopy(label_v) for t in range(1, horizon + 1)}
    prod_pomdp = combine(A, pomdp, labels, thres)
    print("prod_pomdp generation complete")
    return prod_pomdp


def evaluate_policy_mc(pomdp_model, policy, horizon, n_samples, disc, f_out):
    """EVAL(policy): run trajectory for n_samples times and get the avg reward & constraint
    :param pomdp_model: Constrained Product POMDP model
    :param policy: policy returned from SARSOP
    :param horizon: horizon to use in each trajectory
    :param n_samples: number of trajectories in the simulation
    :param disc: discount factor
    :param f_out: output file
    :return reward: avg reward from simulation EVAL (discounted) - float
    :return constraint: avg constraint from simulation EVAL (discounted) - dict of float, key=constraint_indices
    """
    reward = 0
    constraint = {k: 0 for k in pomdp_model.constraint_indices}
    for iteration in range(n_samples):
        if iteration == 0:
            traj = trajectory_out(pomdp_model, policy, horizon, f_out)
        else:
            traj = trajectory_out(pomdp_model, policy, horizon)
        for t, s, o, a in traj[1]:
            reward += (disc ** (t - 1)) * pomdp_model.rewards[1][a][s]
            for k in pomdp_model.constraint_indices:
                if (pomdp_model.horizon, k) in pomdp_model.constraints:
                    constraint[k] += (disc ** (t - 1)) * pomdp_model.constraints[(pomdp_model.horizon, k)][a][s]
    reward /= (n_samples / (1 - disc))
    for k in pomdp_model.constraint_indices:
        constraint[k] /= (n_samples / (1 - disc))
    return reward, constraint


def expo_gradient(pomdp_model, pomdpsol_path, output_path, display_path, discount, timeout, memory, precision, val_b,
                  total_k, num_it_simulation, time_step, eta, delta, lamb_arr, avg_reward_woc_arr, avg_constraint_arr,
                  is_tuning):
    """Run Exponentiated Gradient Method on a Constrained Product POMDP
    :param pomdp_model: Constrained Product POMDP model
    :param pomdpsol_path: path to the `pomdpsol` binary for SARSOP
    :param output_path: path to output file
    :param display_path: path to graph
    :param discount: discount factor
    :param timeout: SARSOP time limit (in seconds)
    :param memory: SARSOP memory limit (in mb)
    :param precision: SARSOP absolute `precision`
    :param val_b: B value
    :param total_k: K value (# iterations)
    :param num_it_simulation: number of simulations to run in each EVAL() step
    :param time_step: time step in each simulation
    :param eta: η value
    :param delta: δ value
    :param lamb_arr: the array to store lambda - RETURN
    :param avg_reward_woc_arr: the array to store avg discounted reward from EVAL() - RETURN
    :param avg_constraint_arr: the array to store avg discounted constraint from EVAL() - RETURN
    :param is_tuning: True (for tuning only) or False
    """

    """ preparation """
    # load the class from pickle, open f_out
    constrained_pomdp = pomdp_model
    f_out = open(output_path, "w")
    # GET initial_dist, states, actions, observations, observation_probability, transitions, rewards, constraints
    ID, S, A, A_original, O, Op, T, R, C = pomdpltlf_prepare(constrained_pomdp)

    """ Exponentiated Gradient Method Body """
    "Tuning"
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
    # not for tuning hyper-parameters => run normally
    else:
        lamb_new = 1
        lamb = val_b / 10  # initialize lambda

    "Constants initialization"
    e = np.exp(1)  # e^1
    if time_step == 0:
        time_step = math.ceil(np.log(0.03) / np.log(discount))

    "Loop body"
    for k in range(total_k):
        "get new reward (CR) by the formula in article"
        lamb_arr[k] = lamb
        CR = {}
        for ia, a in enumerate(A_original):
            CR[("a" + str(ia))] = \
                (lamb_new * R["a" + str(ia)] + lamb * C["a" + str(ia)]) * (1 - discount) / (lamb_new + lamb)

        "create POMDP_LTLf"
        pomdp_k = POMDP_LTLf(initial_dist=ID,
                             states=S,
                             actions=A,
                             observations=O,
                             observation_probability=Op,
                             transitions=T,
                             rewards=CR)

        "solve POMDP problem with SARSOP get policy_k (k=0: solve normally, k>0: use modify_reward_only=True)"
        tic = tictoc.perf_counter()
        if k == 0:
            policy = sarsop(pomdp_k.agent,
                            pomdpsol_path,
                            discount_factor=discount,
                            timeout=timeout,
                            memory=memory,
                            precision=precision,
                            pomdp_name="/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/temp-pomdp",
                            remove_generated_files=False,
                            modify_reward_only=False)  # false if no pre-saved file exists
        else:
            policy = sarsop(pomdp_k.agent,
                            pomdpsol_path,
                            discount_factor=discount,
                            timeout=timeout,
                            memory=memory,
                            precision=precision,
                            pomdp_name="/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/temp-pomdp",
                            remove_generated_files=False,
                            modify_reward_only=True)
        toc = tictoc.perf_counter()
        print("Time for solving iteration k =", (toc - tic))
        "simulation using the policy_k, get {p_k} = avg constraint"
        if tune == "r":
            return  # if tuning for reward, should quit here
        f_out.writelines("\n==========[Running Simulation at k={}, where lambda={}]==========\n".format(k + 1, lamb))
        reward_est, constraint_est = evaluate_policy_mc(constrained_pomdp, policy, time_step, num_it_simulation,
                                                        discount, f_out)
        print("end of iteration k =", k + 1)

        "update lambda with the formula"
        p_k = constraint_est[constrained_pomdp.constraint_indices[0]]  # get avg constraint from constraint_est dict
        power = e ** (eta * (-p_k + 1 - delta))
        avg_reward_woc_arr[k] = reward_est
        avg_constraint_arr[k] = p_k
        lamb = val_b * ((lamb * power) / (val_b - lamb + (lamb * power)))  # update lambda
        if tune == "c":
            return  # if tuning for constraint, should quit here

    """ Insert Final Results at TOP of the file """
    f_out.close()
    f_out = open(output_path, 'r+')
    old_lines = f_out.readlines()  # read old content
    f_out.seek(0)  # go back to the beginning of the file
    f_out.writelines("==========Model information==========\n")
    f_out.writelines("TODO")
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
    # output_path = "results_mid/ISR/isr1x1.txt"
    # display_path = "results_mid/ISR/isr1x1.png"
    output_path = "/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/results_mid/ma_grid3x3/ma_grid5x5_pri2x2_swi_final.txt"
    display_path = "/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/results_mid/ma_grid3x3/ma_grid5x5_pri2x2_swi_final.png"
    # output_path = "results_mid/ma_grid3x3/ma_grid3x3_pri2x1_swi_conf.txt"
    # display_path = "results_mid/ma_grid3x3/ma_grid3x3_pri2x1_swi_conf.png"
    # output_path = "results_mid/ma_grid7x7/ma_grid7x7_pri2x1_swi_conf_3.txt"
    # display_path = "results_mid/ma_grid7x7/ma_grid7x7_pri2x`1_swi_conf_3.png"
    pomdpsol_path = "/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/appl/src/pomdpsol"  # Path to the `pomdpsol` binary

    "create pomdp model"
    # constrained_pomdp = generate(m=4, n=4, horizon=2, p=1, r=1, n_agent=2,
    #                              agent_pri=[[(0, 3)], [(3, 3)]],
    #                              start=[(3, 0), (0, 1)],
    #                              private="goal",
    #                              rew="goal", rewards=[[1], [1]],
    #                              switch=False, p_stay=1.0,
    #                              collide=[(0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0),
    #                                       (2, 2), (2, 3), (3, 0), (3, 2), (3, 3)],
    #                              stay=[(0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0),
    #                                    (2, 2), (2, 3), (3, 0), (3, 2), (3, 3)],
    #                              specifications="G(s & !c)",
    #                              thres=0.5)
    # constrained_pomdp = generate(m=5, n=5, horizon=2, p=1, r=1, n_agent=2,
    #                              agent_pri=[[(2, 4), (0, 2)], [(4, 2), (2, 0)]],
    #                              start=[(2, 0), (0, 2)],
    #                              private="goal",
    #                              rew="goal", rewards=[[1, 1], [1, 1]],
    #                              switch=True, p_stay=0.8,
    #                              collide=[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (0, 2), (1, 2), (3, 2), (4, 2)],
    #                              stay=[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (0, 2), (1, 2), (3, 2), (4, 2)],
    #                              specifications="G(s & !c)",
    #                              thres=0.5)
    constrained_pomdp = generate(m=3, n=3, horizon=2, p=1, r=1, n_agent=2,
                                 agent_pri=[[(1, 2), (0, 1)], [(1, 2)]],
                                 start=[(1, 0), (0, 1)],
                                 private="goal",
                                 rew="goal",
                                 rewards=[[10, 10], [1]],
                                 switch=True, p_stay=0.8,
                                 collide=[(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)],
                                 stay=[(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)],
                                 specifications="G(s & !c)",
                                 thres=0.5)
    # constrained_pomdp = generate(m=7, n=7, horizon=2, p=1, r=1, n_agent=2,
    #                              agent_pri=[[(3, 6), (0, 3)], [(3, 6)]],
    #                              start=[(3, 0), (6, 3)],
    #                              private="goal",
    #                              rew="goal", rewards=[[3, 3], [1]],
    #                              switch=True, p_stay=0.8,
    #                              collide=[(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
    #                                       (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6)],
    #                              stay=[(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
    #                                    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6)],
    #                              specifications="G(s & !c)",
    #                              thres=0.5)
    "parameters for the algorithm"
    total_k = 50  # 50
    lamb_arr = np.zeros(total_k)  # the array to store lambda
    avg_reward_woc_arr = np.zeros(total_k)  # the array to store avg discounted reward from EVAL()
    avg_constraint_arr = np.zeros(total_k)  # the array to store avg discounted constraint from EVAL()
    discount = 0.99
    timeout = 300
    memory = 1024
    precision = 0.05
    B = 50
    eta = 2
    delta = 0.05
    num_it_simulation = 50  # 50

    "Run Exponentiated Gradient"
    expo_gradient(pomdp_model=constrained_pomdp,
                  pomdpsol_path=pomdpsol_path,
                  output_path=output_path,
                  display_path=display_path,
                  discount=discount,  # discount factor
                  timeout=timeout,  # (seconds) to run the algorithm until termination
                  memory=memory,  # (mb) to run the algorithm until termination 1gb
                  precision=precision,  # solver runs until this absolute `precision`
                  val_b=B,  # B value in algorithm
                  total_k=total_k,  # K value in algorithm (number of iterative updating)
                  num_it_simulation=num_it_simulation,  # number of simulations in each EVAL() step
                  time_step=0,  # time step in each simulation = ceiling(ln(0.03)/ln(discount))
                  eta=eta,  # η value in algorithm (how sensitive)
                  delta=delta,  # δ value in algorithm
                  lamb_arr=lamb_arr,
                  avg_reward_woc_arr=avg_reward_woc_arr,
                  avg_constraint_arr=avg_constraint_arr,
                  is_tuning=False)
    print("==========DONE==========")


if __name__ == '__main__':
    main()
