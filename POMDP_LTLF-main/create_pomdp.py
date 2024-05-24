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
        action_original = self.action
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
        self.observations = observations
        self.observation_probability = observation_probability
        self.states = states

        # cast initial_dist from numpy array into a Histogram distribution initial_belief
        initial_belief = list_to_hist(initial_dist, states)
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


def create_pomdp_help(s, L, M, p_m, q_m, r_0, r_1):
    size_states = L * M;
    size_action = 2;
    size_obs = L;

    states = []
    for ss in range(L):
        for mm in range(M):
            states.append((ss, mm))  # need to cast to String/Index to use Sarsop?

    actions = ["left", "right"]

    # choose any action, the transition correspond to the prob-mateix storing prob[state][next state] - ?
    transitions = {}
    transitions["left"] = []  # s' = max(s-1,0) with probability 1, need to be done by case?
    transitions["right"] = []  # s' = ...

    observations = [s for s in range(L)]  # the observation space
    observations_prob = np.zeros((size_states,
                                  size_obs))  # the observation prob kernel - states on row and observations on col
    for s_i in states:
        for o_i in observations:
            if s_i[0] == o_i:
                observations_prob[states.index(s_i)][observations.index(o_i)] = 1

    # for any action, the reward correspond to the array storing reward[action][state]
    reward = {"left": np.zeros(size_states), "right": np.zeros(size_states)}
    for s_i in states:
        if s_i[0] == 0:
            reward["left"][states.index(s_i)] = r_0
            reward["right"][states.index(s_i)] = r_0
        elif s_i[0] == L - 1:
            reward["left"][states.index(s_i)] = r_1
            reward["right"][states.index(s_i)] = r_1

    # initial dist: s given as s parameter, m randomly pick from 0...M-1
    initial_dist = np.zeros(size_states)
    for s_i in states:
        if s_i[0] == s:
            initial_dist[states.index(s_i)] = 1/M

    return initial_dist, states, actions, transitions, observations, observations_prob, reward


def create_pomdp(s, L, M, p_m, q_m, r_0, r_1):
    initial_dist, states_original, actions_original, transitions_original, observations_original, observations_prob, \
        rewards_original = create_pomdp_help(s, L, M, p_m, q_m, r_0, r_1)
    # print(initial_dist)
    # print(states_original)
    # print(actions_original)
    # print(transitions_original)
    # print(observations_original)
    # print(observations_prob)
    # print(rewards_original)

    # states, cast to string
    states = ["s" + str(i) for i in range(len(states_original))]
    states_dict = {}
    for i in range(len(states)):
        states_dict[states[i]] = states_original[i]
    states_class = [MyState(s) for s in states]

    # action, cast to string, length > 1, already satisfied?
    actions = actions_original
    actions_class = [MyAction(a) for a in actions]

    # observations, cast to string
    observations = ["o" + str(i) for i in range(len(observations_original))]
    observation_dict = {}
    for i in range(len(observations)):
        observation_dict[observations[i]] = observations_original[i]
    observations_class = [MyObservation(o) for o in observations]

    # observation_probability
    observation_probability = observations_prob.tolist()

    # transition, map with new actions, already satisfied?
    transitions = {}
    for a in actions_original:
        transitions[a] = transitions_original[a]

    # reward, map with new actions, already satisfied?
    rewards = copy.deepcopy(rewards_original)
    for a in actions_original:
        rewards[a] = rewards.pop(a)

    # NOW THESE ARE READY TO PASSED INTO A SARSOP SOLVER
    # print(states_class)
    # print(actions_class)
    # print(transitions)
    # print(observations_class)
    # print(observation_probability)
    # print(rewards)

    pomdp = POMDP_LTLf(initial_dist,
                       states_class,
                       actions_class,
                       observations_class,
                       observation_probability,
                       transitions,
                       rewards)

    # return the pomdp to run sarsop on
    return pomdp


def main():
    pomdpsol_path = "/home/pomdp_solve/pomdp_ltlf/pomdp_ltlf_code/sarsop/src/pomdpsol"
    # here you set up the params
    s = 1
    L = 10
    M = 30
    p_m = 0.3
    q_m = 0.5
    r_0 = 1
    r_1 = 5
    # call create_pomdp, will return the pomdp ready to be solved using sarsop
    pomdp = create_pomdp(s, L, M, p_m, q_m, r_0, r_1)
    # solve the pomdp using sarsop
    discount = 0.95
    timeout = 60  # max seconds
    memory = 200  # max mb
    precision = 0.005  # abs precision
    policy = sarsop(pomdp.agent,
                    pomdpsol_path,
                    discount_factor=discount,
                    timeout=timeout,
                    memory=memory,
                    precision=precision,
                    remove_generated_files=False)
    # use the policy for simulation (using package builtin methods)
    traj_timestep = 100
    for step in range(traj_timestep):
        # get action using .plan
        action = policy.plan(pomdp.agent)
        # get reward & execute transition
        reward = pomdp.env.state_transition(action, execute=True)
        # get observation
        observation = pomdp.agent.observation_model.sample(pomdp.env.state, action)
        print(pomdp.agent.cur_belief, action, observation, reward)
        # update belief
        new_belief = pomdp_py.update_histogram_belief(pomdp.agent.cur_belief,
                                                      action,
                                                      observation,
                                                      pomdp.agent.observation_model,
                                                      pomdp.agent.transition_model)
        pomdp.agent.set_belief(new_belief)
    return


if __name__ == '__main__':
    main()