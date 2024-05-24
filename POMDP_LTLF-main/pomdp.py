import numpy as np

import math
import random
from copy import deepcopy


class constrained_pomdp:

    def __init__(self,
                 initial_dist,
                 states, actions,
                 transitions,
                 observations,
                 observation_probability,
                 rewards,
                 constraints,
                 constraint_val,
                 constraint_indices,
                 horizon):
        """
        initial_dist
        states                      - how to pass in? MyState only stores 1 state - probably using
        actions                     - how to pass in?
        transitions
        observations
        observation_probability
        rewards
        constraints                 (not considered here)
                                    - PolicyModel?
        """

        """
        initial_dist
        """
        self.initial_dist = initial_dist

        """
        states is a dictionary. 
        For a given time t, states[t] is a list representing the state space at time t.
        """
        self.states = states

        """
        actions is a dictionary. 
        For a given time t, actions[t] is a list representing the action space at time t.
        """
        self.actions = actions

        """
        transitions is a dictionary. 
        For a given time t, transitions[t] is a dictionary. 
        For a given action a, transitions[t][a] is a numpy array representing the transition prob matrix for action a.
        """
        self.transitions = transitions

        """
        observations is a dictionary. 
        For a given time t, observations[t] is a list representing the observation space at time t.
        """
        self.observations = observations
        self.observation_index = {}
        if observations:
            for t in range(1, horizon + 1):
                count = 0
                for o in observations[t]:
                    self.observation_index[(t, o)] = count
                    count += 1

        """
        observation_probability is a dictionary. 
        For a given time t, observation_probability[t] represents the observation kernel at time t. 
        The observation kernel is a two-dimensional numpy array with states on the rows and observations on the columns.
        """
        self.observation_probability = observation_probability

        """
        rewards is a dictionary. 
        For a given time t, rewards[t] is a dictionary. 
        For a given action a, rewards[t][a] is a numpy array representing the instantaneous reward associated with each 
            state under action a.
        """
        self.rewards = rewards

        """
        constraints is a dictionary. 
        For a given time and constraint k, constraints[(t,k)] is a dictionary. 
        For a given action a, rewards[(t,k)][a] is a numpy array representing the instantaneous constraint-reward 
            associated with each state under action a.
        """
        self.constraints = constraints
        self.constraint_val = constraint_val
        self.constraint_indices = constraint_indices

        """
        horizon
        """
        self.horizon = horizon
