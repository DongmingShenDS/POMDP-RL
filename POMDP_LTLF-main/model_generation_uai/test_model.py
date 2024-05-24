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

case1_1_new = pickle.load(
    open('/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_uai/case1.1', 'rb'))
case1_1_old = pickle.load(
    open('/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_uai/case1_1', 'rb'))

initial_dist_new = case1_1_new.initial_dist.tolist()
initial_dist_old = case1_1_old.initial_dist.tolist()
print("initial_dist:", initial_dist_new == initial_dist_old)

states_new = case1_1_new.states
states_old = case1_1_old.states
print("states:", states_new == states_old)

actions_new = case1_1_new.actions
actions_old = case1_1_old.actions
print("actions:", actions_new == actions_old)

transitions_new = case1_1_new.transitions
transitions_old = case1_1_old.transitions
flag = True
for i in range(1, 10, 1):
    if 0 in (transitions_new[i]['U'] == transitions_old[i]['U']):
        flag = False
    if 0 in (transitions_new[i]['D'] == transitions_old[i]['D']):
        flag = False
    if 0 in (transitions_new[i]['L'] == transitions_old[i]['L']):
        flag = False
    if 0 in (transitions_new[i]['R'] == transitions_old[i]['R']):
        flag = False
print("transitions:", flag)

observations_new = case1_1_new.observations
observations_old = case1_1_old.observations
print("observations:", observations_new == observations_old)

observation_probability_new = case1_1_new.observation_probability
observation_probability_old = case1_1_old.observation_probability
flag = True
for i in range(1, 11, 1):
    if 0 in (observation_probability_new[i] == observation_probability_old[i]):
        flag = False
print("observation_probability:", flag)

rewards_new = case1_1_new.rewards
rewards_old = case1_1_old.rewards
flag = True
for i in range(1, 11, 1):
    if 0 in (rewards_new[i]['U'] == rewards_old[i]['U']):
        flag = False
    if 0 in (rewards_new[i]['D'] == rewards_old[i]['D']):
        flag = False
    if 0 in (rewards_new[i]['L'] == rewards_old[i]['L']):
        flag = False
    if 0 in (rewards_new[i]['R'] == rewards_old[i]['R']):
        flag = False
print("rewards:", flag)

constraints_new = case1_1_new.constraints
constraints_old = case1_1_old.constraints
flag = True
if 0 in (constraints_new[(10, 1)]['U'] == constraints_old[(10, 1)]['U']):
    flag = False
if 0 in (constraints_new[(10, 1)]['D'] == constraints_old[(10, 1)]['D']):
    flag = False
if 0 in (constraints_new[(10, 1)]['L'] == constraints_old[(10, 1)]['L']):
    flag = False
if 0 in (constraints_new[(10, 1)]['R'] == constraints_old[(10, 1)]['R']):
    flag = False
print("constraints:", flag)

constraint_val_new = case1_1_new.constraint_val
constraint_val_old = case1_1_old.constraint_val
print("constraint_val:", constraint_val_new == constraint_val_old)

constraint_indices_new = case1_1_new.constraint_indices
constraint_indices_old = case1_1_old.constraint_indices
print("constraint_indices:", constraint_indices_new == constraint_indices_old)

horizon_new = case1_1_new.horizon
horizon_old = case1_1_old.horizon
print("horizon:", horizon_new == horizon_old)

exit(0)