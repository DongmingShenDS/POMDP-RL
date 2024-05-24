#!/usr/bin/env python
# coding: utf-8

# In[1]:
from gridworld_reward import gridworld
from automation import DFA
from pomdp import constrained_pomdp
from combine_reward import combine
import pickle
from copy import deepcopy
import numpy as np

# In[2]: specifications
specifications = '(a3 -> (!a2 U (a1 & F(a2)))) & (!a3 -> (!a1 U (a2 & F(a1))))'
A = DFA(specifications, '/Users/dongmingshen/PycharmProjects/AlvinLab/venv/lib/python3.9/site-packages')
print(A.acc)
print(A.n_qs)
print(A.T)

# In[3]: overall settings
m = 4
n = 4
flags = [0, 1]
p = 1  # Transition uncertainity
r = 1  # Grid location uncertainity
horizon = 10
thres = 0.1
n_flag = len(flags)
f_dict = {flags[i]: i for i in range(n_flag)}

# In[5]: locations
a = (0, 3)
b = (3, 0)
f_loc = (2, 2)  # where the station is located
goal = (3, 3)
# goal = "DFA"

# In[6]: rewards
rew = np.zeros(m * n + 2, dtype=float) - 1
rew[goal[0] * m + goal[1]] = 0

# In[8]: pomdp
pomdp = gridworld(m, n, flags, horizon, p, r, rew, f_loc, goal)
print("pomdp")

# In[9]: labels
# label_v = [deepcopy([deepcopy([()]*m) for i in range(n)]) for j in range(flag)]
# flat_label_v = [x3 for x1 in label_v for x2 in x1 for x3 in x2]
label_v = [()] * ((m * n + 2) * n_flag)
for s, o in pomdp.states[1]:
    x, y = s[0], s[1]
    if o == 0 and s != a and s != b:
        label_v[(n * x + y) * n_flag + f_dict[o]] = ('a3',)
    else:
        if (s, o) == (a, 0):
            label_v[(n * x + y) * n_flag + f_dict[o]] = ('a1', 'a3')
        if (s, o) == (a, 1):
            label_v[(n * x + y) * n_flag + f_dict[o]] = ('a1',)
        if (s, o) == (b, 0):
            label_v[(n * x + y) * n_flag + f_dict[o]] = ('a2', 'a3')
        if (s, o) == (b, 1):
            label_v[(n * x + y) * n_flag + f_dict[o]] = ('a2',)
labels = {t: deepcopy(label_v) for t in range(1, horizon + 1)}

# In[10]: prod pomdp
prod = combine(A, pomdp, labels, thres, goal, flags)
print("prod pomdp")

# In[12]: pickle
pickle.dump(prod, open("models/p_sequence_gridgoal", "wb"))
print("pickle complete")
