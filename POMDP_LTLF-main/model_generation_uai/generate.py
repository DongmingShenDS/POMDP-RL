from gridworld_reward import gridworld
from automation import DFA
from pomdp import constrained_pomdp
from combine_reward import combine
import pickle
from copy import deepcopy
import numpy as np

A = DFA('F(a) & G(!d)', '/Users/dongmingshen/PycharmProjects/AlvinLab/venv/lib/python3.9/site-packages')
print("DFA")
m = 4
n = 4
p = 0.95
horizon = 10
thres = 0.5
rew = np.zeros(m*n,dtype=float)
rew[3] = 2.0
rew[15] = 1.0
pomdp = gridworld(m,n,horizon,p,rew)
print("pomdp")
label_v = [()]*(m*n)
label_v[15] = ('a',)
label_v[6] = ('d',)
labels = {t:deepcopy(label_v) for t in range(1,horizon+1)}
prod = combine(A, pomdp,labels,thres)
print("prod")
pickle.dump(prod, open('case1.1', 'wb'))
print("complete")
exit(0)