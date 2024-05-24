import os
import pickle
import sys
from copy import deepcopy
import numpy as np
from combine import combine
from itertools import compress, count

# from gridworld_reward import gridworld
from automation import DFA

# agent_pri = [[(2, 4), (0, 2)], [(4, 2)]]
# a = [0] * 5
# print(a)

# belief = np.array([1/3, 0, 0, 0, 1/3, 0])
belief = [1/3, 0, 0, 0, 1/3, 0]
compress(count(), belief)
states = ['a', 'b', 'c', 'd', 'e']
pos_b = np.nonzero(belief)
pos_b = list(compress(count(), belief))
print(list(compress(count(), belief)))
s = [states[i] for i in pos_b]
b = [belief[i] for i in pos_b]
st = [str(states[i]) + ": " + str(belief[i]) for i in pos_b]
for i in st:
    print(i, end=', ')
print(end='')

exit(0)











# s = "S:((((1, 0), (0, 1)), (1, 2), (2, 1)), '1'), O:((1, 0), (0, 1)), A:((((1, 2), 'L'),), (((2, 1), 'D'),)), R:0.0"
# ag1_l = s[s.index("S:"):][6:10]
# ag2_l = s[s.index("S:"):][14:18]
# ag1_g = s[s.index("S:"):][23:27]
# ag2_g = s[s.index("S:"):][31:35]
# ag1_state = eval(ag1_l)
# ag2_state = eval(ag2_l)
# ag1_goal = eval(ag1_g)
# ag2_goal = eval(ag2_g)
# ag1_action = s[s.index("A:"):][14:15]
# ag2_action = s[s.index("A:"):][32:33]


# print(ag1_state, ag2_state)
#
# exit(0)






# m = 3
# n = 3
# horizon = 2
# p = 1
# r = 1
# n_agent = 2
# agent_pri = [[(1, 2), (2, 1)], [(2, 1)]]
# start = [(1, 0), (0, 1)]
# private = "goal"
# rew = "goal"
# pomdp = gridworld(m, n, horizon, p, r, n_agent, agent_pri, start, private, rew)
# collide = [(1, 1)]
# avoid = [(0, 0), (0, 2), (2, 0), (2, 2)]
# thres = 0.5
# specifications = "G(!c & !a)"
# A = DFA(specifications, '/Users/dongmingshen/PycharmProjects/AlvinLab/venv/lib/python3.9/site-packages')
# label_v = [()] * (len(pomdp.states[1]))
# for s_i, s in enumerate(pomdp.states[1]):
#     al, x1t, x2t = s
#     if al[0] in collide and al[1] in collide:
#         label_v[s_i] = ('c',)
#     if al[0] in avoid or al[1] in avoid:
#         label_v[s_i] = ('a',)
# for i in range(len(label_v)):
#     if label_v[i] == ('c',):
#         print(pomdp.states[1][i])
# exit(0)
# labels = {t: deepcopy(label_v) for t in range(1, horizon+1)}
# prod_pomdp = combine(A, pomdp, labels, thres)
# exit(0)
#
#
#
#
#
# m = 3
# n = 3
# horizon = 2
# p = 1
# r = 1
# n_agent = 2
# agent_pri = [[(1, 2)], [(2, 1)]]
# start = [(1, 0), (0, 1)]
# private = "goal"
# rew = "goal"
# pomdp = gridworld(m, n, horizon, p, r, n_agent, agent_pri, start, private, rew)
# collide = [(1, 1)]
# avoid = [(0, 0), (0, 2), (2, 0), (2, 2)]
#
# # constrained_pomdp = generate(m=3, n=3, horizon=2, p=1, r=1, n_agent=2,
# #                              agent_pri=[[(1, 2), (0, 1)], [(2, 1), (1, 0)]],
# #                              start=[(1, 0), (0, 1)],
# #                              private="goal",
# #                              rew="goal",
# #                              switch=True,
# #                              collide=[(1, 1)],
# #                              stay=[(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)],
# #                              specifications="G(s & !c)",
# #                              thres=0.5)
#
#
#
# exit(0)
