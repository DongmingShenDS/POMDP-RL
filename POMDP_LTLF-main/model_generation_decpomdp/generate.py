from gridworld_reward import gridworld
from automation import DFA
from pomdp import constrained_pomdp
# from combine_reward import combine
import pickle
from copy import deepcopy
import numpy as np

m = 5
n = 5
horizon = 2
p = 1  # Transition uncertainity
r = 1  # Grid location uncertainity
# number of agents
n_agent = 2
# private information: agent_dst[0] for agent1 goal; agent_dst[1] for agent2 goal
# agent_pri = [[(0, 1), (1, 2)], [(1, 2), (2, 1)]]
agent_pri = [[(2, 4)], [(4, 2)]]
# private information type? (what else other than goal?)
private = "goal"
# initial states: start[0] for agent1 start; start[1] for agent2 start
start = [(2, 0), (0, 2)]
# reward type? (what else other than goal?)
rew = "goal"
pomdp = gridworld(m, n, horizon, p, r, n_agent, agent_pri, start, private, rew)
pickle.dump(pomdp, open('models/ma_pomdp5x5', 'wb'))
print("complete")
exit(0)