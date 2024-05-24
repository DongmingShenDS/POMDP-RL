
import trajectory as tj

import numpy as np

import math
import random
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

import pickle


import numpy as np
import math
import random
from copy import deepcopy
def trajectory(pomdp,policy,belief_sets,conv_combs):
    #return single trajectory in true form and indice form
    #i.e. (t,s,o,a) , (t,index(s),index(o),index(a))
    belief_indices = {(t,tuple(b)):i for t in belief_sets for i,b in enumerate(belief_sets[t])}
    traj = []
    traj_ind = []
    t = 1
    s_ind = random.choices(range(len(pomdp.states[t])),weights = pomdp.initial_dist,k=1)[0]
    s = pomdp.states[t][s_ind]
    o_ind = random.choices(range(len(pomdp.observations[t])),weights=pomdp.observation_probability[t][s_ind],k=1)[0]
    o = pomdp.observations[t][o_ind]
    b_ind = belief_indices[(t,tuple(pomdp.initial_dist))]
    a_ind = random.choices(range(len(pomdp.actions[t])), weights = [policy[(t,b_ind,x)] for x in pomdp.actions[t]],k=1)[0]
    a = pomdp.actions[t][a_ind]
    traj.append((t,s,o,a))
    traj_ind.append((t,s_ind,o_ind,a))
    s_ind_next = random.choices(range(len(pomdp.states[t+1])),weights=pomdp.transitions[t][a][s_ind],k=1)[0]
    s_next = pomdp.states[t+1][s_ind_next]
    o_ind_next = random.choices(range(len(pomdp.observations[t+1])),weights = pomdp.observation_probability[t+1][s_ind_next],k=1)[0]
    o_next = pomdp.observations[t+1][o_ind_next]
    b_ind = random.choices(range(len(belief_sets[t+1])),weights=conv_combs[(t,b_ind,a,o)][1],k=1)[0]
    s,o = s_next,o_next
    s_ind,o_ind = s_ind_next,o_ind_next
    for t in range(2,pomdp.horizon):
        p = [policy[(t,b_ind,x)] for x in pomdp.actions[t]]
        a_ind = random.choices(range(len(pomdp.actions[t])), weights = [policy[(t,b_ind,x)] for x in pomdp.actions[t]],k=1)[0]
        a = pomdp.actions[t][a_ind]
        traj.append((t,s,o,a))
        traj_ind.append((t,s_ind,o_ind,a))
        s_ind_next = random.choices(range(len(pomdp.states[t+1])),weights=pomdp.transitions[t][a][s_ind],k=1)[0]
        s_next = pomdp.states[t+1][s_ind_next]
        o_ind_next = random.choices(range(len(pomdp.observations[t+1])),weights = pomdp.observation_probability[t+1][s_ind_next],k=1)[0]
        o_next = pomdp.observations[t+1][o_ind_next]
        b_ind = random.choices(range(len(belief_sets[t+1])),weights=conv_combs[(t,b_ind,a,o)][1],k=1)[0]
        s,o = s_next,o_next
        s_ind,o_ind = s_ind_next,o_ind_next
    a_ind = random.choices(range(len(pomdp.actions[t+1])), weights = [policy[(t+1,b_ind,x)] for x in pomdp.actions[t+1]],k=1)[0]
    a = pomdp.actions[t+1][a_ind]
    traj.append((t+1,s,o,a))
    traj_ind.append((t+1,s_ind,o_ind,a))
    return traj,traj_ind


class constrained_pomdp:

    def __init__(self,initial_dist,states,actions,transitions,observations,observation_probability,rewards,constraints,constraint_val,constraint_indices,horizon):


        self.initial_dist = initial_dist

        """
        states is a dictionary. For a given time t, states[t] is a list representing the state space at time t.
        """
        self.states = states


        """
        actions is a dictionary. For a given time t, actions[t] is a list representing the action space at time t.
        """
        self.actions = actions


        """
        transitions is a dictionary. For a given time t, transitions[t] is a dictionary. For a given action a, transitions[t][a] is a numpy array representing the transition probability matrix for action a.
        """
        self.transitions = transitions


        """
        observations is a dictionary. For a given time t, observations[t] is a list representing the observation space at time t.
        """
        self.observations = observations
        self.observation_index = {}

        if observations:
            for t in range(1,horizon+1):
                count = 0
                for o in observations[t]:
                    self.observation_index[(t,o)] = count
                    count += 1


        """
        observation_probability is a dictionary. For a given time t, observation_probability[t] represents the observation kernel at time t. The observation kernel is a two-dimensional numpy array with states on the rows and observations on the columns.
        """
        self.observation_probability = observation_probability


        """
        rewards is a dictionary. For a given time t, rewards[t] is a dictionary. For a given action a, rewards[t][a] is a numpy array representing the instantaneous reward associated with each state under action a.
        """
        self.rewards = rewards


        """
        constraints is a dictionary. For a given time and constraint k, constraints[(t,k)] is a dictionary. For a given action a, rewards[(t,k)][a] is a numpy array representing the instantaneous constraint-reward associated with each state under action a.
        """
        self.constraints = constraints
        self.constraint_val = constraint_val
        self.constraint_indices = constraint_indices



        self.horizon = horizon




class simple_pomdp(constrained_pomdp):

    def __init__(self,initial_dist,states,actions,transitions,observations,observation_probability,reward,horizon):
        super().__init__(initial_dist,states,actions,transitions,observations,observation_probability,reward,{},[],horizon)


class constrained_mdp(constrained_pomdp):

    def __init__(self,initial_dist,states,actions,transitions,rewards,constraints,constraint_val,constraint_indices,horizon):
        super().__init__(initial_dist,states,actions,transitions,{},{},rewards,constraints,constraint_val,constraint_indices,horizon)






def get_next_belief(pomdp,cur_belief,action,observation,t):
    if t:
        state_size = len(pomdp.states[t])
    state_size_next = len(pomdp.states[t+1])
    next_belief = np.zeros(state_size_next)
    o_index = pomdp.observation_index[(t+1,observation)]

    for x_next in range(state_size_next):
        if t:
            for x in range(state_size):
                trans_prob = pomdp.transitions[t][action][x][x_next]
                obs_prob = pomdp.observation_probability[t+1][x_next][o_index]
                next_belief[x_next] += cur_belief[x]*trans_prob*obs_prob
        else:
            obs_prob = pomdp.observation_probability[t+1][x_next][o_index]
            next_belief[x_next] = cur_belief[x_next]*obs_prob

    obs_prob = sum(next_belief)
    if obs_prob:
        return (next_belief/obs_prob,obs_prob)
    else:
        return (np.ones(len(next_belief))/len(next_belief),obs_prob)



def find_conv_comb(belief_set,belief):

    m = gp.Model("lp")
    m.setParam('OutputFlag',0)

    bel_set_size = len(belief_set)
    bel_size = len(belief)

    var_dict = {}
    for i in range(bel_set_size):
        var_name = "weight"+str(i)
        var_dict[i] = m.addVar(lb = 0.0, ub = 1.0,vtype=GRB.CONTINUOUS, name=var_name)

    cost_exp = gp.LinExpr()
    for i in range(bel_set_size):
        coefficient = np.linalg.norm(belief_set[i]-belief,1)
        cost_exp.add(gp.LinExpr(var_dict[i]),coefficient)

    m.setObjective(cost_exp, GRB.MINIMIZE)

    sum_expr = gp.LinExpr()
    for i in range(bel_set_size):
        sum_expr.add(gp.LinExpr(var_dict[i]),1)
    m.addConstr(sum_expr == 1, "sumto1")


    for i in range(bel_size):
        conv_comb = gp.LinExpr()
        cons_name = "conv_comb"+str(i)
        for j in range(bel_set_size):
            conv_comb.add(gp.LinExpr(var_dict[j]),belief_set[j][i])
        m.addConstr(conv_comb == belief[i],cons_name)

    m.optimize()

    weight_list = []

    total = 0
    for i in range(len(var_dict)):
        weight_list.append(abs(var_dict[i].x))
        total += abs(var_dict[i].x)

    for w in weight_list:
        w /= total
    return (m.objVal,weight_list)








def construct_approx_mdp(belief_sets,pomdp,conv_combs):
    states = deepcopy(belief_sets)
    transitions = {}
    rewards = {}
    constraints = {}
    for t in pomdp.transitions:
        print("Time: "+str(t))
        transitions[t] = {}
        cur_belief_set = states[t]
        next_belief_set = states[t+1]
        for action in pomdp.actions[t]:
            transitions[t][action] = np.zeros([len(cur_belief_set),len(next_belief_set)])
            bel_count = 0
            for cur_belief in cur_belief_set:
                for obs in pomdp.observations[t+1]:
                    next_reach_belief = get_next_belief(pomdp,cur_belief,action,obs,t)
                    #weights = find_conv_comb(next_belief_set,next_reach_belief[0])[1]
                    weights = conv_combs[(t,bel_count,action,obs)][1]
                    transitions[t][action][bel_count,:] += next_reach_belief[1]*np.array(weights)
                bel_count += 1

    for t in pomdp.rewards:
        rewards[t] = {}
        cur_belief_set = states[t]
        for action in pomdp.actions[t]:
            rewards[t][action] = np.zeros([len(cur_belief_set)])
            bel_count = 0
            for cur_belief in cur_belief_set:
                rewards[t][action][bel_count] = np.inner(pomdp.rewards[t][action],cur_belief)
                bel_count += 1

    for (t,k) in pomdp.constraints:
        constraints[(t,k)] = {}
        cur_belief_set = states[t]
        for action in pomdp.actions[t]:
            constraints[(t,k)][action] = np.zeros([len(cur_belief_set)])
            bel_count = 0
            for cur_belief in cur_belief_set:
                constraints[(t,k)][action][bel_count] = np.inner(pomdp.constraints[(t,k)][action],cur_belief)
                bel_count += 1

    next_belief_set = states[1]
    initial_dist = np.zeros(len(next_belief_set))
    for obs in pomdp.observations[1]:
        next_reach_belief = get_next_belief(pomdp,pomdp.initial_dist,0,obs,0)
        #weights = find_conv_comb(next_belief_set,next_reach_belief[0])[1]
        weights = conv_combs[(0,0,0,obs)][1]
        initial_dist += next_reach_belief[1]*np.array(weights)

    return constrained_mdp(initial_dist,states,pomdp.actions,transitions,rewards,constraints,pomdp.constraint_val,pomdp.constraint_indices,pomdp.horizon)




def occupation_LP(mdp):

    m = gp.Model("lp")
    m.setParam('OutputFlag',0)
    states = mdp.states
    actions = mdp.actions
    var_dict = {}
    for t in states:
        for x in range(len(states[t])):
            for a in actions[t]:
                var_name = "time: "+str(t)+"\n state: "+str(x)+"\n action: "+str(a)
                var_dict[(t,x,a)] = m.addVar(lb = 0.0, ub = 1.0,vtype=GRB.CONTINUOUS, name=var_name)

    rew_exp = gp.LinExpr()
    for t in states:
        for x in range(len(states[t])):
            for a in actions[t]:
                coefficient = mdp.rewards[t][a][x]
                rew_exp.add(gp.LinExpr(var_dict[(t,x,a)]),coefficient)
    m.setObjective(rew_exp, GRB.MAXIMIZE)

    for t in states:
        for x in range(len(states[t])):
            out = gp.LinExpr()
            in_cons = gp.LinExpr()
            for a in actions[t]:
                out.add(gp.LinExpr(var_dict[(t,x,a)]),1)
            if t == 1:
                cons_name = "initial condition: "+str(x)
                m.addConstr(out == mdp.initial_dist[x],cons_name)
            else:
                for x_prev in range(len(states[t-1])):
                    for a in actions[t-1]:
                        coefficient = mdp.transitions[t-1][a][x_prev][x]
                        in_cons.add(gp.LinExpr(var_dict[(t-1,x_prev,a)]),coefficient)
                cons_name = "balance equation: "+str(t)+" state "+str(x)
                m.addConstr(out == in_cons,cons_name)


    for k in mdp.constraint_indices:
        cons_k = gp.LinExpr()
        for t in range(1,mdp.horizon+1):
            if (t,k) in mdp.constraints:
                for x in range(len(states[t])):
                    for a in actions[t]:
                        coefficient = mdp.constraints[(t,k)][a][x]
                        cons_k.add(gp.LinExpr(var_dict[(t,x,a)]),coefficient)
        cons_name = "Reward constraint: "+str(k)
        m.addConstr(cons_k >= mdp.constraint_val[k],cons_name)

    m.optimize()
    print('Printing status')
    print(m.Status)
    if m.Status == 3:
        return (-math.inf,{})

    occupation_measure = {}
    for t in states:
        for x in range(len(states[t])):
            for a in actions[t]:
                occupation_measure[(t,x,a)] = var_dict[(t,x,a)].x
    print("Value of the constraint in LP")
    print(cons_k.getValue())

    return (m.objval,occupation_measure)


def expand_belief_set(pomdp,belief_set,n_max):
    new_belief_set = {}
    max_conv_val = 0.0
    added_belief = None
    for obs in pomdp.observations[1]:
        next_belief = get_next_belief(pomdp,pomdp.initial_dist,0,obs,0)[0]
        if belief_set[1]:
            conv_comb_val = find_conv_comb(belief_set[1],next_belief)[0]
            if conv_comb_val > max_conv_val:
                max_conv_val = conv_comb_val
                added_belief = next_belief
        else:
            added_belief = next_belief
    new_belief_set[1] = deepcopy(belief_set[1])
    if added_belief is not None:
        new_belief_set[1].append(np.array(added_belief))
    for t in range(1,pomdp.horizon):
        print(f"Time: {t}")
        added_belief = None
        for belief in belief_set[t]:
            max_conv_val = 0.0
            for action in pomdp.actions[t]:
                for obs in pomdp.observations[t+1]:
                    next_belief = get_next_belief(pomdp,belief,action,obs,t)[0]
                    if belief_set[t+1]:
                        conv_comb_val = find_conv_comb(belief_set[t+1],next_belief)[0]
                        if conv_comb_val > max_conv_val:
                            max_conv_val = conv_comb_val
                            added_belief = next_belief
                    else:
                        added_belief = next_belief
        new_belief_set[t+1] = deepcopy(belief_set[t+1])
        if added_belief is not None:
            new_belief_set[t+1].append(deepcopy(added_belief))


    return new_belief_set

def is_far_enough(belief_set,belief,delta):

    min_norm = float('inf')
    for b in belief_set:
        coefficient = np.linalg.norm(b-belief,1)
        min_norm = min(min_norm,coefficient)
    if min_norm < delta:
        return False
    else:
        return True

def eps_greedy_expansion(pomdp,policy,belief_sets,conv_combs,n_traj,delta):
    new_belief_sets = deepcopy(belief_sets)

    for iter in range(n_traj):
        traj = tj.trajectory(pomdp,policy,belief_sets,conv_combs)
        cur_belief = pomdp.initial_dist
        a_prev = 0

        for t,s,o,a in traj[0]:
            potential_belief = get_next_belief(pomdp,cur_belief,a_prev,o,t-1)[0]
            added_belief = None
            a_prev = a
            if new_belief_sets[t]:
                if is_far_enough(new_belief_sets[t],potential_belief,delta):
                    added_belief = potential_belief
            else:
                added_belief = potential_belief
            if added_belief is not None:
                new_belief_sets[t].append(added_belief)
            cur_belief = potential_belief

    return new_belief_sets

def gen_policy(mdp,occupation_measure,printPolicy):
    policy = {}
    for t in range(1,mdp.horizon+1):
        for x in range(len(mdp.states[t])):
            total = 0
            for a in mdp.actions[t]:
                policy[(t,x,a)] = abs(occupation_measure[(t,x,a)])
                total += occupation_measure[(t,x,a)]
            if printPolicy:
                print("State occupancy: "+ str(total))
            if total:
                for a in mdp.actions[t]:
                    policy[(t,x,a)] /= total
                    if printPolicy:
                        print("Policy: "+ str((t,x,a))+"-> "+ str(policy[(t,x,a)]))
            else:
                for a in mdp.actions[t]:
                    policy[(t,x,a)] = 1.0/len(mdp.actions[t])
                    if printPolicy:
                        print("Policy: "+ str((t,x,a))+"-> "+ str(policy[(t,x,a)]))
    return policy


def eps_greedify(mdp,policy,printPolicy,eps):
    new_policy = {}
    for t in range(1,mdp.horizon+1):
        for x in range(len(mdp.states[t])):
            total = 0
            for a in mdp.actions[t]:
                new_policy[(t,x,a)] = policy[(t,x,a)]+eps
                total += new_policy[(t,x,a)]
            if printPolicy:
                print("State occupancy: "+ str(total))
            for a in mdp.actions[t]:
                new_policy[(t,x,a)] /= total
                if printPolicy:
                    print("Policy: "+ str((t,x,a))+"-> "+ str(new_policy[(t,x,a)]))
    return new_policy

def evaluate_policy(pomdp,approx_mdp,policy,conv_combs):


    occupancy = {}
    for xp in range(len(pomdp.states[1])):
        for bp in range(len(approx_mdp.states[1])):
            occupancy[(1,xp,bp)] = 0
            for obs in pomdp.observations[1]:
                #next_belief = get_next_belief(pomdp,pomdp.initial_dist,0,obs,0)[0]
                #weights = find_conv_comb(approx_mdp.states[1],next_belief)[1]
                weights = conv_combs[(0,0,0,obs)][1]
                occupancy[(1,xp,bp)] += pomdp.initial_dist[xp]*pomdp.observation_probability[1][xp][pomdp.observation_index[(1,obs)]]*weights[bp]
    for t in range(1,pomdp.horizon):
        print("Time: "+str(t))
        for xp in range(len(pomdp.states[t+1])):
            for bp in range(len(approx_mdp.states[t+1])):
                occupancy[(t+1,xp,bp)] = 0
                for x in range(len(pomdp.states[t])):
                    for a in pomdp.actions[t]:
                        for b in range(len(approx_mdp.states[t])):
                            for obs in pomdp.observations[t+1]:
                                #next_belief = get_next_belief(pomdp,approx_mdp.states[t][b],a,obs,t)[0]
                                #weights = find_conv_comb(approx_mdp.states[t+1],next_belief)[1]
                                weights = conv_combs[(t,b,a,obs)][1]
                                occupancy[(t+1,xp,bp)] += pomdp.transitions[t][a][x][xp]*policy[(t,b,a)]*occupancy[(t,x,b)]*pomdp.observation_probability[t+1][xp][pomdp.observation_index[(t+1,obs)]]*weights[bp]

    reward = 0
    for t in range(1,pomdp.horizon+1):
        for x in range(len(pomdp.states[t])):
            for b in range(len(approx_mdp.states[t])):
                for a in pomdp.actions[t]:
                    reward += occupancy[(t,x,b)]*policy[(t,b,a)]*pomdp.rewards[t][a][x]

    cons_reward = []
    for k in pomdp.constraint_indices:
        cons_k = 0
        for t in range(1,pomdp.horizon+1):
            if (t,k) in pomdp.constraints:
                for x in range(len(pomdp.states[t])):
                    for b in range(len(approx_mdp.states[t])):
                        for a in pomdp.actions[t]:
                            cons_k += occupancy[(t,x,b)]*policy[(t,b,a)]*pomdp.constraints[(t,k)][a][x]
        cons_reward.append(cons_k)

    return (occupancy,reward,cons_reward)

def evaluate_policy_mc(pomdp,policy,belief_sets,conv_combs,n_samples):


    total_reward = 0
    constraint_vals = {k: 0 for k in pomdp.constraint_indices}

    for iter in range(n_samples):
        traj = tj.trajectory(pomdp,policy,belief_sets,conv_combs)
        for t,s,o,a in traj[1]:
            total_reward += pomdp.rewards[t][a][s]
            for k in pomdp.constraint_indices:
                if (t,k) in pomdp.constraints:
                    constraint_vals[k] += pomdp.constraints[(t,k)][a][s]
    total_reward /= n_samples
    for k in pomdp.constraint_indices:
        constraint_vals[k] /= n_samples
    return total_reward,constraint_vals

def gen_cardinal_beliefs(pomdp):
    cardinal_belief_sets = {}
    for t in pomdp.states:
        print(t)
        states = pomdp.states[t]
        cardinal_belief_sets[t] = []
        for i in range(len(states)):
            cardinal_belief_sets[t].append(np.zeros(len(states)))
            cardinal_belief_sets[t][i][i] = 1
    return cardinal_belief_sets

def gen_all_conv_combs(pomdp,belief_set):
    conv_combs = {}
    for obs in pomdp.observations[1]:
        next_belief = get_next_belief(pomdp,pomdp.initial_dist,0,obs,0)[0]
        conv_combs[(0,0,0,obs)] = find_conv_comb(belief_set[1],next_belief)

    for t in range(1,pomdp.horizon):
        print("Time: "+str(t))
        bel_count = 0
        for belief in belief_set[t]:
            for action in pomdp.actions[t]:
                for obs in pomdp.observations[t+1]:
                    next_belief = get_next_belief(pomdp,belief,action,obs,t)[0]
                    conv_combs[(t,bel_count,action,obs)] = find_conv_comb(belief_set[t+1],next_belief)
            bel_count += 1
    return conv_combs


def print_bel_len(belief_sets,pomdp):
    for t in range(1,pomdp.horizon+1):
        print("Number of beliefs at time",t, " :",len(belief_sets[t]))

"""
A toy POMDP for debugging the algorithm.
"""
# initial_dist = np.array([0.5,0.5])
# states = {1:['s0','s1'],2:['s0','s1']}
# actions = {1:['a0','a1'],2:['a0','a1']}
# transitions = {1:{'a0':np.array([[1,0],[0,1]]),'a1':np.array([[0.5,0.5],[0.5,0.5]])}}
# observations = {1:['o0','o1'],2:['o0','o1']}
# observation_probability = {1:np.array([[0.8,0.2],[0.2,0.8]]),2:np.array([[0.8,0.2],[0.2,0.8]])}
# rewards = {1:{'a0':np.array([1,0]),'a1':np.array([0,1])},2:{'a0':np.array([1,0]),'a1':np.array([0,1])}}
# constraints = {(2,1):{'a0':np.array([1,0]),'a1':np.array([1,0])}}
# constraint_val = {1:0.5}
# constraint_indices = [1]
# horizon = 2
#
#
# toy_pomdp = constrained_pomdp(initial_dist,states,actions,transitions,observations,observation_probability,rewards,constraints,constraint_val,constraint_indices,horizon)



toy_pomdp = pickle.load(open('four_0.8_0.1_10','rb'))
#toy_pomdp = pickle.load(open('two_1_0.99_5','rb'))
#toy_pomdp = pickle.load(open('two_0.9_0.5_5','rb'))

print_pomdp = False
if print_pomdp:
    print(type(toy_pomdp))
    print(toy_pomdp)
    print("states")
    print((toy_pomdp.states))
    print("transitions")
    print((toy_pomdp.transitions))
    print("observation_probability")
    print((toy_pomdp.observation_probability))
    print("rewards")
    print((toy_pomdp.rewards))
    print("constraints")
    print((toy_pomdp.constraints))
    print('constraint_val')
    print(toy_pomdp.constraint_val)


belief_sets = gen_cardinal_beliefs(toy_pomdp)

iter_max = 20
for iter in range(iter_max):
    print("\n\n\nIteration: "+str(iter))

    print('\nGenerating convex combinations')
    conv_combs = gen_all_conv_combs(toy_pomdp,belief_sets)
    print_bel_len(belief_sets,toy_pomdp)

    print('\nConstructing approximate MDP')
    approx_mdp = construct_approx_mdp(belief_sets,toy_pomdp,conv_combs)
    #print(approx_mdp.constraints)
    print('\nApproximate MDP construction done')

    print('\nSolving LP')
    occupation_measure = occupation_LP(approx_mdp)
    print("Upper bound: "+str(occupation_measure[0]))

    if occupation_measure[0] != -math.inf:
        print('\nOccupation measure obtained - generating policy')
        printPolicy = False
        policy = gen_policy(approx_mdp,occupation_measure[1],printPolicy)
        print('\nPolicy generated')
        eps_policy = eps_greedify(approx_mdp,policy,printPolicy,0.5)
        if iter:
            print('\nEvaluating policy')
            # policy_occupancy = evaluate_policy(toy_pomdp,approx_mdp,policy,conv_combs)
            print("\nPolicy performance:")
            # print(policy_occupancy[1])
            # print(policy_occupancy[2])
            reward_est,constraint_est = evaluate_policy_mc(toy_pomdp,policy,belief_sets,conv_combs,1000)
            print(f"Estimated reward: {reward_est}\nEstimated constraint values: {constraint_est}")
    print('\nExpanding belief sets')
    #belief_sets = expand_belief_set(toy_pomdp,belief_sets,1)
    #print(belief_sets)
    belief_sets = eps_greedy_expansion(toy_pomdp,eps_policy,belief_sets,conv_combs,20,0.01)





print('\n\n\nDone!\n\n\n')
