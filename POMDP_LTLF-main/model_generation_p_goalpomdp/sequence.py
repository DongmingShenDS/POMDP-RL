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


# In[2]:


A = DFA('!b U (a & F(b))','/Users/krishnachaitanyakalagarla/anaconda3/envs/multipomdp/lib/python3.7/site-packages')


# In[3]:


m = 4
n = 4

p = 0.95 #Transition uncertainity

qr_n = 0.1 # Obstacle uncertainity (right,near)
qr_a = 1 # Obstacle uncertainity (right,away)

ql_n = 0.9 # Obstacle uncertainity (left,near)
ql_a = 1 # Obstacle uncertainity (left,away)

r = 0.9 # Grid location uncertainity


# In[4]:


horizon = 10
thres = 0.1


# In[5]:


p_obstacle_locations = [(3,0),(0,3)]  # what is diff between p_obstacle_locations & obstacle_location?
obstacle_location = (3,0)

goal_location = (3,3)


# In[6]:


rew = np.zeros(m*n,dtype=float)


# In[7]:


rew[0] = 2


# In[8]:


pomdp = gridworld(m,n,p_obstacle_locations,obstacle_location,horizon,p,qr_n,qr_a,ql_n,ql_a,r,rew)


# In[9]:


n_o = len(p_obstacle_locations)
o_dict = {p_obstacle_locations[i]:i for i in range(n_o)}

label_v = [()]*(m*n*n_o)

for s,o in pomdp.states[1]:
    x,y = s[0],s[1]
    if s == o:
        if s == goal_location:
            label_v[(n*x+y)*n_o + o_dict[o]] = ('a','b')
        else:
            label_v[(n*x+y)*n_o + o_dict[o]] = ('b',)
    else:
        if s == goal_location:
            label_v[(n*x+y)*n_o + o_dict[o]] = ('a',)


# In[10]:


labels = {t:deepcopy(label_v) for t in range(1,horizon+1)}


# In[11]:


prod = combine(A,pomdp,labels,thres)


# In[12]:


pickle.dump(prod,open("sequence","wb"))

