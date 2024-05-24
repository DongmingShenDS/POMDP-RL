import sys
import time

import cv2

sys.path.insert(1, "/LS_POMDP_LTLf_RL/Gazebo_GridWorld")
from Gazebo_GW_env import GzbGw
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv


vanilla_PPO = True
PPO_stack_frame = False
if vanilla_PPO:
	env = GzbGw("../../DFA/reachAavoidW.txt", img_obs=True, actions=3, random_loc=True)
if PPO_stack_frame:
	env = DummyVecEnv([lambda: GzbGw("../../DFA/reachAavoidW.txt")])
	env = VecFrameStack(env, n_stack=4)
episodes = 1024

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:  # not done:
		a = int(input("enter input: (0, 1, 2)"))
		if a in [0, 1, 2]:
			obs, reward, done, info = env.step(a)
			print(reward)
		else:
			continue
		if done:
			done = False
			obs = env.reset()
