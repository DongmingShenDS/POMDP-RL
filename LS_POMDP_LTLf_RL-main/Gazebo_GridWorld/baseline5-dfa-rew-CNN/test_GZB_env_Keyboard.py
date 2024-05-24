import sys
import time
import cv2
from Gazebo_GW_env_approx import GzbGw
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv


vanilla_PPO = True
env = GzbGw("../../DFA/orderABavoidW.txt", actions=3, random_loc=True, monitor=True, update_img=True)
episodes = 2048

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:  # not done:
		a = input("enter input: (1, 2, 3)")
		if a in ['1', '2', '3']:
			obs, reward, done, info = env.step(int(a) - 1)
			print(reward)
		else:
			continue
		if done:
			done = False
			obs = env.reset()
