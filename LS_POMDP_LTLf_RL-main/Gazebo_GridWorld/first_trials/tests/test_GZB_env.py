import sys
import time

sys.path.insert(1, "/LS_POMDP_LTLf_RL/Gazebo_GridWorld")
from Gazebo_GW_env import GzbGw
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv


vanilla_PPO = True
PPO_stack_frame = False
if vanilla_PPO:
	env = GzbGw("../../DFA/reachAavoidW.txt", img_obs=False, random_loc=False, actions=3, monitor=True)
if PPO_stack_frame:
	env = DummyVecEnv([lambda: GzbGw("../../DFA/reachAavoidW.txt")])
	env = VecFrameStack(env, n_stack=4)
obs = env.reset()
episodes = 1024

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:  # not done:
		random_action = env.action_space.sample()
		obs, reward, done, info = env.step(random_action)
		# print(obs[0:2])
		# print("action", random_action)
		# if vanilla_PPO:
		# 	obs, reward, done, info = env.step(random_action)
		# 	# make sure the image is updating
		# 	print((sum(sum(obs['img']))))
		# if PPO_stack_frame:
		# 	obs, reward, done, info = env.step([random_action])
		# 	# make sure the observation is saving the last 4 frames
		# 	print(sum(sum(sum(obs['img']))))
		if done:
			done = False
			obs = env.reset()
