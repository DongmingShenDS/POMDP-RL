import csv

import numpy as np
from stable_baselines3 import PPO, DQN
import os
from Gazebo_GW_env import GzbGw
from typing import Any, Callable, Dict, Optional, Type, Union
import gym
import time
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


# from sb3_contrib import RecurrentPPO, QRDQN


def main():
	algorithm = input("enter algorithm name: ")  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	models_dir = input("enter model path: ")  # PPO/models/1660858431
	if not os.path.exists(models_dir):
		print("model dir DNE. STOPPING.")
		exit(0)
	logdir = models_dir.replace("models", "logs")
	model_exact = input("enter exact model .zip: ")
	video_dir = models_dir + '/videos_generate'
	if not os.path.exists(video_dir):
		os.makedirs(video_dir)
	print("algorithm: {}".format(algorithm))
	print("models_dir: {}".format(models_dir))
	cuda_index = '0'
	VIDEO_TIMESTEP = 1024 * 100
	model = None
	env = None

	model_path = models_dir + '/' + model_exact
	if not os.path.exists(model_path):
		print("CANNOT FIND {}. SKIPPING.".format(model_path))
		exit(0)
	time_steps = model_exact.replace(".zip", "")
	print("WORKING ON {}.".format(model_path))
	"Video Generation"
	have_monitor = True
	if algorithm == "PPO":
		# training env
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		model = PPO.load(model_path, env=env)
	elif algorithm == "DQN":
		# training env
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		model = DQN.load(model_path, env=env)
	else:
		exit(0)
	obs = env.reset()
	for i in range(VIDEO_TIMESTEP):
		action, _state = model.predict(obs)
		obs, reward, done, info = env.step(action)
		if done:
			obs = env.reset()
	print("finish at policy {}".format(model_path))


if __name__ == '__main__':
	main()
# PPO
# PPO-important
# 8744960.zip
# 8458240.zip
# 10280960.zip
# 11304960.zip
