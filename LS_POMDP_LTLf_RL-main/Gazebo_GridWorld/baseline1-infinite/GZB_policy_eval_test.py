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
from sb3_contrib import RecurrentPPO, QRDQN


def main():
	# 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
	algorithm = "PPO"  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	models_dir = "PPO-important"  # models/1660710936
	if not os.path.exists(models_dir):
		print("model dir DNE. STOPPING.")
	logdir = models_dir + "/logs"
	print("algorithm: {}".format(algorithm))
	print("models_dir: {}".format(models_dir))
	print("logdir: {}".format(logdir))

	TOTAL_SIM = 100
	FIELDS = ['model_path', 'success_rate', 'ep_len_mean', 'ep_rew_mean']

	model = None
	env = None

	model_path = f"{models_dir}/8458240.zip"
	if not os.path.exists(model_path):
		print("CANNOT FIND {}. SKIPPING.".format(model_path))
		exit(0)
	print("WORKING ON {}.".format(model_path))
	"Init Env & Load model"
	have_monitor = True
	if algorithm == "PPO":
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		model = PPO.load(model_path, env=env)
	elif algorithm == "DQN":
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		model = DQN.load(model_path, env=env)
	else:
		print("Algorithm Not Supported. Stopping.")
		exit(0)
	"Policy Evaluation for TOTAL_SIM steps & write to CSV"
	success_count = 0
	ep_len = 0
	ep_rew = 0
	for i in range(TOTAL_SIM):
		obs = env.reset()
		done = False
		info = {}
		while not done:
			action, _state = model.predict(obs, deterministic=False)
			obs, reward, done, info = env.step(action)
			ep_rew += reward
		ep_len += info[0]['horizon']
		success_count += 1 if info[0]['is_success'] else 0
	success_rate = success_count / TOTAL_SIM
	ep_len_mean = ep_len / TOTAL_SIM
	ep_rew_mean = ep_rew / TOTAL_SIM
	print(success_rate, ep_len_mean, ep_rew_mean)
	print("finish at policy {}".format(model_path))


if __name__ == '__main__':
	main()
