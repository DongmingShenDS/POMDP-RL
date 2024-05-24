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
	algorithm = input("enter algorithm name: ")  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	models_dir = input("enter model dir: ")  # models/1660710936
	if not os.path.exists(models_dir):
		print("model dir DNE. STOPPING.")
	logdir = models_dir.replace("models", "logs")
	video_dir = models_dir + '/videos_ALL'
	if not os.path.exists(video_dir):
		os.makedirs(video_dir)
	eps_len = int(input("enter episode length: "))
	total_eps = int(input("enter total episodes: "))
	print("algorithm: {}".format(algorithm))
	print("models_dir: {}".format(models_dir))
	print("logdir: {}".format(logdir))
	print("eps_len: {}".format(eps_len))  # 10000, 20000*k, 50000

	cuda_index = '0'
	TOTAL_SIM = 100
	FIELDS = ['model_path', 'success_rate', 'ep_len_mean', 'ep_rew_mean']
	CSV_PATH = models_dir + "CSV/traj.csv"
	VIDEO_EPS = -1  # if success_rate > VIDEO_EPS, generate video
	VIDEO_TIMESTEP = 2048
	REMOVE_EPS = -1  # if success_rate < VIDEO_EPS, remove (delete) model
	if os.path.exists(CSV_PATH):
		os.remove(CSV_PATH)
	with open(CSV_PATH, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(FIELDS)
	model = None
	env = None

	index = 0
	time_steps = 0
	while time_steps <= total_eps:
		index += 1
		time_steps = index * eps_len
		model_path = f"{models_dir}/{time_steps}.zip"
		if not os.path.exists(model_path):
			print("CANNOT FIND {}. SKIPPING.".format(model_path))
			continue
		print("WORKING ON {}.".format(model_path))
		"Init Env & Load model"
		image_observation = False
		have_monitor = False
		if algorithm == "PPO":
			env = Monitor(GzbGw(
				"../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor),
				logdir, info_keywords=("is_success", "final_DFA", "horizon", "total_reward")
			)
			model = PPO.load(model_path, env=env)
		elif algorithm == "DQN":  # suggested
			env = Monitor(
				GzbGw("../DFA/reachAavoidW.txt", img_obs=image_observation, actions=3, random_loc=False, monitor=False),
				logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward')
			)
			model = DQN(
				"MlpPolicy", env, gamma=0.999,
				buffer_size=1000000, learning_starts=50000, batch_size=512, learning_rate=7.5 * 1e-4,
				exploration_initial_eps=0,
				exploration_final_eps=0, verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
			model.set_parameters(model_path)
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
				action, _state = model.predict(obs)
				obs, reward, done, info = env.step(action)
			ep_len += info['horizon']
			ep_rew += info['total_reward']
			success_count += 1 if info['is_success'] else 0
		success_rate = success_count / TOTAL_SIM
		ep_len_mean = ep_len / TOTAL_SIM
		ep_rew_mean = ep_rew / TOTAL_SIM
		with open(CSV_PATH, 'a', newline='') as csv_file:
			dict_object = csv.DictWriter(csv_file, fieldnames=FIELDS)
			line = {
				'model_path': model_path,
				'success_rate': success_rate,
				'ep_len_mean': ep_len_mean,
				'ep_rew_mean': ep_rew_mean
			}
			dict_object.writerow(line)
		"Video Generation (if good policy)"
		if success_rate > VIDEO_EPS:
			env = DummyVecEnv([lambda: env])
			env = VecVideoRecorder(
				env, video_dir, record_video_trigger=lambda x: x == 0,
				video_length=VIDEO_TIMESTEP, name_prefix=str(time_steps)
			)
			if algorithm == "vanilla_PPO":
				model = PPO.load(model_path, env=env)
			elif algorithm == "vanilla_DQN":  # suggested
				model = DQN(
					"MlpPolicy", env, gamma=0.999,
					buffer_size=1000000, learning_starts=50000, batch_size=512, learning_rate=7.5 * 1e-4,
					exploration_initial_eps=0,
					exploration_final_eps=0, verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
				)
				model.set_parameters(model_path)
			else:
				exit(0)
			obs = env.reset()
			if image_observation:
				obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
			for i in range(VIDEO_TIMESTEP):
				action, _state = model.predict(obs)
				obs, reward, done, info = env.step(action)
				if image_observation:
					obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
			env.close()
		"Delete model (if bad policy)"
		if success_rate < REMOVE_EPS:
			os.remove(model_path)
		print("finish at policy {}".format(model_path))


if __name__ == '__main__':
	main()
