import csv

import numpy as np
from stable_baselines3 import PPO, DQN
from copy import deepcopy
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
	if algorithm not in ['PPO', 'DQN']:
		print("Algorithm Not Supported. Stopping.")
		exit(0)
	cuda_index = input("enter cuda index (0-9): ")
	model_path = input("enter model path without .zip: ")
	models_dir = f"{algorithm}/models/{int(time.time())}"
	logdir = f"{algorithm}/logs/{int(time.time())}"  # tensorboard --logdir=logs
	video_dir = models_dir + '/videos'
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	if not os.path.exists(video_dir):
		os.makedirs(video_dir)
	print(algorithm)
	env = None
	eval_env = None
	vid_env = None
	model = None
	img_obs = False
	have_monitor = True
	random_loc = True
	TIMESTEPS = 1024  # length of each episode
	SAVE_FREQ = 20
	EVAL_FREQ = 200  # how many TIMESTEPS per evaluation
	TOTAL_SIM = 100
	FIELDS = ['model_path', 'success_rate', 'ep_len_mean', 'ep_rew_mean']
	CSV_PATH = models_dir + "/traj.csv"
	VIDEO_EPS = 0.1  # if success_rate > VIDEO_EPS, generate video
	VIDEO_TIMESTEP = 2048
	REMOVE_EPS = -1  # if success_rate < VIDEO_EPS, remove (delete) model
	if algorithm == "PPO":
		# training env
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		# eval env
		eval_env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		eval_env = DummyVecEnv(
			[lambda: Monitor(eval_env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
		model = PPO.load(model_path + '.zip', env=env)
	elif algorithm == "DQN":
		# training env
		env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		# eval env
		eval_env = GzbGw("../../DFA/reachAavoidW.txt", update_img=have_monitor, monitor=have_monitor)
		eval_env = DummyVecEnv(
			[lambda: Monitor(eval_env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
		model = DQN.load(model_path + '.zip', env=env)
	env.reset()
	with open(CSV_PATH, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(FIELDS)
	eps = 0
	while eps < 500000:
		"Train"
		eps += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
		env.reset()
		if eps % SAVE_FREQ == 0:
			model_path = f"{models_dir}/{TIMESTEPS * eps}"
			model.save(model_path)
		print("finished episode {}".format(eps))
		"Eval & Video"
		# # evaluating policy
		# if eps % EVAL_FREQ != 0:
		# 	continue
		# time_steps = eps * TIMESTEPS
		# print("WORKING ON {}.".format(time_steps))
		# success_count = 0
		# ep_len = 0
		# ep_rew = 0
		# for i in range(TOTAL_SIM):
		# 	obs = eval_env.reset()
		# 	done = False
		# 	info = {}
		# 	step = 0
		# 	while step < 1024:
		# 		step += 1
		# 		action, _state = model.predict(obs, deterministic=False)
		# 		obs, reward, done, info = eval_env.step(action)
		# 	if algorithm == 'PPO':
		# 		ep_len += info[0]['horizon']
		# 		ep_rew += info[0]['total_reward']
		# 		success_count += 1 if info[0]['is_success'] else 0
		# 	elif algorithm == 'DQN':
		# 		ep_len += info[0]['horizon']
		# 		ep_rew += info[0]['total_reward']
		# 		success_count += 1 if info[0]['is_success'] else 0
		# success_rate = success_count / TOTAL_SIM
		# ep_len_mean = ep_len / TOTAL_SIM
		# ep_rew_mean = ep_rew / TOTAL_SIM
		# with open(CSV_PATH, 'a', newline='') as csv_file:
		# 	dict_object = csv.DictWriter(csv_file, fieldnames=FIELDS)
		# 	line = {
		# 		'model_path': model_path,
		# 		'success_rate': success_rate,
		# 		'ep_len_mean': ep_len_mean,
		# 		'ep_rew_mean': ep_rew_mean
		# 	}
		# 	dict_object.writerow(line)
		# # video
		# if success_rate > VIDEO_EPS:
		# 	print("Generating Videos")
		# 	if algorithm == "PPO":
		# 		# video env
		# 		vid_env = Monitor(GzbGw(
		# 			"../../DFA/reachAavoidW.txt", update_img=True, monitor=False),
		# 			logdir, info_keywords=("is_success", "final_DFA", "horizon", "total_reward")
		# 		)
		# 		vid_env = DummyVecEnv([lambda: vid_env])
		# 		vid_env = VecVideoRecorder(
		# 			vid_env, video_dir, record_video_trigger=lambda x: x == 0,
		# 			video_length=VIDEO_TIMESTEP, name_prefix=str(time_steps)
		# 		)
		# 	elif algorithm == "DQN":
		# 		# video env
		# 		vid_env = GzbGw("../../DFA/reachAavoidW.txt", update_img=True, monitor=False)
		# 		vid_env = DummyVecEnv(
		# 			[lambda: Monitor(vid_env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		# 		)
		# 		vid_env = VecNormalize(vid_env, norm_obs=True, norm_reward=True)
		# 		vid_env = VecVideoRecorder(
		# 			vid_env, video_dir, record_video_trigger=lambda x: x == 0,
		# 			video_length=VIDEO_TIMESTEP, name_prefix=str(time_steps)
		# 		)
		#
		# 	obs = vid_env.reset()
		# 	if img_obs:
		# 		obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
		# 	for i in range(VIDEO_TIMESTEP):
		# 		action, _state = model.predict(obs, deterministic=True)
		# 		obs, reward, done, info = vid_env.step(action)
		# 		if img_obs:
		# 			obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
		# 	vid_env.close()


if __name__ == '__main__':
	main()
	# PPO-important/8458240
# 8744960.zip
# 8458240.zip
# 10280960.zip
# 11304960.zip