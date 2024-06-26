import csv
import numpy as np
from stable_baselines3 import PPO, DQN
from copy import deepcopy
import os
from Gazebo_GW_env_exact import GzbGw
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
	moni = input("monitor T or F: ")  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	if moni not in ['T', 'F']:
		print("Stopping.")
		exit(0)
	have_monitor = True if moni == 'T' else False
	ENV_COUNT = 10
	random_loc = True
	TIMESTEPS = 10000  # length of each episode
	SAVE_FREQ = 4
	EVAL_FREQ = 200  # how many TIMESTEPS per evaluation
	TOTAL_SIM = 100
	FIELDS = ['model_path', 'success_rate', 'ep_len_mean', 'ep_rew_mean']
	CSV_PATH = models_dir + "/traj.csv"
	DFA_PATH = "../../DFA/orderABavoidW.txt"
	print("running specification {}".format(DFA_PATH))
	VIDEO_EPS = 0.1  # if success_rate > VIDEO_EPS, generate video
	VIDEO_TIMESTEP = 2048
	REMOVE_EPS = -1  # if success_rate < VIDEO_EPS, remove (delete) model
	if algorithm == "PPO":
		# training env
		# env = GzbGw(DFA_PATH, update_img=have_monitor, monitor=have_monitor)
		# env = DummyVecEnv(
		# 	[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))
		# 	 for _ in range(ENV_COUNT)]
		# )
		# env = VecNormalize(env, norm_obs=True, norm_reward=True)
		env = Monitor(GzbGw(DFA_PATH, update_img=have_monitor, monitor=have_monitor), logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))
		if img_obs:
			model = PPO(
				"MultiInputPolicy", env, gamma=0.999, batch_size=512, learning_rate=7.5 * 1e-4,
				verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
		else:
			model = PPO(
				"MlpPolicy", env, gamma=0.999, verbose=1,
				tensorboard_log=logdir, device='cuda:' + cuda_index
			)
	elif algorithm == "DQN":
		# training env
		env = GzbGw(DFA_PATH, update_img=have_monitor, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))
			 for _ in range(ENV_COUNT)]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		# eval env
		eval_env = GzbGw(DFA_PATH, update_img=have_monitor, monitor=have_monitor)
		eval_env = DummyVecEnv(
			[lambda: Monitor(eval_env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))
			 for _ in range(ENV_COUNT)]
		)
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
		if img_obs:
			model = DQN(
				"MultiInputPolicy", env, gamma=0.999,
				buffer_size=100000, learning_starts=10000, batch_size=512, learning_rate=2.5 * 1e-4,
				exploration_final_eps=0.1, verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
		else:
			model = DQN(
				"MlpPolicy", env, gamma=0.999,
				buffer_size=500000, learning_starts=50000, batch_size=512, learning_rate=2.5 * 1e-4,
				exploration_fraction=0.2, exploration_final_eps=0.05,
				verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
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
		# 		action, _state = model.predict(obs)
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
	# video
	# if success_rate > VIDEO_EPS:
	# 	print("Generating Videos")
	# 	if algorithm == "PPO":
	# 		# video env
	# 		vid_env = Monitor(GzbGw(
	# 			DFA_PATH, update_img=True, monitor=False),
	# 			logdir, info_keywords=("is_success", "final_DFA", "horizon", "total_reward")
	# 		)
	# 		vid_env = DummyVecEnv([lambda: vid_env])
	# 		vid_env = VecVideoRecorder(
	# 			vid_env, video_dir, record_video_trigger=lambda x: x == 0,
	# 			video_length=VIDEO_TIMESTEP, name_prefix=str(time_steps)
	# 		)
	# 	elif algorithm == "DQN":
	# 		# video env
	# 		vid_env = GzbGw(DFA_PATH, update_img=True, monitor=False)
	# 		vid_env = DummyVecEnv(
	# 			[lambda: Monitor(vid_env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
	# 		)
	# 		vid_env = VecNormalize(vid_env, norm_obs=True, norm_reward=True)
	# 		vid_env = VecVideoRecorder(
	# 			vid_env, video_dir, record_video_trigger=lambda x: x == 0,
	# 			video_length=VIDEO_TIMESTEP, name_prefix=str(time_steps)
	# 		)
	# 	obs = vid_env.reset()
	# 	if img_obs:
	# 		obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
	# 	for i in range(VIDEO_TIMESTEP):
	# 		action, _state = model.predict(obs)
	# 		obs, reward, done, info = vid_env.step(action)
	# 		if img_obs:
	# 			obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
	# 	vid_env.close()


if __name__ == '__main__':
	main()
