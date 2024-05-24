import cv2
import numpy as np
from stable_baselines3 import PPO, DQN
import os
from Gazebo_GW_env import GzbGw
from typing import Any, Callable, Dict, Optional, Type, Union
import gym
import time
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO, QRDQN


def main():
	algorithm = "vanilla_DQN"  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	video_dir = f"videos/{int(time.time())}/"
	if not os.path.exists(video_dir):
		os.makedirs(video_dir)

	print(algorithm)
	if algorithm == "vanilla_DQN":  # suggested
		vid_len = 1000  # time_steps to record
		env = Monitor(GzbGw("../../DFA/reachAavoidW.txt", img_obs=False, random_loc=False, monitor=False),
					  info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))
		env = DummyVecEnv([lambda: env])
		env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda x: x == 0, video_length=vid_len,
							   name_prefix="x")
	model = DQN.load("tests/models/1660248606/25000.zip", env=env)

	obs = env.reset()
	# obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
	for i in range(vid_len):
		action, _state = model.predict(obs)
		obs, reward, done, info = env.step(action)
		# img = env.render()
		# cv2.imwrite(video_dir + f'/{i}.jpg', img)
		# cv2.waitKey(1)
		# obs['img'] = np.transpose(obs['img'], (0, 3, 1, 2))
	env.close()


if __name__ == '__main__':
	main()
