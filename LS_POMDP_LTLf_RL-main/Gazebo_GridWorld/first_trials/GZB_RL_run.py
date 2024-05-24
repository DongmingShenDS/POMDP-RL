from stable_baselines3 import PPO, DQN
import os
from Gazebo_GW_env import GzbGw
from typing import Any, Callable, Dict, Optional, Type, Union
import gym
import time
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO, QRDQN


def main():
	# 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
	algorithm = input("enter algorithm name: ")  # vanilla_PPO, PPO_stack_frame, vanilla_DQN
	cuda_index = input("enter cuda index (0-9): ")
	models_dir = f"models/{int(time.time())}-{algorithm}"
	logdir = f"logs/{int(time.time())}-{algorithm}"  # tensorboard --logdir=logs
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	print(algorithm)
	env = None
	model = None
	img_obs = False
	have_monitor = False
	random_loc = True
	if algorithm == "PPO":
		env = Monitor(GzbGw(
			"../DFA/reachAavoidW.txt", img_obs=img_obs, actions=3, random_loc=random_loc, monitor=have_monitor),
			logdir, info_keywords=("is_success", "final_DFA", "horizon", "total_reward")
		)
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
	# elif algorithm == "PPO_stack_frame":
	# 	env = DummyVecEnv([lambda: Monitor(GzbGw("../DFA/reachAavoidW.txt"), logdir, info_keywords=("is_success",))])
	# 	env = VecFrameStack(env, n_stack=4)
	# 	model = PPO("MultiInputPolicy", env, gamma=0.999,
	# 				verbose=1, tensorboard_log=logdir, device='cuda')
	elif algorithm == "DQN":
		# env = Monitor(GzbGw(
		# 	"../DFA/reachAavoidW.txt", img_obs=img_obs, actions=3, random_loc=random_loc, monitor=have_monitor),
		# 	logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward')
		# )
		env = GzbGw("../DFA/reachAavoidW.txt", img_obs=img_obs, actions=3, random_loc=random_loc, monitor=have_monitor)
		env = DummyVecEnv(
			[lambda: Monitor(env, logdir, info_keywords=("is_success", "final_DFA", "horizon", 'total_reward'))]
		)
		env = VecNormalize(env, norm_obs=True, norm_reward=True)
		if img_obs:
			model = DQN(
				"MultiInputPolicy", env, gamma=0.999,
				buffer_size=100000, learning_starts=10000, batch_size=512, learning_rate=7.5 * 1e-4,
				exploration_final_eps=0.1, verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
		else:
			model = DQN(
				"MlpPolicy", env, gamma=0.999,
				buffer_size=1000000, learning_starts=50000, batch_size=512, learning_rate=7.5 * 1e-4,
				exploration_final_eps=0.1, verbose=1, tensorboard_log=logdir, device='cuda:' + cuda_index
			)
	# elif algorithm == "multi_process_DQN":  # not working right now
	# 	num_cpu = 4
	# 	env = Monitor(GzbGw("../DFA/reachAavoidW.txt", img_obs=False, actions=2, random_loc=True, monitor=False),
	# 				  logdir, info_keywords=("is_success", "final_DFA", "horizon", 'reward'))
	# 	envs = [env for i in range(num_cpu)]
	# 	env = SubprocVecEnv(envs)
	else:
		print("Algorithm Not Supported. Stopping.")
		exit(0)
	env.reset()
	TIMESTEPS = 20000  # how frequently to save the model
	eps = 0
	while eps < 25000:
		eps += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algorithm}")
		model.save(f"{models_dir}/{TIMESTEPS * eps}")
		print("finished episode {}".format(eps))


if __name__ == '__main__':
	main()
