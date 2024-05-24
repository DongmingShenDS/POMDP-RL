import os

import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from Gazebo_GW_env import GzbGw
from tianshou.utils.net.common import Net
from tianshou.data import Batch


class FixDim(gym.ActionWrapper):
	def __init__(self, env):
		super().__init__(env)

	def action(self, action):
		return np.array([action])


def main():
	logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))  # TensorBoard is supported!

	env = GzbGw("../../DFA/reachAavoidW.txt", update_img=False, monitor=False)
	ts.env.DummyVectorEnv([lambda: env])
	# env = gym.make('CartPole-v0')
	lr, epoch, batch_size = 1e-4, 100, 512
	train_num, test_num = 1, 100
	gamma, n_step, target_freq = 0.999, 3, 500
	buffer_size = 1000000
	eps_train, eps_test = 0.1, 0.05
	step_per_epoch, step_per_collect = 10000, 10
	train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(train_num)])
	test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(test_num)])

	state_shape = env.observation_space.shape or env.observation_space.n
	action_shape = env.action_space.shape or env.action_space.n
	net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
	optim = torch.optim.Adam(net.parameters(), lr=lr)

	policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq, is_double=True)

	train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num),
										exploration_noise=True)
	train_collector = ts.data.Collector(policy, train_envs, ts.data.PrioritizedReplayBuffer(buffer_size, alpha=0.7, beta=0.5),
										exploration_noise=True)
	test_collector = ts.data.Collector(policy, test_envs,
									   exploration_noise=True)  # because DQN uses epsilon-greedy method

	# policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
	# train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
	# test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

	trainer = ts.trainer.offpolicy_trainer(
		policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
		test_num, batch_size, update_per_step=1 / step_per_collect,
		train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
		test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
		save_best_fn=save_best_fn,
		save_checkpoint_fn=save_checkpoint_fn,
		logger=logger)



	print(f'Finished training! Use {trainer["duration"]}')


if __name__ == '__main__':
	main()
