import argparse
import os
import pprint

import tianshou as ts
import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from Gazebo_GW_env import GzbGw

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='../../DFA/reachAavoidW.txt')
    parser.add_argument('--reward-threshold', type=float, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=500)  # 500
    parser.add_argument('--step-per-epoch', type=int, default=20000)  # 20000
    parser.add_argument('--step-per-collect', type=int, default=10)
    # parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='TS/logs')
    # parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=1)
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    env = GzbGw("../../DFA/reachAavoidW.txt", update_img=False, monitor=False)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(args.training_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: env for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net = Net(
        state_shape=args.state_shape, action_shape=args.action_shape, hidden_sizes=args.hidden_sizes, device=args.device
    ).to(args.device)

    # orthogonal initialization
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = ts.policy.DQNPolicy(
        net, optim, args.gamma, estimation_step=3, target_update_freq=500, is_double=True
    )

    # collector
    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.PrioritizedReplayBuffer(args.buffer_size, alpha=0.7, beta=0.5),
        exploration_noise=True
    )
    test_collector = ts.data.Collector(
        policy, test_envs,
        exploration_noise=True
    )

    # log
    log_path = os.path.join(args.logdir, "dqn")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }, ckpt_path
        )
        print(epoch, ckpt_path)
        return ckpt_path

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    # trainer
    eps_train = 0.1
    eps_test = 0.02
    trainer = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        args.epoch, args.step_per_epoch, args.step_per_collect,
        args.test_num, args.batch_size, update_per_step=1 / args.step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=args.resume,
        logger=logger
    )

    print(trainer)
    # for epoch, epoch_stat, info in trainer:
    #     print(f"Epoch: {epoch}")
    #     print(epoch_stat)
    #     print(info)
    # assert stop_fn(info["best_reward"])

    if __name__ == "__main__":
        env = GzbGw("../../DFA/reachAavoidW.txt", update_img=False, monitor=False)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


def test_dqn_resume(args=get_args()):
    args.resume = True
    test_dqn(args)


if __name__ == "__main__":
    test_dqn()
