import sys
sys.path.insert(1, "/LS_POMDP_LTLf_RL/Gazebo_GridWorld")
from Gazebo_GW_env import GzbGw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
import os
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(logdir):
	os.makedirs(logdir)

vanilla_PPO = False
PPO_stack_frame = True
if vanilla_PPO:
	env = GzbGw("../DFA/reachAavoidW.txt")
if PPO_stack_frame:
	env = DummyVecEnv([lambda: GzbGw("../../DFA/reachAavoidW.txt")])
	env = VecFrameStack(env, n_stack=4)
env.reset()
episodes = 1024

vanilla_PPO = False
PPO_stack_frame = True
if vanilla_PPO:
	GzbGw("../../DFA/reachAavoidW.txt")
if PPO_stack_frame:
	env = DummyVecEnv([lambda: GzbGw("../../DFA/reachAavoidW.txt")])
	env = VecFrameStack(env, n_stack=4)
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda')
print(model.env.observation_space)