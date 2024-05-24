import sys
sys.path.insert(1, "/LS_POMDP_LTLf_RL/Gazebo_GridWorld")
from stable_baselines3 import PPO
import os
from Gazebo_GW_env import GzbGw
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(logdir):
	os.makedirs(logdir)

env = GzbGw("../../DFA/reachAavoidW.txt")
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda')
model1 = PPO.load("models/1660248606/100000.zip", env=env, verbose=1, tensorboard_log=logdir, device='cuda')
model2 = PPO.load("models/1660248606/500000.zip", env=env, verbose=1, tensorboard_log=logdir, device='cuda')
for k, v in model.policy.state_dict().items():
	print("model1" + k)
	print(model1.policy.state_dict()[k])
	print("model2" + k)
	print(model2.policy.state_dict()[k])
