import sys
sys.path.insert(1, "/LS_POMDP_LTLf_RL/Gazebo_GridWorld")
from stable_baselines3.common.env_checker import check_env
from Gazebo_GW_env import GzbGw

env = GzbGw("../../DFA/reachAavoidW.txt", img_obs=False, actions=2, random_loc=True)
# It will check your custom environment and output additional warnings if needed
check_env(env)
