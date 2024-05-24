import random
import gym
from gym import spaces
import numpy as np
import cv2
import time
import math
from itertools import chain, combinations


class DFA:
	def __init__(self, dfatxt_path):
		self.specification = ""  # LTLf specification as str
		self.atomics = []  # atomic proposition
		self.Q = []  # possible DFA states
		self.q0 = None  # DFA starting state
		self.n_qs = -1  # number of possible DFA states
		self.acc = []  # accepted DFA states
		self.T = {}  # DFA transitions
		self.irrecoverable = []  # irrecoverable DFA states (if reached, can never back to acc)
		with open(dfatxt_path, 'r') as file:
			file_contents = [line.rstrip() for line in file]
			length = len(file_contents)
		ptr_atomics, ptr_n_qs, ptr_Q, ptr_q0, ptr_acc, ptr_T = 0, 0, 0, 0, 0, 0
		for i in range(length):
			if i == 0:
				self.specification += file_contents[i]
			if "A.atomics" in file_contents[i]:
				ptr_atomics = i + 1
			if "A.n_qs:" in file_contents[i]:
				ptr_n_qs = i + 1
			if "A.Q:" in file_contents[i]:
				ptr_Q = i + 1
			if "A.q0:" in file_contents[i]:
				ptr_q0 = i + 1
			if "A.acc:" in file_contents[i]:
				ptr_acc = i + 1
			if "A.T:" in file_contents[i]:
				ptr_T = i + 1
		self.atomics = eval(file_contents[ptr_atomics])
		self.n_qs = int(file_contents[ptr_n_qs])
		self.Q = eval(file_contents[ptr_Q])
		self.q0 = str(file_contents[ptr_q0])
		self.acc = eval(file_contents[ptr_acc])
		self.T = eval(file_contents[ptr_T])
		self.atomics_powerset = power_set(self.atomics)
		# handle irreversible DFA states
		reachable = {}
		for q in self.Q:
			reachable[q] = []
		for q in self.Q:  # all DFA states
			for p in power_set(self.atomics):  # all atomics power set elements
				reachable[q].append(self.T.get((q, p)))
		for q in self.Q:
			irr_flag = True
			for accepted in self.acc:
				if accepted in reachable[q]:  # any accepted state can be reached from q (in all possible transitions)
					irr_flag = False
			if irr_flag:  # np accepted state can be reached from q in any transition
				self.irrecoverable.append(q)


def power_set(iterable):
	"""Return power set of some iterable in tuple format"""
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def right_turn(agent_theta):
	"""Right turn 15 degree"""
	agent_theta += math.pi / 12
	if agent_theta >= 2 * math.pi:
		agent_theta -= 2 * math.pi
	return agent_theta


def left_turn(agent_theta):
	"""Left turn 15 degree"""
	agent_theta -= math.pi / 12
	if agent_theta < 0:
		agent_theta += 2 * math.pi
	return agent_theta


def forward(agent_pos, agent_theta, step_size):
	"""Move forward in current direction"""
	dx = math.cos(agent_theta) * step_size
	dy = math.sin(agent_theta) * step_size
	new_agent_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
	return new_agent_pos


def reach_goal(agent_pos, goal, eps):
	"""If reach any goal (within eps distance of agent_pos)"""
	if math.dist(agent_pos, goal) <= eps:
		return True
	return False


def wall_bound(wall_actual, agent_r):
	"""Helper for walls: agent_pos in wall_bound => collision"""
	a, b, c, d = wall_actual  # actual wall coordinate in order top-left, top-right, bottom-left, bottom-right
	aa = (a[0] - agent_r, a[1] - agent_r)
	bb = (b[0] + agent_r, b[1] - agent_r)
	cc = (c[0] - agent_r, c[1] + agent_r)
	dd = (d[0] + agent_r, d[1] + agent_r)
	return aa, bb, cc, dd  # wall collision bound in order ...


def wall_near_range(wall_actual, near_radius):
	"""Helper for walls: agent_pos in wall_bound => collision"""
	a, b, c, d = wall_actual  # actual wall coordinate in order top-left, top-right, bottom-left, bottom-right
	aa = (a[0] - near_radius, a[1] - near_radius)
	bb = (b[0] + near_radius, b[1] - near_radius)
	cc = (c[0] - near_radius, c[1] + near_radius)
	dd = (d[0] + near_radius, d[1] + near_radius)
	return aa, bb, cc, dd  # wall collision bound in order ...


def collision_with_walls(wall_bb, agent_pos):
	"""Check if agent run into some wall"""
	ax, ay = agent_pos
	for j in range(len(wall_bb)):
		(x1, y1), (x2, y2) = wall_bb[j][0], wall_bb[j][-1]
		if x1 < ax < x2 and y1 < ay < y2:
			return True
	return False


class GzbGw(gym.Env):

	def __init__(self, dfa, img_obs=True, actions=3, random_loc=False, monitor=True):
		"""
		:param dfa: dfa txt file path
		:param img_obs: if using the whole image as part of the observation
		:param actions: number of actions (if 3, manually forward; if 2 auto-forward)
		:param random_loc: agent random starting location
		:param monitor: do we have a monitor (if on server, use False)
		"""
		super(GzbGw, self).__init__()
		# Define action and observation space: must be gym.spaces objects
		"Action Space"
		self.actions = actions
		self.action_space = spaces.Discrete(actions)
		"Observation Space"
		self.if_image_obs = img_obs
		if self.if_image_obs:
			self.observation_space = spaces.Dict(
				spaces={
					"vec": spaces.Box(low=0, high=600, shape=(4,), dtype=np.float32),
					"img": spaces.Box(low=0, high=255, shape=(600, 600, 3), dtype='uint8'),
				}
			)
		else:
			self.observation_space = spaces.Box(low=0, high=1200, shape=(104,), dtype=np.float32)
		"Handle LTLf Specifications"
		self.dfa = DFA(dfa)
		self.atomic_prop = tuple()
		self.dfa_state = ''
		self.reward = 0
		"Other"
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		self.random_initial_loc = random_loc
		self.metadata = {'render.modes': ['human', 'rgb_array']}
		self.monitor = monitor

	def display_img(self):
		"""display GW image"""
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		"Display Agent"
		agent_location_display = (round(self.agent_location[0]), round(self.agent_location[1]))
		cv2.circle(self.img, agent_location_display, self.A_RADIUS, self.A_COLOR, -1)
		forward_loc = forward(self.agent_location, self.agent_angle, 4 * self.STEP_SIZE)
		forward_loc_display = (round(forward_loc[0]), round(forward_loc[1]))
		cv2.arrowedLine(self.img, agent_location_display, forward_loc_display, self.A_COLOR, 4)
		"Display Objects"
		cv2.circle(self.img, self.position1, self.O_RADIIUS, self.COLOR1, -1)
		cv2.circle(self.img, self.position2, self.O_RADIIUS, self.COLOR2, -1)
		"Display Walls"
		for i in range(len(self.walls)):
			cv2.rectangle(self.img, self.walls[i][0], self.walls[i][-1], self.WALL_COLOR, -1)
		"display"
		if self.monitor:
			cv2.imshow('Gazebo_GridWorld', self.img)
			cv2.waitKey(1)

	def step(self, action):
		"""Handle transition of the env"""
		"Update time step & Check if finish"
		self.time_step += 1
		if self.time_step == self.HORIZON:
			self.done = True
			self.success = False
			self.reward = -self.HORIZON
		"Takes step after fixed time"
		t_end = time.time() + 0
		k = -1
		while time.time() < t_end:
			if k == -1:
				k = cv2.waitKey(1)
			else:
				continue
		"Actions: 0=Left15, 1=Right15, 2=Forward, ?3=Stay"
		# Do action
		if self.actions == 3:
			new_agent_location = self.agent_location
			if action == 0:
				self.agent_angle = left_turn(self.agent_angle)
			elif action == 1:
				self.agent_angle = right_turn(self.agent_angle)
			elif action == 2:
				new_agent_location = forward(self.agent_location, self.agent_angle, self.STEP_SIZE)
		elif self.actions == 2:
			if action == 0:
				self.agent_angle = left_turn(self.agent_angle)
			elif action == 1:
				self.agent_angle = right_turn(self.agent_angle)
			new_agent_location = forward(self.agent_location, self.agent_angle, self.STEP_SIZE / 2)
		"Check collision with walls"
		w_collide, a_reach = False, False
		if collision_with_walls(self.wall_bounds, new_agent_location):
			w_collide = True  # if collide with wall, don't update agent location (not allowed transition)
		else:  # if not collide with wall, update agent location
			self.agent_location = new_agent_location
		"Check if near walls"
		if collision_with_walls(self.wall_near_range, self.agent_location):
			self.near_wall_flag = 1.0
		else:
			self.near_wall_flag = 0.0
		"Check if reaching any goals"
		if reach_goal(new_agent_location, self.position1, 5 * self.A_RADIUS):
			a_reach = True
		if reach_goal(new_agent_location, self.position2, 5 * self.A_RADIUS):
			a_reach = True
		"Update DFA stuff"
		original_dfa_state = self.dfa_state
		if w_collide:  # if collide with wall, don't update agent location (not allowed transition)
			self.atomic_prop += tuple('w')
		if a_reach:
			self.atomic_prop += tuple('a')
		self.atomic_prop = tuple(sorted(self.atomic_prop))
		new_dfa_state = self.dfa.T[(self.dfa_state, self.atomic_prop)]  # get new dfa state from A.T
		self.atomic_prop = tuple()  # reset atomic prop
		if original_dfa_state != new_dfa_state:
			print("DFA STATE changed from {} to {}".format(original_dfa_state, new_dfa_state))
			self.dfa_state = new_dfa_state  # update dfa state
		"Rewards according to DFA"  # TODO
		if not self.done:
			if self.dfa_state in self.dfa.acc:  # specification satisfied, goal reached
				self.reward = self.HORIZON
				self.total_reward += self.reward
				print("Reached accepted DFA: {}. STOPPING. TotalReward={}".format(self.dfa_state, self.total_reward))
				self.done = True  # goal-pomdp setting
				self.success = True
			elif self.dfa_state in self.dfa.irrecoverable:  # not possible to satisfy specification, goal not possible
				self.reward = -self.HORIZON
				self.total_reward += self.reward
				print("Reached irrecoverable DFA: {}. STOPPING. TotalReward={}".format(self.dfa_state, self.total_reward))
				self.done = True
				self.success = False
			else:  # specification not satisfied
				self.reward = -1
				"A* Reward according to SOME distance"
				# need to change for other DFAs
				self.dist_reward = (1 - min(math.dist(self.agent_location, self.position1),
											math.dist(self.agent_location, self.position2)) / self.MAX_DIST) ** 2
				self.reward += self.dist_reward
				self.total_reward += self.reward
		else:
			self.total_reward += self.reward
			print("HORIZON REACHED. STOPPING. TotalReward={}".format(self.total_reward))
		"Display the image after the update"
		self.display_img()
		"Observations"
		ax, ay = self.agent_location
		theta = self.agent_angle
		near_wall = self.near_wall_flag
		if self.if_image_obs:
			self.observation = {
				"vec": np.array([ax, ay, theta, near_wall], dtype=np.float32),
				"img": self.img
			}
		else:
			p1x, p1y = self.position1
			p2x, p2y = self.position2
			walls = np.array(self.walls).flatten()
			arr = np.concatenate((np.array([ax, ay, theta, near_wall, p1x, p1y, p2x, p2y]), walls), axis=None)
			self.observation = np.array(arr, dtype=np.float32)
		"Info"
		info = {}
		if self.done:
			info["is_success"] = self.success
			info["final_DFA"] = self.dfa_state
			info["horizon"] = self.time_step
			info["total_reward"] = self.total_reward
		return self.observation, self.reward, self.done, info

	def reset(self):
		"""Initialization for the actual environment"""
		"Important Constants"
		self.success = False
		self.done = False
		self.time_step = 0
		self.reward = 0
		self.dist_reward = 0
		self.total_reward = 0  # assume discount=1
		self.HORIZON = 1024
		self.EDGE_LEN = 500
		self.WALL_LEN = 100
		self.WIDTH = 15
		self.A_RADIUS = 10  # agent radius
		self.O_RADIIUS = 12  # object radius
		self.STEP_SIZE = 4
		self.MAX_DIST = math.sqrt(2) * self.EDGE_LEN
		"Reset DFA stuff"
		self.atomic_prop = tuple()  # empty atomic prop at beginning
		self.dfa_state = self.dfa.q0  # dfa state should be q0 = '1'
		"Walls & Edges"
		self.walls = []
		# edges
		(x, y) = (50, 50)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.EDGE_LEN), (x + self.WIDTH, y + self.EDGE_LEN)])
		self.walls.append([(x, y), (x + self.EDGE_LEN, y), (x, y + self.WIDTH), (x + self.EDGE_LEN, y + self.WIDTH)])
		(x, y) = (550, 550)
		self.walls.append([(x - self.WIDTH, y - self.EDGE_LEN), (x, y - self.EDGE_LEN), (x - self.WIDTH, y), (x, y)])
		self.walls.append([(x - self.EDGE_LEN, y - self.WIDTH), (x, y - self.WIDTH), (x - self.EDGE_LEN, y), (x, y)])
		# horizontal walls
		(x, y) = (50, 160)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		(x, y) = (350, 170)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		(x, y) = (210, 280)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		(x, y) = (440, 320)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		(x, y) = (240, 445)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		# vertical walls
		(x, y) = (245, 60)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		(x, y) = (130, 360)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		(x, y) = (440, 445)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		"Init Image"
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		"Initial Positions & COLORS"
		self.agent_angle = 0  # up=0, clockwise, radian, 15deg=math.pi/12
		self.button_direction = 2
		self.position1 = (100, 100)
		self.COLOR1 = (127, 127, 127)
		self.position2 = (500, 500)
		self.COLOR2 = (0, 255, 255)
		self.A_COLOR = (255, 255, 255)
		self.WALL_COLOR = (58, 98, 128)
		"Wall Bounds"
		# if A_loc inside "Wall Bounds" => collide
		self.wall_bounds = []
		for i in range(len(self.walls)):
			self.wall_bounds.append(wall_bound(self.walls[i], self.A_RADIUS))
		"Wall Near Range"
		# if A_loc inside "Wall Near Range" => near wall trigger
		self.wall_near_range = []
		for i in range(len(self.walls)):
			self.wall_near_range.append(wall_near_range(self.walls[i], 2 * self.A_RADIUS))
		self.near_wall_flag = 0.0
		"Agent Location Init"
		if self.random_initial_loc:
			while True:
				self.agent_location = (random.randint(100, 500), random.randint(100, 500))
				if not collision_with_walls(self.wall_near_range, self.agent_location) and \
					not reach_goal(self.agent_location, self.position1, 5 * self.A_RADIUS) and \
					not reach_goal(self.agent_location, self.position2, 5 * self.A_RADIUS):
					break
		else:
			self.agent_location = (400, 400)
		"Observations"
		self.display_img()
		ax, ay = self.agent_location
		theta = self.agent_angle
		near_wall = self.near_wall_flag
		if self.if_image_obs:
			self.observation = {
				"vec": np.array([ax, ay, theta, near_wall], dtype=np.float32),
				"img": self.img
			}
		else:
			p1x, p1y = self.position1
			p2x, p2y = self.position2
			walls = np.array(self.walls).flatten()
			arr = np.concatenate((np.array([ax, ay, theta, near_wall, p1x, p1y, p2x, p2y]), walls), axis=None)
			self.observation = np.array(arr, dtype=np.float32)
		return self.observation

	def render(self, mode='human'):
		"""pass"""
		# cv2.imshow('Gazebo_GridWorld', self.img)
		# cv2.waitKey(1)
		img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
		return img_rgb

	def close(self):
		"""pass"""
		...


def main():
	"""testing only"""
	g = GzbGw("../DFA/reachAavoidW.txt")
	g.reset()


if __name__ == '__main__':
	main()
