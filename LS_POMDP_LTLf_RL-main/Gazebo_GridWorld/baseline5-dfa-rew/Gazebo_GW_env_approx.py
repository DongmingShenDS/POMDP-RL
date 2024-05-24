import random
from copy import deepcopy
import gym
from gym import spaces
import numpy as np
import cv2
import time
import math
from itertools import chain, combinations
from stable_baselines3.common.env_checker import check_env
import shapely
from shapely.geometry import LineString, Point, Polygon


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
		self.intermediate = []  # intermediate DFA states (if reached, can get to acc), exclude q0
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
			reachable[q] = set()
		for q in self.Q:  # all DFA states
			for p in power_set(self.atomics):  # all atomics power set elements
				reachable[q].add(self.T.get((q, p)))
		# print(reachable)
		for q in self.Q:
			irr_flag = True
			for accepted in self.acc:
				if accepted in reachable[q]:  # any accepted state can be reached from q (in all possible transitions)
					irr_flag = False
			if irr_flag and q not in self.irrecoverable:  # np accepted state can be reached from q in any transition
				self.irrecoverable.append(q)
		# print(self.irrecoverable)
		for q in self.Q:
			if q not in self.irrecoverable and q not in self.intermediate and q != self.q0:
				self.intermediate.append(q)
		# print(self.intermediate)


def power_set(iterable):
	"""Return power set of some iterable in tuple format"""
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def right_turn(agent_theta):
	"""Right turn 15 degree"""
	agent_theta += math.pi / 12
	return agent_theta % (2 * math.pi)


def left_turn(agent_theta):
	"""Left turn 15 degree"""
	agent_theta -= math.pi / 12
	return agent_theta % (2 * math.pi)


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


def lidar_ray(agent_pos, ray_orientation, ray_len):
	"""one lidar ray (return start pt and end pt)"""
	dx = math.cos(ray_orientation) * ray_len
	dy = math.sin(ray_orientation) * ray_len
	end_pt = (agent_pos[0] + dx, agent_pos[1] + dy)
	return agent_pos, end_pt


def get_lidar_rays(agent_pos, a_orientation, ray_len, num_rays=8):
	"""get all lidar rays"""
	d_theta = 2 * math.pi / num_rays
	rays = []
	for i in range(num_rays):
		ray_orientation = (a_orientation + d_theta * i) % (2 * math.pi)
		rays.append(lidar_ray(agent_pos, ray_orientation, ray_len))
	return rays


def get_wall_edges(wall):
	"""return 4 line segments of a wall"""
	line1 = [wall[0], wall[1]]
	line2 = [wall[1], wall[3]]
	line3 = [wall[3], wall[2]]
	line4 = [wall[2], wall[0]]
	return line1, line2, line3, line4


def get_object_bbox(object_loc, object_radius):
	"""return 4 line segments of a bbox of an square object"""
	pt1 = [(object_loc[0]), (object_loc[1] - object_radius)]
	pt2 = [(object_loc[0] + object_radius), (object_loc[1])]
	pt3 = [(object_loc[0]), (object_loc[1] + object_radius)]
	pt4 = [(object_loc[0] + object_radius), (object_loc[1])]
	line1 = [pt1, pt3]
	line2 = [pt2, pt4]
	return line1, line2


def ray_intersect_shape(ray, shape):
	"""ray = 2 pts, shape = 4 lines"""
	min_dist = 10000
	intersection = None
	for line in shape:
		if not if_intersect(ray, line):
			continue
		point_of_intersection = line_intersection(ray[0], ray[1], line[0], line[1])
		if point_of_intersection is None:
			continue
		dist = math.dist(point_of_intersection, ray[0])
		if dist < min_dist:
			min_dist = dist
			intersection = point_of_intersection
	return intersection, min_dist


def line_intersection(a, b, c, d):
	line1 = LineString([a, b])
	line2 = LineString([c, d])
	int_pt = line1.intersection(line2)
	if int_pt.is_empty:
		point_of_intersection = None
	elif isinstance(int_pt, Point):
		point_of_intersection = np.array([int_pt.x, int_pt.y])
	elif isinstance(int_pt, LineString):
		point_of_intersection = None
	else:
		point_of_intersection = None
	return point_of_intersection


def if_intersect(ray, line):
	"""ray can be any line segments, line can ONLY be horizontal/vertical"""
	if line[0][0] == line[1][0]:  # horizontal
		return ray[0][0] <= line[0][0] <= ray[1][0] or ray[1][0] <= line[0][0] <= ray[0][0]
	elif line[0][1] == line[1][1]:  # vertical
		return ray[0][1] <= line[0][1] <= ray[1][1] or ray[1][1] <= line[1][1] <= ray[0][1]


class GzbGw(gym.Env):

	def __init__(self, dfa, actions=3, random_loc=True, monitor=False, update_img=False, horizon=1024, rays=8):
		"""
		:param dfa: dfa txt file path
		:param actions: number of actions (if 3, manually forward; if 2 auto-forward)
		:param random_loc: agent random starting location
		:param monitor: do we have a monitor (if on server, use False)
		"""
		super(GzbGw, self).__init__()
		# Define action and observation space: must be gym.spaces objects
		self.RAY_COUNT = rays
		"Action Space"
		self.actions = actions
		self.action_space = spaces.Discrete(actions)
		"Observation Space"
		self.observation_space = spaces.Box(low=0, high=1200, shape=(self.RAY_COUNT * 4 + 5,), dtype=np.float32)
		"Handle LTLf Specifications"
		self.dfa_path = dfa
		self.dfa = DFA(dfa)
		self.atomic_prop = tuple()
		self.dfa_state = ''
		self.reward = 0
		"Other"
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		self.random_initial_loc = random_loc
		self.metadata = {'render.modes': ['human', 'rgb_array']}
		self.monitor = monitor
		self.update_img = update_img
		self.horizon = horizon
		self.change_dfa_rew = 100

	def display_img(self):
		if not self.update_img:
			return
		"""display GW image"""
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		"Display Agent"
		agent_location_display = (round(self.agent_location[0]), round(self.agent_location[1]))
		cv2.circle(self.img, agent_location_display, self.A_RADIUS, self.A_COLOR, -1)
		forward_loc = forward(self.agent_location, self.agent_angle, 4 * self.STEP_SIZE)
		forward_loc_display = (round(forward_loc[0]), round(forward_loc[1]))
		cv2.arrowedLine(self.img, agent_location_display, forward_loc_display, self.A_COLOR, 4)
		"Display Objects"
		position1_display = (round(self.position1[0]), round(self.position1[1]))
		position2_display = (round(self.position2[0]), round(self.position2[1]))
		cv2.circle(self.img, position1_display, self.O_RADIUS, self.COLOR1, -1)
		cv2.circle(self.img, position2_display, self.O_RADIUS, self.COLOR2, -1)
		"Display Walls"
		for i in range(len(self.walls)):
			cv2.rectangle(self.img, self.walls[i][0], self.walls[i][-1], self.WALL_COLOR, -1)
		"Display Actual Rays"
		for i in range(len(self.actual_rays)):
			pt1 = tuple([int(round(x)) if isinstance(x, float) else x for x in self.actual_rays[i][0]])
			pt2 = tuple([int(round(x)) if isinstance(x, float) else x for x in self.actual_rays[i][1]])
			if self.detections[i] == [0, 0, 1]:
				cv2.line(self.img, pt1, pt2, self.A_COLOR, 1)
			else:
				cv2.line(self.img, pt1, pt2, self.RED, 1)
		"display"
		if self.monitor:
			cv2.imshow('Gazebo_GridWorld', self.img)
			cv2.waitKey(1)

	def step(self, action):
		"""Handle transition of the env"""
		self.reward = 0
		"Update time step & Check if finish"
		self.time_step += 1
		if self.time_step == self.horizon:
			self.done = True
		self.success = False
		"Takes step after fixed time"
		if self.monitor & self.update_img:
			t_end = time.time() + 0
			k = -1
			while time.time() < t_end:
				if k == -1:
					k = cv2.waitKey(1)
				else:
					continue
		"Object Move"
		# object 1: stay in the upper half
		a1 = random.randint(0, 1)
		if a1 == 0:
			self.p1_angle = left_turn(2 * self.p1_angle)
		elif a1 == 1:
			self.p1_angle = right_turn(2 * self.p1_angle)
		new_p1_location = forward(self.position1, self.p1_angle, self.STEP_SIZE / 4)
		if not collision_with_walls(self.wall_bounds, new_p1_location) and new_p1_location[1] < 600 / 2:
			self.position1 = new_p1_location
		# object 2: stay in the upper half
		a2 = random.randint(0, 1)
		if a2 == 0:
			self.p2_angle = left_turn(2 * self.p2_angle)
		elif a2 == 1:
			self.p2_angle = right_turn(2 * self.p2_angle)
		new_p2_location = forward(self.position2, self.p2_angle, self.STEP_SIZE / 4)
		if not collision_with_walls(self.wall_bounds, new_p2_location) and new_p2_location[1] > 600 / 2:
			self.position2 = new_p2_location
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
		w_collide, a_reach, b_reach = False, False, False
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
			b_reach = True
		"Update DFA stuff"
		original_dfa_state = self.dfa_state
		if w_collide:  # if collide with wall, don't update agent location (not allowed transition)
			self.atomic_prop += tuple('w')
		if self.dfa_path == "../../DFA/orderABavoidW.txt":
			if a_reach:
				self.atomic_prop += tuple('a')
			if b_reach:
				self.atomic_prop += tuple('b')
		elif self.dfa_path == "../../DFA/reachAavoidW.txt":
			if a_reach or b_reach:
				self.atomic_prop += tuple('a')
		self.atomic_prop = tuple(sorted(self.atomic_prop))
		new_dfa_state = self.dfa.T[(self.dfa_state, self.atomic_prop)]  # get new dfa state from A.T
		self.atomic_prop = tuple()  # reset atomic prop
		if original_dfa_state != new_dfa_state:  # state change
			print("DFA changed from {} to {} at timestep {}".format(original_dfa_state, new_dfa_state, self.time_step))
			self.dfa_state = new_dfa_state  # update dfa state
			if self.dfa_state in self.dfa.intermediate:  # state change to intermediate
				self.reward += self.change_dfa_rew
		"Rewards according to DFA"
		if self.dfa_state in self.dfa.acc:  # specification satisfied, goal reached
			self.reward += 1
			self.total_reward += self.reward
			self.success = True
		elif self.dfa_state in self.dfa.irrecoverable:  # not possible to satisfy specification, goal not possible
			self.reward += 0
			self.total_reward += self.reward
			self.done = True  # early stopping if reach irrecoverable
		else:  # specification not satisfied
			self.reward += 0
			self.total_reward += self.reward
		"Lidar Rays"
		self.lidar_rays = get_lidar_rays(self.agent_location, self.agent_angle, self.RAY_LEN, self.RAY_COUNT)
		self.actual_rays = deepcopy(self.lidar_rays)
		self.detections = [[0, 0, 1]] * self.RAY_COUNT  # walls=[0, 0, 1], o1=[0, 1, 0], 02=[1, 0, 0]
		self.lidar_distance = [self.RAY_LEN] * self.RAY_COUNT
		for i, ray in enumerate(self.lidar_rays):
			for line in self.wall_lines:
				if not if_intersect(self.lidar_rays[i], line):
					continue
				point_of_intersection = line_intersection(ray[0], ray[1], line[0], line[1])
				if point_of_intersection is None:
					continue
				dist = math.dist(point_of_intersection, ray[0])
				if dist < self.lidar_distance[i]:
					intersection = point_of_intersection
					self.lidar_rays[i] = ray[0], intersection
					self.actual_rays[i] = ray[0], intersection
					self.lidar_distance[i] = dist
			for j, shape in enumerate(self.object_bbox):
				intersection, min_dist = ray_intersect_shape(self.lidar_rays[i], shape)
				if intersection is None:
					continue
				if min_dist < self.lidar_distance[i]:
					self.lidar_rays[i] = ray[0], intersection
					self.actual_rays[i] = ray[0], intersection
					self.lidar_distance[i] = min_dist
					if j == 0:
						self.detections[i] = [0, 1, 0]
					if j == 1:
						self.detections[i] = [1, 0, 0]
		"Display Image"
		self.display_img()
		"Observations"
		ax, ay = self.agent_location
		theta = self.agent_angle
		near_wall = self.near_wall_flag
		dfa = self.dfa_state
		lidar_distance = np.array(self.lidar_distance, dtype=np.float32).flatten()
		detections = np.array(self.detections, dtype=np.float32).flatten()
		others = np.array([ax, ay, theta, near_wall, dfa], dtype=np.float32).flatten()
		self.observation = np.concatenate((others, lidar_distance, detections), axis=None)
		"Info"
		info = {
			"is_success": self.success,
			"final_DFA": self.dfa_state,
			"horizon": self.time_step,
			"total_reward": self.total_reward
		}
		return self.observation, self.reward, self.done, info

	def reset(self):
		"""Initialization for the actual environment"""
		"Important Constants"
		self.success = False
		self.done = False
		self.time_step = 0
		self.reward = 0
		self.total_reward = 0  # assume discount=1
		self.HORIZON = 1024
		self.EDGE_LEN = 500
		self.WALL_LEN = 100
		self.WIDTH = 15
		self.A_RADIUS = 10  # agent radius
		self.O_RADIUS = 12  # object radius
		self.STEP_SIZE = 4
		self.RAY_LEN = math.sqrt(2) * self.EDGE_LEN
		"Reset DFA stuff"
		self.atomic_prop = tuple()  # empty atomic prop at beginning
		self.dfa_state = self.dfa.q0  # dfa state should be q0 = '1'
		"Walls & Edges"
		self.walls = []
		self.wall_lines = []
		# edges
		(x, y) = (50, 50)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.EDGE_LEN), (x + self.WIDTH, y + self.EDGE_LEN)])
		self.walls.append([(x, y), (x + self.EDGE_LEN, y), (x, y + self.WIDTH), (x + self.EDGE_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x + self.WIDTH, y + self.EDGE_LEN), (x + self.WIDTH, y)))
		self.wall_lines.append(((x, y + self.WIDTH), (x + self.EDGE_LEN, y + self.WIDTH)))
		(x, y) = (550, 550)
		self.walls.append([(x - self.WIDTH, y - self.EDGE_LEN), (x, y - self.EDGE_LEN), (x - self.WIDTH, y), (x, y)])
		self.walls.append([(x - self.EDGE_LEN, y - self.WIDTH), (x, y - self.WIDTH), (x - self.EDGE_LEN, y), (x, y)])
		self.wall_lines.append(((x - self.WIDTH, y - self.EDGE_LEN), (x - self.WIDTH, y)))
		self.wall_lines.append(((x, y - self.WIDTH), (x - self.EDGE_LEN, y - self.WIDTH)))
		# horizontal walls
		(x, y) = (50, 160)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x, y + self.WIDTH / 2), (x + self.WALL_LEN, y + self.WIDTH / 2)))
		(x, y) = (350, 170)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x, y + self.WIDTH / 2), (x + self.WALL_LEN, y + self.WIDTH / 2)))
		(x, y) = (210, 280)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x, y + self.WIDTH / 2), (x + self.WALL_LEN, y + self.WIDTH / 2)))
		(x, y) = (440, 320)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x, y + self.WIDTH / 2), (x + self.WALL_LEN, y + self.WIDTH / 2)))
		(x, y) = (240, 445)
		self.walls.append([(x, y), (x + self.WALL_LEN, y), (x, y + self.WIDTH), (x + self.WALL_LEN, y + self.WIDTH)])
		self.wall_lines.append(((x, y + self.WIDTH / 2), (x + self.WALL_LEN, y + self.WIDTH / 2)))
		# vertical walls
		(x, y) = (245, 60)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		self.wall_lines.append(((x + self.WIDTH / 2, y), (x + self.WIDTH / 2, y + self.WALL_LEN)))
		(x, y) = (130, 360)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		self.wall_lines.append(((x + self.WIDTH / 2, y), (x + self.WIDTH / 2, y + self.WALL_LEN)))
		(x, y) = (440, 445)
		self.walls.append([(x, y), (x + self.WIDTH, y), (x, y + self.WALL_LEN), (x + self.WIDTH, y + self.WALL_LEN)])
		self.wall_lines.append(((x + self.WIDTH / 2, y), (x + self.WIDTH / 2, y + self.WALL_LEN)))
		"Init Image"
		self.img = np.zeros((600, 600, 3), dtype='uint8')
		"Initial Positions & COLORS"
		self.agent_angle = 0  # up=0, clockwise, radian, 15deg=math.pi/12
		self.button_direction = 2
		self.position1 = (100, 100)
		self.p1_angle = 0
		self.COLOR1 = (127, 127, 127)
		self.position2 = (500, 500)
		self.p2_angle = 0
		self.COLOR2 = (0, 255, 255)
		self.A_COLOR = (255, 255, 255)
		self.RED = (0, 0, 255)
		self.WALL_COLOR = (58, 98, 128)
		"Object BBox"
		self.object_bbox = []
		self.object_bbox.append(get_object_bbox(self.position1, self.O_RADIUS))
		self.object_bbox.append(get_object_bbox(self.position2, self.O_RADIUS))
		"Wall Bounds & Wall BBox Lines & Wall Near Range"
		# if A_loc inside "Wall Bounds" => collide
		# if A_loc inside "Wall Near Range" => near wall trigger
		self.wall_bounds = []
		self.wall_bbox = []
		self.wall_near_range = []
		for i in range(len(self.walls)):
			self.wall_bounds.append(wall_bound(self.walls[i], self.A_RADIUS))
			self.wall_bbox.append(get_wall_edges(self.walls[i]))
			self.wall_near_range.append(wall_near_range(self.walls[i], 2 * self.A_RADIUS))
		self.near_wall_flag = 0.0
		"Agent Location Init"
		if self.random_initial_loc:
			while True:
				self.agent_location = (random.randint(200, 400), random.randint(200, 400))
				if not collision_with_walls(self.wall_near_range, self.agent_location) and \
						not reach_goal(self.agent_location, self.position1, 5 * self.A_RADIUS) and \
						not reach_goal(self.agent_location, self.position2, 5 * self.A_RADIUS):
					break
		else:
			self.agent_location = (400, 400)
		"Lidar Rays"
		self.lidar_rays = get_lidar_rays(self.agent_location, self.agent_angle, self.RAY_LEN, self.RAY_COUNT)
		self.actual_rays = deepcopy(self.lidar_rays)
		self.detections = [[0, 0, 1]] * self.RAY_COUNT  # walls=[0, 0, 1], o1=[0, 1, 0], 02=[1, 0, 0]
		self.lidar_distance = [self.RAY_LEN] * self.RAY_COUNT
		for i, ray in enumerate(self.lidar_rays):
			intersection, min_dist = ray_intersect_shape(self.lidar_rays[i], self.wall_lines)
			if intersection is None:
				continue
			if min_dist < self.lidar_distance[i]:
				self.lidar_rays[i] = ray[0], intersection
				self.actual_rays[i] = ray[0], intersection
				self.lidar_distance[i] = min_dist
			for j, shape in enumerate(self.object_bbox):
				intersection, min_dist = ray_intersect_shape(self.lidar_rays[i], shape)
				if intersection is None:
					continue
				if min_dist < self.lidar_distance[i]:
					self.lidar_rays[i] = ray[0], intersection
					self.actual_rays[i] = ray[0], intersection
					self.lidar_distance[i] = min_dist
					if j == 0:
						self.detections[i] = [0, 1, 0]
					if j == 1:
						self.detections[i] = [1, 0, 0]
		"Display Image"
		self.display_img()
		"Observations"
		ax, ay = self.agent_location
		theta = self.agent_angle
		near_wall = self.near_wall_flag
		dfa = self.dfa_state
		lidar_distance = np.array(self.lidar_distance, dtype=np.float32).flatten()
		detections = np.array(self.detections, dtype=np.float32).flatten()
		others = np.array([ax, ay, theta, near_wall, dfa], dtype=np.float32).flatten()
		self.observation = np.concatenate((others, lidar_distance, detections), axis=None)
		return self.observation

	def render(self, mode='human'):
		"""pass"""
		img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
		return img_rgb

	def close(self):
		"""pass"""
		...


def main():
	"""testing only"""
	dfa = DFA("../../DFA/orderABavoidW.txt")
	exit(0)
	env = GzbGw("../../DFA/reachAavoidW.txt", update_img=True, monitor=True)
	check_env(env)
	exit(0)


if __name__ == '__main__':
	main()
