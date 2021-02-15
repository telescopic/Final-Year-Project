from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# from my_obs import obs
from flatland.envs.observations import TreeObsForRailEnv

from render_env import render_env

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from utils_observation_utils import normalize_observation
from utils_fast_tree_obs import FastTreeObs

from actor_critic import ActorCriticAgentWithPPO

from network import Network

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np

from matplotlib import pyplot as plt
import os

import wandb

if __name__ == '__main__':
	# Environment parameters
	# small_v0
	n_agents = 5
	x_dim = 30
	y_dim = 30
	n_cities = 2
	max_rails_between_cities = 2
	max_rails_in_city = 3
	seed = 42
	use_fast_tree_obs = False

	# Observation parameters
	observation_tree_depth = 2
	observation_radius = 10
	observation_max_path_depth = 30

	# Set the seeds
	random.seed(seed)
	np.random.seed(seed)

	# Break agents from time to time
	malfunction_parameters = MalfunctionParameters(
		malfunction_rate=1. / 10000,  # Rate of malfunctions
		min_duration=15,  # Minimal duration
		max_duration=50  # Max duration
	)

	# Observation builder
	predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
	tree_observation = None
	
	if use_fast_tree_obs:
		tree_observation = FastTreeObs(max_depth=observation_tree_depth)
		print("Using FastTreeObs")
	else:
		tree_observation = TreeObsForRailEnv(
			max_depth=observation_tree_depth, predictor=predictor)
		print("Using StandardTreeObs")

	speed_profiles = {
		1.: 1.0,  # Fast passenger train
		1. / 2.: 0.0,  # Fast freight train
		1. / 3.: 0.0,  # Slow commuter train
		1. / 4.: 0.0  # Slow freight train
	}

	env = RailEnv(
		width=x_dim,
		height=y_dim,
		rail_generator=sparse_rail_generator(
			max_num_cities=n_cities,
			grid_mode=False,
			max_rails_between_cities=max_rails_between_cities,
			max_rails_in_city=max_rails_in_city),
		schedule_generator=sparse_schedule_generator(speed_profiles),
		number_of_agents=n_agents,
		malfunction_generator_and_process_data=malfunction_from_params(
			malfunction_parameters),
		obs_builder_object=tree_observation,
		random_seed=seed)
	rewards = []
	obs, info = env.reset()

	if use_fast_tree_obs:
		state_size = tree_observation.observation_dim
	else:
		# Calculate the state size given the depth of the tree observation and the
		# number of features
		n_features_per_node = env.obs_builder.observation_dim
		n_nodes = 0
		for i in range(observation_tree_depth + 1):
			n_nodes += np.power(4, i)

		state_size = n_features_per_node * n_nodes

	agent_obs = [None] * env.get_num_agents()

	max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
	step_size = 100  # to print progress
	rewards_step_size = 10

	### CHANGE THIS VARIABLE
	checkpoint_steps = 20 # save model every 100 episodes
	num_episodes = 1

	hyperparams = {
		'actor_lr': 0.005,
		'critic_lr': 0.001,
		'clip': 0.2,
		'gamma': 0.95,
		'num_updates': 30
	}

	A2C_agent = Network(in_dim=state_size, out_dim=5)
	A2C_agent.load_state_dict(torch.load('actor.pth'))

	rendered_frames = []
	first_frame = None
	acc_rewards = []
	cntr = 0

	for i in range(num_episodes):

		print("Episode: " + str(i))

		cntr += 1

		obs, info = env.reset(True, True)


		for i in range(max_steps):
			render_env(env, 'Render/render'+str(i)+".png")

			# if i % step_size == 0:
			# 	print("=", end="", flush=True)

			for agent in env.get_agent_handles():
				if obs[agent]:
					agent_obs[agent] = normalize_observation(
						obs[agent],
						observation_tree_depth,
						observation_radius=observation_radius)
				else:
					agent_obs[agent] = None

			action_dict = {}

			for agent in env.get_agent_handles():
				if agent_obs[agent] is None:
					continue
				else:
					action_dict[agent] = np.random.choice([0, 1, 2, 3, 4, 5])
					# out = F.softmax(A2C_agent(agent_obs[agent]), dim=-1)
					# action = np.argmax(out.detach().numpy())
					# action_dict[agent] = action

			if agent_obs[0] is None:
				action_dict[1] = 2
			elif agent_obs[1] is None:
				action_dict[0] = 2

			next_obs, all_rewards, done, info = env.step(action_dict)

			obs = next_obs
