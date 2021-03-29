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
#from utils_fast_tree_obs import FastTreeObs

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import time

from utils.buffer import ReplayBuffer
from algorithms.attention_sac import AttentionSAC

import wandb
wandb.init(project='flatland', name='maac-run-6.0-gcp')

## envs
env_params = [
        {
            # Test_0
            "n_agents": 1,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_0
            "n_agents": 2,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_0
            "n_agents": 3,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_0
            "n_agents": 4,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 5,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 20,
            "x_dim": 35,
            "y_dim": 35,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 58,
            "x_dim": 40,
            "y_dim": 40,
            "n_cities": 5,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]
##

def train(env):
	n_agents = 5
	x_dim = 50
	y_dim = 50
	n_cities = 4
	max_rails_between_cities = 2
	max_rails_in_city = 3
	seed = 0
	use_fast_tree_obs = False

	# Observation parameters
	observation_tree_depth = 4
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

	action_size = 5
	
	DEVICE = 'cpu'
	# if torch.cuda.is_available():
	# 	DEVICE = 'gpu'

	buffer_length = 10000
	steps_to_save_model = 10
	step_size = 100
	num_steps = 100 # update every 100 steps
	avg_steps = 20 # num steps to average and plot rewards
	reward_q = []
	batch_size = 100

	agent_obs = np.array([None] * env.get_num_agents())

	max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
	num_episodes = 100000

	agent_init_params = []
	sa_size = []

	for i in range(n_agents):
		agent_init_params.append({'num_in_pol': state_size, 'num_out_pol': action_size})
		sa_size.append((state_size, action_size))

	hyperparams = {
	"tau": 0.01, # ddpg soft update
	"pi_lr": 0.00001,
	"q_lr": 0.00005,
	"pol_hidden_dim": 256,
	"critic_hidden_dim": 256,
	"attend_heads": 8
	}

	model = AttentionSAC(
			agent_init_params=agent_init_params,
			sa_size=sa_size,
			tau=hyperparams["tau"],
			pi_lr=hyperparams["pi_lr"],
			q_lr=hyperparams["q_lr"],
			pol_hidden_dim=hyperparams["pol_hidden_dim"],
			critic_hidden_dim=hyperparams["critic_hidden_dim"],
			attend_heads=hyperparams["attend_heads"])
	model.init_dict = {}

	replay_buffer = ReplayBuffer(buffer_length, n_agents,
								[state_size for i in range(n_agents)],
								[action_size for i in range(n_agents)])

	print("MAX STEPS: "+ str(max_steps))
	print("NUM EPISODES: ",num_episodes)
	print("HYPERPARAMS: ")
	print(hyperparams)

	start_time = time.time()

	for ep in range(num_episodes):
		print("Episode "+str(ep)+":", flush=True)
		obs, info = env.reset(True, True)
		model.prep_rollouts(device=DEVICE)
		reward_sum_for_this_episode = 0

		for steps in range(max_steps):
			if steps%step_size == 0:
				print("=", end="", flush=True)
			for agent in env.get_agent_handles():
				if obs[agent] is not None:
					if use_fast_tree_obs:
						agent_obs[agent] = obs[agent]
					else:
						agent_obs[agent] = normalize_observation(
							obs[agent],
							observation_tree_depth,
							observation_radius=observation_radius)
				else:
					agent_obs[agent] = np.array([0.]*state_size)
			
			action_dict = {}
			agent_actions = []

			torch_obs = [Variable(torch.Tensor([agent_obs[i]]), requires_grad=False) for i in range(n_agents)]
			torch_agent_actions = model.step(torch_obs, explore=True)
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]


			for i in range(n_agents):
				dist = torch_agent_actions[i][0]
				idx = -1
				for j in range(action_size):
					if dist[j] != 0:
						idx = j
						break
				action_dict[i] = idx

			next_obs, all_rewards, done, info = env.step(action_dict)

			rewards = []
			dones = []

			next_agent_obs = np.array([None] * env.get_num_agents())

			for agent in env.get_agent_handles():
				if next_obs[agent] is not None:
					if use_fast_tree_obs:
						next_agent_obs[agent] = next_obs[agent]
					else:
						next_agent_obs[agent] = normalize_observation(
							obs[agent],
							observation_tree_depth,
							observation_radius=observation_radius)
				else:
					next_agent_obs[agent] = np.array([0.]*state_size)

			for i in range(n_agents):
				reward_sum_for_this_episode += all_rewards[i]
				rewards.append(all_rewards[i])
				dones.append(done[i])

			replay_buffer.push(
				np.array([agent_obs]),
				np.array(agent_actions),
				np.array([rewards]),
				np.array([next_agent_obs]), 
				np.array([dones])
			)


			if steps % num_steps == 0:
				model.prep_training(device=DEVICE)
				sample = replay_buffer.sample(batch_size, norm_rews=False)
				#print(sample)
				model.update_critic(sample)
				model.update_policies(sample)
				model.update_all_targets()
				model.prep_rollouts(device=DEVICE)

		reward_sum_for_this_episode /= n_agents
		reward_q.append(reward_sum_for_this_episode)

		if len(reward_q) == avg_steps:
			wandb.log({'reward': np.mean(reward_q)})
			reward_q = []

		print()

		if ep%steps_to_save_model == 0:
			print("\nSaving model")
			model.save(os.getcwd() + "/model.pt")
			cur_time = time.time()
			time_elapsed = (cur_time - start_time)//60
			print("Time Elapsed: " + str(time_elapsed) + "\n")

if __name__ == '__main__':
	for env in env_params:
		print("Using env:")
		print(env)
		train(env)