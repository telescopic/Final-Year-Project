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

from actor_critic import ActorCriticAgentWithPPO

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from matplotlib import pyplot as plt
import os


def generate_and_save_plot(y_data, filename):
    plt.plot(y_data)
    plt.savefig(filename)
    plt.show()
    print("Saving file: " + os.getcwd() + "/" + filename)
    plt.close()


if __name__ == '__main__':
    # Environment parameters
    # small_v0
    n_agents = 5
    x_dim = 50
    y_dim = 50
    n_cities = 4
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 42
    negative_inf = -1000
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
    tree_observation = TreeObsForRailEnv(
        max_depth=observation_tree_depth, predictor=predictor)

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

    ### CHANGE THIS VARIABLE
    num_episodes = 1

    hyperparams = {
        'actor_lr': 0.005,
        'critic_lr': 0.001,
        'clip': 0.2,
        'gamma': 0.95,
        'num_updates': 10
    }

    A2C_agent = ActorCriticAgentWithPPO(
        state_size=state_size, action_size=5, hyperparams=hyperparams)

    #render_env(env)

    for i in range(num_episodes):
        print("Episode: " + str(i))

        obs, info = env.reset(True, True)

        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(
                    obs[agent],
                    observation_tree_depth,
                    observation_radius=observation_radius)

        obs_list = [[] for i in range(n_agents)]
        actions_list = [[] for i in range(n_agents)]
        log_prob_list = [[] for i in range(n_agents)]
        rewards_list = [[] for i in range(n_agents)]

        for i in range(max_steps):
            if i % step_size == 0:
                print("=", end="", flush=True)
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
                if not obs[agent]:
                    continue

                action, log_prob = A2C_agent.get_action(agent_obs[agent])
                obs_list[agent].append(agent_obs[agent])
                actions_list[agent].append(action)
                log_prob_list[agent].append(log_prob)

                action_dict[agent] = action

            next_obs, all_rewards, done, info = env.step(action_dict)

            obs = next_obs

            for agent in env.get_agent_handles():
                if obs[agent]:
                    rewards_list[agent].append(all_rewards[agent])

        for agent in env.get_agent_handles():
            rewards_list[agent] = A2C_agent.compute_discounted_rewards(
                rewards_list[agent])

        #print(rewards_list)
        for agent in env.get_agent_handles():
            A2C_agent.learn(obs_list[agent], actions_list[agent],
                            log_prob_list[agent], rewards_list[agent])

        print(flush=True)

    generate_and_save_plot(A2C_agent.actor_loss_log, 'actor_loss.png')
    generate_and_save_plot(A2C_agent.critic_loss_log, 'critic_loss.png')

    ## save models
    torch.save(A2C_agent.actor.state_dict(), os.getcwd()+'/actor.pth')
    torch.save(A2C_agent.critic.state_dict(), os.getcwd()+'/critic.pth')
    
