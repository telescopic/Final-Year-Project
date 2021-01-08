from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

import numpy as np

from utils_deadlock_check import check_if_all_blocked
from utils.observation_utils import normalize_observation
from utils.choose_env import get_env
from agents.agent import Model

## WANDB INIT ##
import wandb
wandb.init(project="ENTER WANDB PROJECT NAME HERE", name = "ENTER RUN NAME HERE")


# Observation parameters
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30


# Break agents from time to time
malfunction_parameters = MalfunctionParameters(
    malfunction_rate=1. / 10000,  # Rate of malfunctions
    min_duration=20,  # Minimal duration
    max_duration=50  # Max duration
)

# Observation builder
predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

speed_profiles = {
    1.: 1.0,  # Fast passenger train
    1. / 2.: 0.0,  # Fast freight train
    1. / 3.: 0.0,  # Slow commuter train
    1. / 4.: 0.0  # Slow freight train
}

# Calculate the state size given the depth of the tree observation and the number of features

n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
state_size = tree_observation.observation_dim * n_nodes
# state_size = 231

# The action space of flatland is 5 discrete actions
action_size = 5

n_episodes = 2000
n_count = 0

## BEGIN ENV PARAMS ##

max_rails_between_cities = 2

## END ENV PARAMS ##

## BEGIN INSTANTIATE MODELS ##

agent = Model(state_size, action_size)

## END INSTANTIATE MODELS ##

def get_actions(obs, n_agents, info):
    actions = {}
    for _idx in obs.keys():
        if obs[_idx][0]!=None and info['action_required'][_idx]==True and info['malfunction'][_idx]!=1:
            normalized_obs = normalize_observation(obs[0])
            action = agent.take_action(normalized_obs)
            actions[_idx] = action
        else:
            actions[_idx] = 0
    return actions

scores=[]
env_count = 0
best_reward = -np.inf

for episode_count in range(n_episodes):
    # print("EPISODE: ",episode_count)

    env_params = get_env(env_count)
    env = RailEnv(
        width=env_params['x_dim'],
        height=env_params['y_dim'],
        rail_generator=sparse_rail_generator(
        max_num_cities=env_params['n_cities'],
        grid_mode=False,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=env_params['max_rails_in_city']
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=env_params['n_agents'],
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=DummyObservationBuilder(),
        random_seed=42
    )
    env_count += 1

    obs,info=env.reset(True,True)

    max_steps = int(4 * 2 * (env.height + env.width + (env_params['n_agents'] / env_params['n_cities'])))
    score = 0
    for step in range(max_steps):
        
        actions = get_actions(obs, env_params['n_agents'], info)
        next_obs,all_rewards,done,info=env.step(actions)
        score += np.mean(all_rewards)
        obs = next_obs

        if done['__all__']:
            break
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print("Episode", episode_count, "Avg Reward:", avg_score)
    wandb.log({"Mean Reward": avg_score})

    if(avg_score > best_reward):
      best_reward = avg_score
      agent.save_models()

        