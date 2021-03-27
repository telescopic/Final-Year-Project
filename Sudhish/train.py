from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

import numpy as np

#from utils_deadlock_check import check_if_all_blocked
from utils.observation_utils import normalize_observation
from utils.choose_env import get_env
from agents.agent import Model

import os
import json
import xlwt
import pickle
## WANDB INIT ##
#import wandb
#wandb.init(project="ENTER WANDB PROJECT NAME HERE", name = "ENTER RUN NAME HERE")


from xlwt import Workbook 
  
# Workbook is created 
wb = Workbook() 

sheet1 = wb.add_sheet('Sheet 1')

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

n_episodes = 20
n_count = 0

## BEGIN ENV PARAMS ##

max_rails_between_cities = 2

## END ENV PARAMS ##

## BEGIN INSTANTIATE MODELS ##

agent = Model(state_size, action_size)

## END INSTANTIATE MODELS ##

def get_actions(obs, n_agents, info):
    """actions = {}
    for _idx in obs.keys():
        if obs[_idx][0]!=None and info['action_required'][_idx]==True and info['malfunction'][_idx]!=1:
            normalized_obs = normalize_observation(obs[0])
            action = agent.take_action(normalized_obs)
            actions[_idx] = action
        else:
            actions[_idx] = 0
    return actions"""

scores=[]
env_count = 0
best_reward = -np.inf
x=0
for episode_count in range(n_episodes):
    # print("EPISODE: ",episode_count)

    #env_params = get_env(env_count)
    directory1 = r'D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs'
    for filename in os.listdir(directory1):
        with open(os.path.join(directory1, filename)) as f:
            data=json.load(f)
        filename=filename.replace('.json','.pkl')
        with open(os.path.join(directory1, filename),'rb') as f:
            env=RailEnv(pickle.load(f))
        #env_count += 1

        obs,info=env.reset(True,True)

        max_steps = int(4 * 2 * (env.height + env.width + (env_params['n_agents'] / env_params['n_cities'])))
        score = 0
        for step in range(max_steps):
            
            #actions = get_actions(obs, env_params['n_agents'], info)
            actions=data[step]
            actionlist=actions.values()
            next_obs,all_rewards,done,info=env.step(actions)
            sheet1.write(x, 0, 1)
            sheet1.write(x, 1, actionlist[0])
            sheet1.write(x,2,all_rewards[0])
            sheet1.write(x,3,next_obs[0])
            sheet1.write(x,4,done[0])
            sheet1.write(x+1, 0, obs[1])
            sheet1.write(x+1, 1, actionlist[1])
            sheet1.write(x+1,2,all_rewards[1])
            sheet1.write(x+1,3,next_obs[1])
            sheet1.write(x+1,4,done[1])
            sheet1.write(x+1, 0, obs[2])
            sheet1.write(x+2, 1, actionlist[2])
            sheet1.write(x+2,2,all_rewards[2])
            sheet1.write(x+2,3,next_obs[2])
            sheet1.write(x+2,4,done[2])
            sheet1.write(x+3, 0, obs[3])
            sheet1.write(x+3, 1, actionlist[3])
            sheet1.write(x+3,2,all_rewards[3])
            sheet1.write(x+3,3,next_obs[3])
            sheet1.write(x+3,4,done[3])
            sheet1.write(x+4, 0, obs[4])
            sheet1.write(x+4, 1, actionlist[4])
            sheet1.write(x+4,2,all_rewards[4])
            sheet1.write(x+4,3,next_obs[4])
            sheet1.write(x+4,4,done[4])
            x=x+5
            score += np.mean(all_rewards)
            obs = next_obs

            if done['__all__']:
                break
        
        scores.append(score)
        wb.save('xlwt example.xls') 
        avg_score = np.mean(scores[-100:])
        print("Episode", episode_count, "Avg Reward:", avg_score)
        #wandb.log({"Mean Reward": avg_score})

        if(avg_score > best_reward):
          best_reward = avg_score
          agent.save_models()

        