from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
import numpy as np

#from utils_deadlock_check import check_if_all_blocked
from utils.observation_utils import normalize_observation
from flatland.utils.rendertools import RenderTool
from utils.choose_env import get_env
from utils.observation_utils import normalize_observation
from agents.agent import Model

import os
import json
import xlwt
import pickle
import PIL
import pandas as pd
## WANDB INIT ##
#import wandb
#wandb.init(project="ENTER WANDB PROJECT NAME HERE", name = "ENTER RUN NAME HERE")


from xlwt import Workbook 
  
# Workbook is created 
wb = Workbook('xlwt example.xls') 

sheet1 = wb.add_sheet('Sheet 1')

# Observation parameters
observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

malfunction_parameters = MalfunctionParameters(
    malfunction_rate=0,  # Rate of malfunctions
    min_duration=20,  # Minimal duration
    max_duration=50  # Max duration
)


predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

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

scores=[]
env_count = 0
best_reward = -np.inf
x=0
obslist=[]
normalizeobslist=[]
action_list=[]
reward_list=[]
next_statelist=[]
next_statenormalized=[]
donelist=[]
agentcountlist=[]
varlist=[]
resetenvlist=[]
envresetcount=[]
envreset=-1
flag=0
i=100
while i<150:
	env_file = "D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs/Level_{}.pkl".format(i)
	json_file= "D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs/Level_{}.json".format(i)

	env = RailEnv(width=1, height=1,
	                      rail_generator=rail_from_file(env_file),
	                      schedule_generator=schedule_from_file(env_file),
	                      malfunction_generator_and_process_data=malfunction_from_params(
	                          malfunction_parameters),
	                      obs_builder_object=tree_observation)
	tempdata=[]
	with open(json_file) as f:
	  data = json.load(f)
	  for actions in data:
	  	temp={}
	  	for key in actions:
	  		temp[int(key)]=actions[key]
	  	tempdata.append(temp)


	#done = dict()
	##actionvalue=0
	#while done["__all__"] == False:
		#actionvalue+=1
		#print(actionvalue)
	score = 0
	obs,info=env.reset(True,True)
	flag=1
	"""env_renderer = RenderTool(env, gl="PILSVG")
	env_renderer.render_env()
	image = env_renderer.get_image()
	pil_image = PIL.Image.fromarray(image)
	pil_image.show()
	print("Env Loaded")"""
	print(len(data))
	for actions in tempdata:
		#print(actions)
		#break
		actionlist=list(actions.values())
		next_obs,all_rewards,done,info=env.step(actions)
		#print(obs[0])
		"""
		sheet1.write(x, 0, str(obs[0]))
		sheet1.write(x, 1, actionlist[0])
		sheet1.write(x,2,all_rewards[0])
		sheet1.write(x,3,str(next_obs[0]))
		sheet1.write(x,4,done[0])
		sheet1.write(x+1, 0, str(obs[1]))
		sheet1.write(x+1, 1, actionlist[1])
		sheet1.write(x+1,2,all_rewards[1])
		sheet1.write(x+1,3,str(next_obs[1]))
		sheet1.write(x+1,4,done[1])
		sheet1.write(x+2, 0, str(obs[2]))
		sheet1.write(x+2, 1, actionlist[2])
		sheet1.write(x+2,2,all_rewards[2])
		sheet1.write(x+2,3,str(next_obs[2]))
		sheet1.write(x+2,4,done[2])
		sheet1.write(x+3, 0,str( obs[3]))
		sheet1.write(x+3, 1, actionlist[3])
		sheet1.write(x+3,2,all_rewards[3])
		sheet1.write(x+3,3,str(next_obs[3]))
		sheet1.write(x+3,4,done[3])
		sheet1.write(x+4, 0, str(obs[4]))
		sheet1.write(x+4, 1, actionlist[4])
		sheet1.write(x+4,2,all_rewards[4])
		sheet1.write(x+4,3,str(next_obs[4]))
		sheet1.write(x+4,4,done[4])
		x=x+5"""
		var=0
		#print(len(env.get_agent_handles()))
		while var<len(env.get_agent_handles()):
			varlist.append(var)
			obslist.append(str(obs[var]))
			if obs[var]:
				normalizeobslist.append(normalize_observation(obs[var], observation_tree_depth, observation_radius=observation_radius))
			else:
				normalizeobslist.append("None")
			action_list.append(actionlist[var])
			reward_list.append(all_rewards[var])
			next_statelist.append(str(next_obs[var]))
			if next_obs[var]:
				next_statenormalized.append(normalize_observation(obs[var], observation_tree_depth, observation_radius=observation_radius))
			else:
				next_statenormalized.append("None")
			donelist.append(done[var])
			agentcountlist.append(len(env.get_agent_handles()))
			var=var+1
			if flag==1:
				resetenvlist.append(1)
				envreset=envreset+1
				envresetcount.append(envreset)
				flag=0
			else:
				resetenvlist.append(0)
				envresetcount.append(envreset)
		#print(done)
		#score += np.mean(all_rewards)
		obs = next_obs
		if done['__all__']:
			done['__all__']==False
			print("FINISHED")
			obs,info=env.reset(True,True)
			flag=1
		
		#scores.append(score)
		#wb.save('D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/xlwt_example.xls')
	i=i+1

	"""env_renderer = RenderTool(env, gl="PILSVG")
	env_renderer.render_env()
	image = env_renderer.get_image()
	pil_image = PIL.Image.fromarray(image)
	pil_image.show()
	print("Env Loaded")"""
df=pd.DataFrame(list(zip(varlist,obslist,normalizeobslist,action_list,reward_list,next_statelist,next_statenormalized,donelist,agentcountlist,resetenvlist,envresetcount)),columns=['Agent ID','State','Normalized State','Action','Reward','Next_State','Normalized_NextState','Done','Number of Agents','ResetEnvFlag','Envresetcounter'])
df.to_csv('expert.csv',index=False) 