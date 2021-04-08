from flatland.envs.agent_utils import RailAgentStatus
from utils_distance import get_distance_to_target

def augment_reward(obs, next_obs):
	'''
	Given an observation for a train, returns
	the augmented reward to the train for being
	in that state
	'''
	if RailAgentStatus.get_agent_status(obs) == RailAgentStatus.DONE:
		return +10
	elif RailAgentStatus.get_agent_status(obs) == RailAgentStatus.STOPPED:
		return -1
	elif RailAgentStatus.get_agent_status(obs) == RailAgentStatus.DEADLOCKED:
		return -10:
	elif get_distance_to_target(obs) - get_distance_to_target(next_obs) > 0:
		return +0.3 
	else:
		return +0.1