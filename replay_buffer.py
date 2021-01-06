'''
TODO: add module docstring
'''
from random import sample
class ReplayBuffer:
	'''
	Replay buffer with random sampling
	'''
	def __init__(self, max_buffer_size):
		self.max_size = max_buffer_size
		self.buffer = []
		self.cur_idx = 0
		self.cur_size = 0

	def add(self, obs, action, reward, next_obs):
		if self.cur_size < self.max_size:
			self.buffer.append([obs, action, reward, next_obs])
			self.cur_size += 1
		else:
			self.buffer[self.cur_idx] = [obs, action, reward, next_obs]
		
		self.cur_idx = (self.cur_idx + 1)%self.max_size

	def sample(self, sample_size):
		if sample_size > self.cur_size:
			return []

		return sample(self.buffer, sample_size)
