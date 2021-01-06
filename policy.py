'''
TODO: module docstring
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
	'''
	Policy network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, activation=F.leaky_relu):
		'''
		Params:
			* in_dim: input layer dimensions
			* out_dim: output layer dimensions
			* hidden_dim: hidden layer dimensions
			* activation: non linear activation function
		'''
		self.layer1 = nn.Linear(in_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, out_dim)

		self.activation = activation

	def forward(self, obs):
		'''
		Params:
			* obs: one or more observations
		Returns:
			* probabilties of each action being taken
		'''
		self.layer1_out = self.activation(self.layer1(observations))
		self.layer2_out = self.activation(self.layer2(self.layer1_out))
		self.layer3_out = self.layer3(self.layer2_out)

		probabilities = F.softmax(self.layer3_out, dim=1)

		return probabilities
