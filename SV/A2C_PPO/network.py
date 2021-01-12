import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs):
        obs = torch.Tensor(obs)
        layer1_out = F.relu(self.fc1(obs))
        layer2_out = F.relu(self.fc2(layer1_out))
        out = self.fc3(layer2_out)

        return out
