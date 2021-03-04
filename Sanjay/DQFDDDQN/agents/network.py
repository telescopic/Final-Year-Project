import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, lam_L2, n_actions, input_dims, model_name, model_dir):
        super(DeepQNetwork, self).__init__()
        self.model_dir = model_dir
        self.model_file = os.path.join(self.model_dir, model_name)
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay=lam_L2)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        probs = F.relu(self.fc4(x))

        return probs

    def save_model(self):

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        T.save(checkpoint, self.model_file)
        print('--SAVED MODEL--')

    def load_model(self):
        checkpoint = T.load(self.model_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('--LOADED MODEL--')