from network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from torch.distributions import MultivariateNormal, Categorical


class ActorCriticAgentWithPPO:
    def __init__(self, state_size, action_size, hyperparams):
        self.init_hyperparams(hyperparams)
        self.actor = Network(in_dim=state_size, out_dim=action_size)
        self.critic = Network(in_dim=state_size, out_dim=1)

        self.try_loading_existing_model_weights()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.actor.parameters(), lr=self.critic_lr)

        self.cov_var = torch.full(size=(action_size, ), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.avg_actor_loss = 0
        self.avg_critic_loss = 0

    def try_loading_existing_model_weights(self):
        try:
            self.actor.load_state_dict(os.getcwd()+'/actor.pth')
            self.critic.load_state_dict(os.getcwd()+'/critic.pth')
            print("Using pre-trained weights")
        except(Exception):
            print("Training Actor-Critic from scratch!")

    def init_hyperparams(self, hyperparams):
        self.actor_lr = hyperparams['actor_lr']
        self.critic_lr = hyperparams['critic_lr']
        self.gamma = hyperparams['gamma']
        self.clip = hyperparams['clip']
        self.num_updates = hyperparams['num_updates']

    def get_action(self, obs):

        actions = F.softmax(self.actor(obs), dim=-1)

        #dist = MultivariateNormal(actions, self.cov_mat)

        dist = Categorical(actions)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def compute_discounted_rewards(self, rewards):
        rwds = []

        discounted_reward = 0
        for rew in reversed(rewards):
            discounted_reward = rew + discounted_reward * self.gamma

            rwds.insert(0, discounted_reward)

        return torch.tensor(rwds)

    def evaluate(self, obs, actions):
        V = self.critic(obs)

        acts = F.softmax(self.actor(obs))
        dist = Categorical(acts)
        log_probs = dist.log_prob(actions)

        return V, log_probs

    def learn(self, batch_obs, batch_actions, batch_log_probs, batch_rewards):
        batch_obs = torch.Tensor(batch_obs)
        batch_actions = torch.Tensor(batch_actions)
        batch_log_probs = torch.Tensor(batch_log_probs)
        batch_rewards = torch.Tensor(batch_log_probs)

        self.avg_actor_loss = 0
        self.avg_critic_loss = 0

        V, _ = self.evaluate(batch_obs, batch_actions)

        A_k = batch_rewards - V.detach()

        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.num_updates):
            V, curr_log_probs = self.evaluate(batch_obs, batch_actions)

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            self.avg_actor_loss += actor_loss
            #self.actor_loss_log.append(actor_loss)

            V = V.view(-1, 1)

            critic_loss = nn.MSELoss()(V, batch_rewards)
            self.avg_critic_loss += critic_loss
            #self.critic_loss_log.append(critic_loss)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.avg_actor_loss /= self.num_updates
        self.avg_critic_loss /= self.num_updates
