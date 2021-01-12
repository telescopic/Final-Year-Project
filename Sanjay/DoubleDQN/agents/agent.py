from agents.network import DeepQNetwork
import numpy as np
import torch as T
from utils.replay_buffer import ReplayBuffer
class Model():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, 
                replace=1000, chkpt_dir='tmp/dqn',algo=None, env_name=None ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.replace_target_cntr = replace 
        self.action_space=[i for i in range(self.n_actions)]
        self.learn_step_counter=0


        self.memory=ReplayBuffer(mem_size,input_dims,n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, 
                                    input_dims = self.input_dims, 
                                    model_name = env_name+"_"+algo+"_q_eval", 
                                    model_dir=chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions, 
                                    input_dims = self.input_dims, 
                                    model_name = env_name+"_"+algo+"_q_next", 
                                    model_dir=chkpt_dir)

    def decrement_epsilon(self):
        self.epsilon=self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def store_transitions(self, state, action, reward, next_state, done):
        temp=0
        try:
            for i in range(len(state)):
                temp=i
                if done[i]==True:
                    next_state.insert(i,[0]*231)
                self.memory.store_transition(state[i],action[i],reward[i],next_state[i], done[i])
                # print("DONE", done)
        except:
            print(len(state), len(action), len(reward), len(next_state), len(done))
            print("TEMP",temp)
            print("STATE", state)
            print("ACTION", action)
            print("REWARD", reward)
            print("NEXT STATE", next_state)
            print("DONE", done)
    def sample_memory(self):
        state, action, reward, new_state, done=self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, new_states, dones

    def take_action(self, obs):
        if(np.random.random() > self.epsilon):
            state = T.tensor(obs, dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action 

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            #print("SKIPPING")
            return

        if self.learn_step_counter%self.replace_target_cntr==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        #print("LEARNING")
        states, actions, rewards, next_states, dones = self.sample_memory()
        #print(states)
        indices = np.arange(self.batch_size)
        #print(indices)
        # print(self.q_eval.forward(states))
        #print(actions)
        q_pred = self.q_eval.forward(states.float())[indices, actions.cpu().numpy()]
        q_next = self.q_next.forward(next_states.float())
        q_eval=self.q_eval.forward(next_states.float())

        #print("PRED",q_pred, "EVAL",q_eval, "NEXT",q_next)

        max_action = T.argmax(q_eval, dim=1)
        #print(dones)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_action]

        #print(rewards, q_next[indices, max_action], q_target, q_pred)

        self.q_eval.optimizer.zero_grad()
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        #print("LOSS",loss.item())
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        return loss.item()

    def save_model(self):
        self.q_eval.save_model()
        self.q_next.save_model()

    def load_model(self):
        self.q_eval.load_model()
        self.q_next.load_model()