from agents.network import DeepQNetwork
import numpy as np
import torch as T
from utils.replay_buffer import ReplayBuffer
from utils.demo_replay_buffer import DemoReplayBuffer

import pandas as pd
import pickle 
import ast
from tqdm import tqdm

class Model():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                mem_size, expert_mem_size, batch_size, n_step,lam_n_step, lam_sup, lam_L2, eps_min=0.01, eps_dec=5e-7, 
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

        self.n_step = n_step

        self.lam_n_step = lam_n_step
        self.lam_sup = lam_sup
        self.lam_L2 = lam_L2

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.demo_memory = DemoReplayBuffer(expert_mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.lam_L2, self.n_actions, 
                                    input_dims = self.input_dims, 
                                    model_name = env_name+"_"+algo+"_q_eval", 
                                    model_dir=chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.lam_L2, self.n_actions, 
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

    def store_transitions_demo(self, state, action, reward, n_reward, next_state, done):
        self.demo_memory.store_transition(state, action, reward, n_reward, next_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done=self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, new_states, dones

    def sample_memory_demo(self):
        state, action, reward, n_reward, new_state, done=self.demo_memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        n_rewards = T.tensor(n_reward).to(self.q_eval.device)
        #print("NEW STATE LEN",len(new_state[0]))
        try:
            new_states = T.tensor(new_state).to(self.q_eval.device)
            dones = T.tensor(done).to(self.q_eval.device)

            return states, actions, rewards, n_rewards, new_states, dones
        except:
            print("NEW STATE",new_state)
            print("DONE",done)

    def take_action(self, obs):
        if(np.random.random() > self.epsilon):
            state = T.tensor(obs, dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action 

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8


    # def demonstration(self, no_of_iterations):
    #     ## load the expert agent and store it in the replay buffer
    #     expert_agent = pd.read_csv("C:/Users/sanja/Desktop/College/8th Semester/Final Year Project/Codebase/Final-Year-Project-GIT/Sanjay/DQFDDDQN/agents/expert.csv")
    #     print("[LOG]LOADED CSV")
    #     n_reward_dict = {}
    #     n_state_dict = {}
    #     n_gamma_dict = {}
    #     n_running_total = {}
    #     no_of_agents = int(expert_agent.iloc[[0]]['no_of_agents'])
    #     env_id = int(expert_agent.iloc[[0]]['env_id'])

    #     for i in range(no_of_agents):
    #         n_reward_dict[i] = []
    #         n_state_dict[i] = []
    #         n_running_total[i] = 0.0
    #         n_gamma_dict[i] = 1

    #     for i, initial_row in expert_agent.iterrows():

    #         if no_of_agents != int(expert_agent.iloc[[i]]['no_of_agents']) or initial_row['reset_env_flag']==1:

    #             no_of_agents = int(expert_agent.iloc[[i]]['no_of_agents'])
    #             env_id = int(expert_agent.iloc[[i]]['env_id'])
    #             for i in range(no_of_agents):
    #                 n_reward_dict[i] = []
    #                 n_state_dict[i] = []
    #                 n_running_total[i] = 0.0
    #                 n_gamma_dict[i] = 1

    #         agent_id = initial_row["agent_id"]

    #         if initial_row['norm_state'] == None:
    #             pass
    #         else:
    #             for j in range(self.n_step):
    #                 row = i+(j*self.n_step)
    #                 num_curr_agents = int(expert_agent.iloc[[row]]['no_of_agents'])
    #                 curr_env_id = int(expert_agent.iloc[[row]]['env_id'])
    #                 if row <= len(expert_agent) and no_of_agents == num_curr_agents and env_id == curr_env_id:
    #                     curr_row = expert_agent.iloc[[row]]
                        
    #                     if j == self.n_step-1:
    #                         #print("Norm Next State",curr_row['norm_next_state'])
    #                         #print("Norm Next State Item",curr_row['norm_next_state'].isnull().item())

    #                         if curr_row['norm_next_state'].isnull().item():
    #                             #print(curr_row['reward'] * n_gamma_dict[agent_id])
    #                             n_reward_dict[agent_id].append(curr_row['reward'] * n_gamma_dict[agent_id])
    #                         else:
    #                             # send next state to nn
    #                             next_state = curr_row['norm_next_state'].item()
                                
    #                             #print("Norm Next State NN",np.fromstring(next_state[1:-2]))
    #                             action_values = self.q_next.forward(curr_row['norm_next_state'].item())
    #                             n_reward_dict[agent_id].append(T.max(action_values).cpu().numpy()[0] * n_gamma_dict[agent_id])

    #                         n_step_reward = sum(n_reward_dict[agent_id])
    #                         #store it in replay buffer
    #                         if curr_row['norm_next_state']==None:
    #                             self.store_transitions_demo(initial_row['norm_state'], initial_row['action'], initial_row['reward'], n_step_reward, [0]*231, initial_row['done'])
    #                         else:
    #                             self.store_transitions_demo(initial_row['norm_state'], initial_row['action'], initial_row['reward'], n_step_reward, initial_row['norm_next_state'], initial_row['done'])
                            
    #                     else:
    #                         n_reward_dict[agent_id].append(curr_row['reward'] * n_gamma_dict[agent_id])

    #                     n_gamma_dict[agent_id] *= 0.99

    #             n_reward_dict[agent_id] = []
    #             n_gamma_dict[agent_id] = 1
            
    #     print("[LOG]LOADED EXPERT DEMONSTRATIONS")
    #     print("[LOG]Size:", self.demo_memory.mem_cntr)

    #     with open("expert_data.pkl","rb") as output:
    #         pickle.dump(self.demo_memory, output, -1)
    #     print('[LOG]SAVED REPLAY BUFFER')
    #     for i in range(no_of_iterations):
    #         self.pre_learn()

    def demonstration(self, no_of_iterations):
        f = open("expert.pkl","rb")
        expert_agent = pickle.load(f)
        n_reward_dict = {}
        n_state_dict = {}
        n_gamma_dict = {}
        n_running_total = {}
        no_of_agents = expert_agent[0][8]
        env_id = expert_agent[0][10]

        for i in range(no_of_agents):
            n_reward_dict[i] = []
            n_state_dict[i] = []
            n_running_total[i] = 0.0
            n_gamma_dict[i] = 1

        for i in range(len(expert_agent)):
            if (no_of_agents != expert_agent[i][8] or expert_agent[i][9] == 1):

                no_of_agents = expert_agent[i][8]
                env_id = expert_agent[i][10]
                n_reward_dict = {}
                n_state_dict = {}
                n_gamma_dict = {}
                n_running_total = {}

                for j in range(no_of_agents):
                    n_reward_dict[j] = []
                    n_state_dict[j] = []
                    n_running_total[j] = 0.0
                    n_gamma_dict[j] = 1

            agent_id = expert_agent[i][0]
            # print(expert_agent[i][2].isnull())
            if expert_agent[i][2] == "None":
                pass
            else:
                for j in range(self.n_step):

                    row = i + (j * self.n_step)
                    if row < len(expert_agent)-1:
                        num_curr_agents = expert_agent[row][8]
                        curr_env_id = expert_agent[row][10]

                        if no_of_agents == num_curr_agents and env_id == curr_env_id:
                            
                            curr_row = expert_agent[row]

                            if j == self.n_step-1:
                                if curr_row[6] == "None":
                                    n_reward_dict[agent_id].append(curr_row[4] * n_gamma_dict[agent_id])
                                else:
                                    try:
                                        next_state = T.from_numpy(curr_row[6]).to(self.q_next.device)
                                    except:
                                        print(type(curr_row[6]))
                                    action_values = self.q_next.forward(next_state.float())
                                    # print(T.max(action_values).detach().cpu().numpy())
                                    n_reward_dict[agent_id].append(T.max(action_values).detach().cpu().numpy() * n_gamma_dict[agent_id])

                                n_step_reward = T.tensor(sum(n_reward_dict[agent_id]), dtype=T.float32)

                                if expert_agent[i][6] == "None":
                                    self.store_transitions_demo(expert_agent[i][2], expert_agent[i][3], expert_agent[i][4], n_step_reward, [0]*231, expert_agent[i][7])
                                else:
                                    self.store_transitions_demo(expert_agent[i][2], expert_agent[i][3], expert_agent[i][4], n_step_reward, expert_agent[i][6], expert_agent[i][7])
                            else:
                                n_reward_dict[agent_id].append(curr_row[4] * n_gamma_dict[agent_id])
                            n_gamma_dict[agent_id] *= 0.99
                        else:
                            break
                    # else:
                    #     print("ROW",row)
                    #     print("ROW-J",row-j)
                    #     print("")
                    #     next_state = expert_agent[row-j][6]
                    #     if next_state =="None":
                    #         n_reward_dict[agent_id].append(expert_agent[row-j][4] * n_gamma_dict[agent_id])
                    #     else:
                    #         next_state = T.from_numpy(next_state).to(self.q_next.device)
                    #         action_values = self.q_next.forward(next_state.float())
                    #         n_reward_dict[agent_id].append(T.max(action_values).detach().cpu().numpy() * n_gamma_dict[agent_id])

                    #     n_step_reward = sum(n_reward_dict[agent_id])
                    #     if curr_row[6] == "None":
                    #         self.store_transitions_demo(expert_agent[row-j][2], expert_agent[row-j][3], expert_agent[row-j][4], n_step_reward, [0]*231, expert_agent[row-j][7])
                    #     else:
                    #         self.store_transitions_demo(expert_agent[row-j][2], expert_agent[row-j][3], expert_agent[row-j][4], n_step_reward, expert_agent[row-j][6], expert_agent[row-j][7])
                    #     break
                n_reward_dict[agent_id] = []
                n_gamma_dict[agent_id] = 1

        print("[LOG]LOADED EXPERT DEMONSTRATIONS")
        print("[LOG]Size:", self.demo_memory.mem_cntr)

        # with open("demo_memory.pkl","rb") as output:
        #     pickle.dump(self.demo_memory, output, -1)
        print("[LOG] Starting pre training ")
        for i in tqdm(range(no_of_iterations)):
            self.pre_learn() 

    def pre_learn(self):

        if self.learn_step_counter%self.replace_target_cntr==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
        states, actions, rewards, n_rewards, next_states, dones = self.sample_memory_demo()
        # print(len(next_states))
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states.float())[indices, actions.cpu().numpy()]
        q_next = self.q_next.forward(next_states.float())
        q_eval=self.q_eval.forward(next_states.float())

        max_action = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_action]

        q_loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        n_step_loss = self.q_eval.loss(q_target, n_rewards).to(self.q_eval.device)
        
        supervised_learning_loss = 0.0
        for i in range(self.batch_size):
            ae = actions.cpu().numpy()[i]    
            max_value = float("-inf")
            for a in self.action_space:
                max_value = max(q_next[i][a] + self.loss_l(ae, a), max_value)
            # print(max_value)
            supervised_learning_loss += dones[i] * (max_value - q_next[i][ae])

        self.q_eval.optimizer.zero_grad()
        #print(q_loss.dtype, n_step_loss.dtype, supervised_learning_loss.dtype)
        loss = 0.5*q_loss + self.lam_n_step*n_step_loss + self.lam_sup*supervised_learning_loss
        #print(loss)
        loss.to(self.q_eval.device)
        # print(loss.double().dtype)
        loss.backward()
        #print("backwards doen")
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

        q_pred = self.q_eval.forward(states.float())[indices, actions.cpu().numpy()]
        q_next = self.q_next.forward(next_states.float())
        q_eval = self.q_eval.forward(next_states.float())

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