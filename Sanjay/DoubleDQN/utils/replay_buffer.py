import numpy as np

class ReplayBuffer():
    def __init__(self,max_size, input_shape, n_actions):
        self.mem_size=max_size
        self.mem_cntr=0

        self.state_memory=[None] * max_size
        self.new_state_memory=[None] * max_size
        self.action_memory=[None] * max_size
        self.reward_memory=[None] * max_size
        # self.terminal_memory=np.zeros(self.mem_size,dtype=np.uint8)
        self.terminal_memory=[None] * max_size
        

    def store_transition(self, state, action, reward, next_state, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index]=state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.new_state_memory[index]=next_state
        self.terminal_memory[index]=done
        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_cntr, self.mem_size)

        batch=np.random.choice(max_mem, batch_size, replace=False).astype(int) #max index of max_mem and shape batchsize. replace=False makes it so indexes aren't repeated
        states=[self.state_memory[i] for i in batch]
        actions=[self.action_memory[i] for i in batch]
        rewards=[self.reward_memory[i] for i in batch]
        next_states=[self.new_state_memory[i] for i in batch]
        dones=[self.terminal_memory[i] for i in batch]

        return states, actions, rewards, next_states, dones