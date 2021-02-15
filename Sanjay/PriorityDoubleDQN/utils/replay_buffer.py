import numpy as np

class ReplayBuffer():
    def __init__(self,max_size, input_shape, n_actions, alpha):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha

        self.state_memory = np.array([None] * max_size)
        self.new_state_memory = np.array([None] * max_size)
        self.action_memory = np.array([None] * max_size)
        self.reward_memory = np.array([None] * max_size)
        self.terminal_memory = np.array([None] * max_size)
        self.priorities = np.array([0] *max_size, dtype=np.float)
        

    def store_transition(self, state, action, reward, next_state, done):

        index = self.mem_cntr % self.mem_size
        #print(index)
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.priorities[index] = max(self.priorities) if self.mem_cntr > 0 else 1.0
        #print("SELF.PRIORITY",self.priorities)
        self.mem_cntr += 1

    def update_priorities(self, indices, errors, offset):
        priorities = abs(errors) + offset
        # print("PRIORITy",priorities)
        self.priorities[indices] = priorities
        
        #print("SELF.PRIOc",self.priorities)

    def sample_buffer(self,batch_size, beta):
        max_mem=min(self.mem_cntr, self.mem_size)
        priorities = self.priorities[:max_mem]

        # if np.all(self.priorities==self.priorities[0]):
        #     print("NOT EQUAL",self.priorities)
        
        # print("PRIORITIES", priorities)
        probabilities = (priorities ** self.alpha) / ((priorities ** self.alpha).sum()) # Pr = Pi ^ a / P ^ a
        #print("PROB",probabilities)
        batch=np.random.choice(max_mem, batch_size, p = probabilities, replace=False).astype(int) #max index of max_mem and shape batchsize. replace=False makes it so indexes aren't repeated
        
        importance = (max_mem * probabilities[batch]) ** (-beta)
        importance = importance / importance.max() #normalizing importance keeping it between 0 and 1
        
        states=[self.state_memory[i] for i in batch]
        actions=[self.action_memory[i] for i in batch]
        rewards=[self.reward_memory[i] for i in batch]
        next_states=[self.new_state_memory[i] for i in batch]
        dones=[self.terminal_memory[i] for i in batch]

        #print("states", states)

        return states, actions, rewards, next_states, dones, importance