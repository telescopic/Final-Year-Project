# from DQFDDDQN.utils.demo_replay_buffer import ReplayBuffer
class TransitionSaver:
    def __init__(self, replaybuffer):
        self.memory = replaybuffer
        self.transitions_state = []
        self.transitions_action = []
        self.transitions_reward = []
        self.transitions_n_reward = []
        self.transitions_next_state = []
        self.transitions_done = []
        self.GAMMA = 0.99
        self.curr_n_reward = 0

    def add_transition(self, state, action, reward, n_reward, next_state, done):
        if not done:
            self.transitions_state.insert(0, state)
            self.transitions_action.insert(0, action)
            self.transitions_reward.insert(0, reward)
            self.transitions_n_reward.insert(0, n_reward)
            self.transitions_next_state.insert(0, next_state)
            self.transitions_done.insert(0, done)           

            transitions_n_reward = []
            gamma = 1
            for i in range(len(self.transitions_reward)):
                self.curr_n_reward = self.curr_n_reward + gamma*self.transitions_reward[i]
                transitions_n_reward.append(self.curr_n_reward)
                gamma = gamma * self.GAMMA
            self.transitions_n_reward = transitions_n_reward

        else:
            to_return = (self.transitions_state, self.transitions_action, self.transitions_reward, self.transitions_n_reward, self.transitions_next_state, self.transitions_done)
            
            self.transitions_state = []
            self.transitions_action = []
            self.transitions_reward = []
            self.transitions_n_reward = []
            self.transitions_next_state = []
            self.transitions_done = []

            return to_return