import pandas as pd
#from demo_replay_buffer import ReplayBuffer
import pickle
# class Test():
#     def __init__(self):
#         msg = "Hi"
        
# test = {}
# test[0] = [1,2,3]
# test[1] = ["hi", "test"]
# test[2] = Test()

# f = open("test.pkl","wb")
# pickle.dump(test,f)
# f.close()

# f = open("expert.pkl","rb")
# expert_agent = pickle.load(f)
# no_of_agents = 5
# if(no_of_agents != expert_agent[1][8] or expert_agent[1][9] == 1):
#     no_of_agents = expert_agent[1][8]

expert_agent = pd.read_csv("expert.csv")
print((expert_agent.norm_state == "None").sum())
print(len(expert_agent))

print(len(expert_agent) - (expert_agent.norm_state == "None").sum())
# print(int(expert_agent.iloc[[0]]['no_of_agents']))
# demo_memory = ReplayBuffer(130000,[231],5)

# for i, row in expert_agent.iterrows():
#     print(row["state"], row["action"], row["reward"], [0]*231, row["done"])
#     print(i)
#     if row["done"]==True:
#         if row["state"] == None:
#             pass
#         else:
#             demo_memory.store_transition(row["norm_state"], row["action"], row["reward"], [0]*231, row["done"])
#     else:
#         print(row["done"])
#         demo_memory.store_transition(row["norm_state"], row["action"], row["reward"], row["norm_next_state"], row["done"])
#     break
# print("REWARD",demo_memory.reward_memory)



# initial_gamma = 1, gamma = 0.99, n=5
# [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
# # n_step_reward = 0.0
# # for i in reward_arr:
# #     for i in range(n):
#         n_step_reward += 

