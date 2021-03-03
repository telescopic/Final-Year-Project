import json
import os
json_file= "D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs/Level_100.json"
with open(json_file) as f:
  data = json.load(f)
count =0
for actions in data:
	count+=1
print(count)