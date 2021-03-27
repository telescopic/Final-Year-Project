import json
import os
"""directory="D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/actions/envs"
for filename in os.listdir(directory):
	with open(os.path.join(directory, filename)) as f:
		data = json.load(f)
	for actions in data:
		temp={}
		for key in actions:
			temp[int(key)]=actions[key]
		actions=temp
	print(data)"""
json_file= "D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs/Level_430.json"
tempdata=[]
with open(json_file) as f:
	data=json.load(f)
	print(data)
	print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
	for actions in data:
		temp={}
		for key in actions:
			temp[int(key)]=actions[key]
		print("TEMP",temp)
		actions=temp
		print("ACTIONS",actions)
		tempdata.append(actions)
	print(tempdata)