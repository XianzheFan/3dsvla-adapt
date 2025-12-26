import json
dict = {'111':['222']}
file_path = '/new/algo/user8/zhangmingxu/Grounded-Segment-Anything/try_1105_2.json'

# Writing the dictionary to the JSON file
with open(file_path, 'w') as json_file:
    json.dump(dict, json_file, indent=4)
