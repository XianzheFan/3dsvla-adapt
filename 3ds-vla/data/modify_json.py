import os
import json
import tqdm
json_folder = './train_json'
target_folder = './train_json_single'
if not os.path.exists(target_folder):
    os.mkdir(target_folder)
json_list = os.listdir(json_folder)
for json_file in tqdm.tqdm(json_list):
    if 'bimanual' in json_file:
        continue
    with open(os.path.join(json_folder,json_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # split_list = data['image'].split('/')
    # if split_list[1] == 'peract_bimanual':
    #     split_list[1] = 'RLBench'
    # # split_list[2] = 'train_dataset'
    # data['image'] = '/'.join(split_list)
    
    with open(os.path.join(target_folder,json_file), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
