import os
import re
import pickle
import json
import numpy as np
def discretize_1(q):
    q = [element / 0.01 for element in q]
    q = [round(element) for element in q]
    return q

def discretize_2(q):
    q = [element / 0.02 for element in q]
    q = [round(element) for element in q]
    return q

def format_prompt_question(data_item):
    task= data_item['variation_description']
    gripper_pos_now = data_item['gripper_pos_n-1']
    gripper_open_now = data_item['gripper_open_n-1']
    
    #if isinstance(gripper_pos_now, np.ndarray):
    #    gripper_pos_now = gripper_pos_now.tolist()
    #if isinstance(gripper_open_now, np.ndarray):
    #    gripper_open_now = gripper_open_now.tolist()

    
    if gripper_pos_now is not None:
        pos=gripper_pos_now[:3]
        rot=gripper_pos_now[3:]
        rot = discretize_1(rot)
        pos = discretize_1(pos)

        #gripper_pos_now_str = ', '.join(map(str, gripper_pos_now))
        pos_str = ', '.join(map(str, pos))
        rot_str = ', '.join(map(str, rot))
        # prompt=("Below is an instruction that describes a task. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         "### Instruction:\nI fail to {task}. Correct the gripper's position, rotation and open-status of completing the task. "
        #         "The gripper position now is [{pos_str}]. The gripper rotation quaternion now is [{rot_str}]. The gripper open-status now is {gripper_open_now} \n\n"
        #         "### Response:").format_map({'task': task, 'pos_str': pos_str, 'rot_str': rot_str, 'gripper_open_now': gripper_open_now})
        prompt=(
                "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current gripper position is [{pos_str}]. The current gripper rotation is [{rot_str}]. The current gripper state is {gripper_open_now}").format_map({'task': task, 'pos_str': pos_str, 'rot_str': rot_str, 'gripper_open_now': gripper_open_now})
    else:
        prompt=(
                "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.").format_map({'task': task})
    #prompt=(
    #            "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.").format_map({'task': task})
    return prompt

def format_prompt_answer(data_item):
    gripper_pose_future = data_item['gripper_pos_n']
    gripper_open_future = data_item['gripper_open_n']

    if isinstance(gripper_pose_future, np.ndarray):
        gripper_pose_future = gripper_pose_future.tolist()
    if isinstance(gripper_open_future, np.ndarray):
        gripper_open_future = gripper_open_future.tolist()

    pos=gripper_pose_future[:3]
    rot=gripper_pose_future[3:]
    rot = discretize_1(rot)
    pos = discretize_1(pos)
    
    gripper_pose_future_str = ', '.join(map(str, gripper_pose_future))
    pos_str = ', '.join(map(str, pos))
    rot_str = ', '.join(map(str, rot))
    prompt=("The gripper position is [{pos_str}]. The gripper rotation quaternion is [{rot_str}]. "
            "The gripper open status is {gripper_open_future} ").format_map({'pos_str': pos_str, 'rot_str': rot_str, 'gripper_open_future': gripper_open_future})
    return prompt

def find_episode_folders(root_path):
    episode_folders = []
    episode_pattern = re.compile(r'episode\d+')

    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if episode_pattern.match(dirname):
                episode_folders.append(os.path.join(dirpath, dirname))

    return episode_folders

def get_image_paths(image_folder):
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    
    #img_idx = [int(os.path.basename(path).split('.')[0]) for path in image_paths]
    img_idx = []
    valid_image_paths = []
    for path in image_paths:
        try:
            idx = int(os.path.basename(path).split('.')[0])
            img_idx.append(idx)
            valid_image_paths.append(path)
        except (ValueError, IndexError):
            continue
    image_paths = valid_image_paths
            

    img_idx.sort()
    image_paths = [os.path.join(image_folder, str(idx)+'.png') for idx in img_idx]
    #image_paths.sort()
    return image_paths

def idx_filter(image_paths, gripper_pos, gripper_open, touch_force):
    idxs= [int(os.path.basename(path).split('.')[0]) for path in image_paths]
    if isinstance(gripper_pos, np.ndarray):
        gripper_pos = gripper_pos.tolist()
    if isinstance(gripper_open, np.ndarray):
        gripper_open = gripper_open.tolist()

    pos=[pos[:3] for pos in gripper_pos]
    rot=[pos[3:] for pos in gripper_pos]
    rot = [discretize_2(i) for i in rot]
    pos = [discretize_2(i) for i in pos]

    pop_idxs=[]

    for i in range(len(idxs)-1):
        if idxs[i]-idxs[i+1]==-1 and pos[i]==pos[i+1] and rot[i]==rot[i+1]:
            pop_idxs.append(i)
    for i in reversed(pop_idxs):
        idxs.pop(i)
        image_paths.pop(i)
        gripper_pos.pop(i)
        gripper_open.pop(i)
        touch_force.pop(i)
        pos.pop(i)
        rot.pop(i)
    
    for i in range(len(idxs)-1,0,-1):
        pos_diff = sum(abs(a-b) for a,b in zip(pos[i], pos[i-1]))
        rot_diff = sum(abs(a-b) for a,b in zip(rot[i], rot[i-1]))
        if pos_diff<=3 and rot_diff<=4:
            idxs.pop(i-1)
            image_paths.pop(i-1)
            gripper_pos.pop(i-1)
            gripper_open.pop(i-1)
            touch_force.pop(i-1)
            pos.pop(i-1)
            rot.pop(i-1)
    return image_paths, gripper_pos, gripper_open, touch_force

#def idx_touch_filter(image_paths, gripper_pos, gripper_open, touch_force):

#   return image_paths, gripper_pos, gripper_open, touch_force       

        

def get_dataset(root_path):
    #episode_folders = find_episode_folders(root_path)
    episode_folders = []
    failed_episodes = ['2024-8-19-4-15-11-100-78', '2024-8-19-4-15-11-6-13','2024-8-19-4-15-11-37-84','2024-8-19-4-15-11-54-54',
                       '2024-8-19-4-15-11-41-64','2024-8-19-4-15-11-68-7','2024-8-19-4-15-11-75-41','2024-8-19-4-15-11-44-71','2024-8-19-4-15-11-69-17',
                       '2024-8-19-4-15-11-3-51','2024-8-19-4-15-11-53-94','2024-8-19-4-15-11-21-67','2024-8-19-4-15-11-3-81','2024-8-19-4-15-11-45-73',
                       '2024-8-19-4-15-11-44-26','2024-8-19-4-15-11-95-53','2024-8-19-4-15-11-92-54','2024-8-19-4-15-12-80-32']
    failes =[]
    idx = 0
    for item in os.listdir(root_path):
        idx += 1
        if item not in failed_episodes:
            episode_folders.append(os.path.join(root_path, item))
        else:
            failes.append(item)

    ann=[]
    for episode_folder in episode_folders:
        image_folder = os.path.join(episode_folder, 'front_rgb')
        with open(os.path.join(episode_folder, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(episode_folder, 'variation_descriptions.pkl'), 'rb') as f:
            variation_description = pickle.load(f)
        image_paths = get_image_paths(image_folder)
        image_idx= [int(os.path.basename(path).split('.')[0]) for path in image_paths]
        #print(image_idx)
        #print(episode_folder)
        #print(len(demo))
        #print(demo[0])
        gripper_pos = [demo[idx].gripper_pose for idx in image_idx]
        gripper_open=[demo[idx].gripper_open for idx in image_idx]
        touch_force=[demo[idx].gripper_touch_forces for idx in image_idx]

        image_paths, gripper_pos, gripper_open,touch_force=idx_filter(image_paths, gripper_pos, gripper_open, touch_force)

        #image_paths, gripper_pos, gripper_open,touch_force=idx_touch_filter(image_paths, gripper_pos, gripper_open, touch_force)

        #gripper_pos =gripper_pos.tolist()
        #gripper_open=gripper_open.tolist()
        #for i, obs in enumerate(demo):
        #    gripper_pos.append(obs.gripper_pose)
        #    gripper_open.append(obs.gripper_open)
        #print(len(gripper_pos), len(image_paths), len(gripper_open), len(variation_description))
        #print(gripper_pos)
        #print(image_paths)
        #print(gripper_open)
        #print(variation_description)
        assert len(gripper_pos) == len(image_paths)==len(gripper_open)
        for i in range(len(gripper_pos)):
            if i==0:
                data_item = {'image': image_paths[i], 'gripper_pos_n': gripper_pos[i+1].tolist(), 'gripper_open_n': gripper_open[i+1], 'variation_description': variation_description[0], 'gripper_pos_n-1': None, 'gripper_open_n-1': None}
            elif i==len(gripper_pos)-1:
                break    
            else:
                data_item = {'image': image_paths[i], 'gripper_pos_n': gripper_pos[i+1].tolist(), 'gripper_open_n': gripper_open[i+1], 'variation_description': variation_description[0], 'gripper_pos_n-1': gripper_pos[i].tolist(), 'gripper_open_n-1': gripper_open[i]}
            
            question = format_prompt_question(data_item)
            answer = format_prompt_answer(data_item)
            data_item['question'] = question
            data_item['answer'] = answer
            name=data_item['image'].split('/')[-3]+'_'+data_item['image'].split('/')[-1].split('.')[0]
        
            with open(f'/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/data/rlbench_dataset_0829/{name}.json', 'w') as f:
                json.dump(data_item, f)
        # assert(0)
    #for item in failes:
    #    print(item)
    #print(idx)
    return ann

if __name__ == '__main__':
    root_path = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672'
    output_path = './rlbench_dataset_0829'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ann = get_dataset(root_path)
    # print(len(ann))
    # #print(ann[:10])
    # #print(x_max, y_max, z_max)      0.5253432989120483 0.41891759634017944 1.4720853567123413                                                        
    # #print(x_min, y_min, z_min)      -0.12893009185791016 -0.46705663204193115 0.7693771123886108
    
    # for idx,item in enumerate(ann):
    #     name=item['image'].split('/')[-3]+'_'+item['image'].split('/')[-1].split('.')[0]
    #     with open(f'data/rlbench_dataset/{name}.json', 'w') as f:
    #         json.dump(item, f)

