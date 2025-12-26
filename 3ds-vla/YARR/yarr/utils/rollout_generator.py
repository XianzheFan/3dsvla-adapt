from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent, ActResult
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
import sys
import os
import re
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import traceback
import json
#sys.path.append('....')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '....')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

#from cloris.LLaMA_Adapter_v3_Demo_release_336_large_Chinese_ori.test_model import generate_answer
# sys.path.append('/new/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/')
# sys.path.append('/new/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/YARR/yarr/utils/')
#print(1)
from test_model_2 import generate_answer
from load_pickle import generate_pointcloud, rotation_matrix_to_quaternion2, depth_to_depthscale,worldpoint_to_pixel,pixel_to_worldpoint2,prepose_before_contact,postpose_before_contact,goto_original_pose,find_nearest_non_yellow_pixel_in_circle,find_nearest_non_yellowgreengrey_pixel_in_circle,find_nearest_segmentation_pixel_in_circle

def direction_vector(v1, v2,task):
    # 计算单位向量
    v1 = np.array(v1)
    v2 = np.array(v2)
    # print(v1,v2)
    # assert(0)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # print(v1/norm_v1,v2/norm_v2)
    # print(v1,v2)
    # v1_n = v1/norm_v1
    # v2_n = v2/norm_v2
    # print(v1_n,v2_n)
    # dot_product = v1_n[0]*v2_n[0] + v1_n[1]*v2_n[1]
    dot_product = np.dot(v1/norm_v1, v2/norm_v2)
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    if angle > 90 and 'tap' not in task:
        angle = 180-angle
    # print(dot_product,angle,np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))
    # assert(0)
    # print('angle',angle,v1,v2)
    return np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

def deal_answer(answer,obs,idx,img_dir,rgb,seed,step,task_name,rotation_array,act_result):
    
    flag_2d=0
    task_name = '_'.join(task_name.split(' '))
    try:
        print_flag=0 
        try:
            if "position is [" in answer and ']' in answer:     
                position = [int(float(x)) for x in answer.split("position is [")[1].split("]")[0].split(",")]
            elif "position is" in answer and ']' in answer:     
                position = [int(float(x)) for x in answer.split("position is")[1].split("]")[0].split(",")]
            elif "[" in answer and ']' in answer:
                position = [int(float(x)) for x in re.findall(r"\[(.*?)\]", answer)[0].split(",")]
                print(position)
            
            else:
                position = [28,-1,140]  # 默认值
        except:
            position = [int(float(x)) for x in answer.split("position is [")[1].split(".")[0].split(",")]
        position_init = position
        # if len(position)==3 and 'clock' in task_name:
        #     position = [round(x/0.01) for x in act_result.action[:3]]
        #     print('--------3d position',position)
        if len(position)==2:
                flag_2d=1
                position.reverse()
                #print(2)
                print_flag=1
                death_in_meters = obs['front_depth']
                front_camera_extrinsics = obs['misc']['front_camera_extrinsics']
                front_camera_intrinsics = obs['misc']['front_camera_intrinsics']
                death_in_meters=np.transpose(death_in_meters, (1,2,0))
                death_in_meters=np.squeeze(death_in_meters)

                img_dir_copy =img_dir
                img_dir = os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_draw.png".format(task_name=task_name,idx1=seed,idx2=step))
                img_dir_adj = os.path.join(img_dir_copy, "image_{task_name}_{idx1}_{idx2}_draw_adj.png".format(task_name=task_name,idx1=seed,idx2=step))
                img = Image.fromarray(rgb)
                img_2=Image.fromarray(rgb)
                img_3=Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)
                draw_2 = ImageDraw.Draw(img_2)
                radius = 5  # 半径为 5 像素
                rgb_value = img.getpixel((position[0], position[1]))
                #print('rgb_value: ',rgb_value)
                draw_2.ellipse((position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius), fill='red')
                img_2.save(img_dir)
                #position_2=find_nearest_non_yellow_pixel_in_circle(img,position)
                # if 'rope' not in task_name:
                position_2=find_nearest_segmentation_pixel_in_circle(img_dir,position,death_in_meters)
                # else:
                #     position_2 = position
                # print(position,position_2)
                # print(position_2)

                # 绘制一个更大的点
                
                if np.array(position_2) is not None and (position[0] != position_2[0] or position[1] != position_2[1]) and 'slide' not in task_name:
                    draw.ellipse((position_2[0] - radius, position_2[1] - radius, position_2[0] + radius, position_2[1] + radius), fill='red')
                    img.save(img_dir_adj)
                    position = position_2
                    rgb_value_2 = img_3.getpixel((position[0], position[1]))
                    # print('rgb_value_2: ',rgb_value_2)
                    
                position=pixel_to_worldpoint2(death_in_meters,front_camera_extrinsics,front_camera_intrinsics,position[0],position[1])
                #print('pridict 3d position:    ',position)
                position=discretize(position)
                if 'rope' in task_name and step <2:
                    position[0] -= 2
                
        if len(position)!=3:
            position = [28,-1,140]
    except Exception as e:
            #print(3)
            print(e)
            traceback.print_exc()
            position = [28,-1,140]  # 默认值

    
    try:
        if "quaternion is [" in answer:
            rotation = [int(x) for x in answer.split("quaternion is [")[1].split("]")[0].split(",")]
        elif 'z-axis' in answer:
            rot_y = [int(x) for x in re.findall(r"\[(.*?)\]", answer)[2].split(",")]
            rot_z = [int(x) for x in re.findall(r"\[(.*?)\]", answer)[1].split(",")]
            print('rot_y: ',rot_y)
            print('rot_z: ',rot_z)
            rotation = rotation_matrix_to_quaternion(rot_y, rot_z)
            print(rotation)
        elif "[" in answer:
             rotation = [int(x) for x in re.findall(r"\[(.*?)\]", answer)[1].split(",")]
             print(rotation)
        else:
            rotation = [0, 1, 0, 0]
    except:
            rotation = [0, 1, 0, 0]
    if rotation==[0,0,0,0]:
        rotation=[0,1,0,0]
    if len(rotation)!=4:
        rotation = [0, 1, 0, 0]
    try:
        # 判断 answer 是哪种格式并解析 open status
        if "open status is" in answer:
            open_status_match = re.search(r"open status is (\d+(\.\d+)?)", answer)
            open_status = float(open_status_match.group(1)) if open_status_match else 0.0
        else:
            # 处理第二种格式
            open_status = float(answer.split("].")[-1].strip())
            print(open_status)

    except Exception as e:
        print(e)
        traceback.print_exc()
        open_status = 1.0  # 默认值

    if open_status!=0.0 and open_status!=1.0:
        open_status=1.0
    open_status_array = np.array([open_status], dtype=np.float32)

    # 转换为 NumPy 数组
    position_array = np.array(position).astype(np.float32)
    if 'rope' in task_name and len(position_init) == 3:
        # rotation_array = rotation_array
        rotation_array = np.array(rotation).astype(np.float32)
        rotation_array[0] = -abs(rotation_array[0])
        open_status_array= [1.0]
    else:
        rotation_array = np.array(rotation).astype(np.float32)
    #open_status_array = np.array(open_status)
    
    position_array=[element*0.01 for element in position_array]
    # if 'clock' in task_name:
    #     position_array[2] = 1.032
    if 'rope' in task_name and len(position_array) ==3:
        position_array[2] = 0.75274
        # open_status_array = [0.0]
    rotation_array=[element*0.01 for element in rotation_array]
    norm=np.linalg.norm(rotation_array)
    rotation_array=[element/norm for element in rotation_array]
    if 'slide' in task_name or 'beat' in task_name or 'rope' in task_name:
        ignore_collisions_array = np.array([1.0]).astype(np.float32)
    else:
        ignore_collisions_array = np.array([0.0]).astype(np.float32)

    print('rotation_array: ',rotation_array)    

    return ActResult(np.concatenate([position_array, rotation_array, open_status_array, ignore_collisions_array])),print_flag,flag_2d,open_status_array,rotation_array

def rotation_matrix_to_quaternion(rot_y, rot_z):
    """
    Converts a rotation matrix (with only Y and Z axis information) into a quaternion.
    
    Args:
    rot_y: The second column of the rotation matrix (Y axis direction).
    rot_z: The third column of the rotation matrix (Z axis direction).
    
    Returns:
    A list of 4 elements representing the quaternion (x, y, z, w).
    """
    # Ensure rot_y and rot_z are normalized
    rot_y = rot_y / np.linalg.norm(rot_y)
    rot_z = rot_z / np.linalg.norm(rot_z)
    
    # Compute the missing X axis using cross product: X = Y × Z
    rot_x = np.cross(rot_y, rot_z)
    
    # Normalize the computed X axis to ensure orthogonality
    rot_x = rot_x / np.linalg.norm(rot_x)
    
    # Construct the full 3x3 rotation matrix
    # R = np.column_stack((rot_x, rot_y, rot_z))
    R = np.eye(3)
    R[:,0] = rot_x
    R[:,1] = rot_y
    R[:,2] = rot_z

    q = Quaternion(matrix=R)
    w,x,y,z = q
    # return q.tolist()
    return [x,y,z,w]

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    
    Args:
    q: A list or array of 4 elements representing the quaternion (w, x, y, z)
    
    Returns:
    A 3x3 rotation matrix as a numpy array
    """
    x, y, z, w = q
    
    # Compute the elements of the rotation matrix
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                  [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                  [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
    
    return R

def discretize(q):
    q = [element / 0.01 for element in q]
    q = [round(element) for element in q]
    return q

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, #agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 1,
                  record_enabled: bool = False,env_device: torch.device = None,logdir: str = None,eval_cfg=None,task_name=None):
        
        '''def get_task_name():
            if hasattr(env, '_task_class'):
                #eval_task_name = env._task_class.__name__.init_episode(env._task_class.__name__,0)[0]
                #eval_task_name = env._task_class.init_episode(0)[0]
                eval_task_name = env._task._task.init_episode(0)[0]
            elif hasattr(env, '_task_classes'):
                if env.active_task_id != -1:
                    task_id = (env.active_task_id) % len(env._task_classes)
                    #eval_task_name = env._task_classes[task_id].init_episode(env._task_classes[task_id],0)[0]
                    #eval_task_name = env._task_classes[task_id].init_episode(0)[0]
                    eval_task_name = env._task._task.init_episode(0)[0]
                else:
                    eval_task_name = ''
            else:
                raise Exception('Neither task_class nor task_classes found in eval env')
            return eval_task_name
        task_name_2 = get_task_name() '''
        task_name_blank = ' '.join(task_name.split('_'))
        # print(task_name_blank)
        # assert(0)
        task_name = '_'.join(task_name.split(' '))
        # if 'umbrella' in task_name:
        #     task_name = 'put_umbrella_out_of_umbrella_stand'
        #if eval:
        
        obs,desc = env.reset_to_demo(eval_demo_seed)

        task_name_2 = desc[0]
        # if 'umbrella' in task_name_2:
        #     task_name_2 = 'put umbrella out of umbrella stand'
        
            #print(obs.keys())
            #print(obs['gripper_joint_positions'])
            #print(obs['gripper_open'])
        ##else:
        ##    obs = env.reset()

        touch_idx=0
        #agent.reset()
        #llama_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/xcy/llama_model_weights"
        llama_dir=eval_cfg.framework.llama_path
        #adapter_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/exp/exp-0815-plain-6k/checkpoint-11.pth"
        #adapter_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/exp/exp-0829-plain-5k_touch3d/checkpoint-5.pth"
        adapter_dir=eval_cfg.framework.adapter_path
        box_flag=eval_cfg.framework.box_flag
        box_flag_2=box_flag
        target_flag = eval_cfg.framework.target_flag
        aff_flag=eval_cfg.framework.aff_flag
        rotmat_flag=eval_cfg.framework.rotmat_flag
        task_name_flag=eval_cfg.framework.task_name_flag
        
        human_frame_idx=0
        if task_name_flag:
            prompt = "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format_map({'task': task_name_blank})
        else:
            prompt = "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format_map({'task': task_name_2.split(' ')[0]})

        #obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        img_dir= os.path.join(logdir,task_name, 'images')
        # print(img_dir)
        # assert(0)
        text_dir= os.path.join(logdir,task_name, 'text.txt')
        print('episode_len',episode_length)
        with open(text_dir, 'a') as f:
            f.write('episode_len: '+str(episode_length)+'\n')

        pos,rot=goto_original_pose()
        act_result = ActResult(np.concatenate([pos,rot,[1.0],[0.0]]))
        #print("act_result: ",act_result.action)
        transition = env.step(act_result)
        obs = dict(transition.observation)
        if rotmat_flag:
            human_frame_dir='/new/algo/user8/lixiaoqi/heng/keyframe6'
            Name_Catlist = {'put_rubbish_in_bin': 'put_in_rubbishbins_frames', 'stack_blocks': 'stack_box_frames', 'put_toilet_roll_on_stand': 'put_toilet_roll_on_stand_frames', 'slide_block_to_target': 'slide_box_to_target_frames_check', 'water_plants': 'water_plants_frames_check', 'sweep_to_dustpan': 'sweep_to_dustpan_frames'}
            Name_Catlist.update({'close_box':'close_box_frames_check','close_drawer':'close_drawer_frames_check','close_fridge':'close_fridge_frames_check','close_jar':'close_jar_frames_check','close_laptop_lid':'close_laptop_lid_frames_check','close_microwave':'close_microwave_frames_check','lamp_on':'lamp_on_frames_check','open_door':'open_door_frames','open_fridge':'close_fridge_frames_check','open_microwave':'open_microwave_frames','place_wine_at_rack_location':'rack_location_frames_check','push_button':'push_button_frames','take_usb_out_of_computer':'take_usb_out_of_computer_frames_check','toilet_seat_down':'toilet_seat_down_1_frames_check','turn_tap':'turn_tap_frames','unplug_charger':'unplug_charger_frames'})
        
        
            rot_json_list = []
            for root, dirs, files in os.walk(human_frame_dir):
                for dir_name in dirs:
                    if 'extract' in dir_name and 'hmp' not in dir_name and Name_Catlist[task_name] in root:
                        for file in os.listdir(os.path.join(root,dir_name)):
                            if file.endswith('rot.json'):
                                rot_json_list.append(os.path.join(root,dir_name,file))
            rot_json_list.sort(key=lambda x: int(re.search(r'(\d+)_extract', x).group(1)))
            # print(rot_json_list)
            # assert(0)
            human_frame_len=len(rot_json_list)

        world_point = None
        top_left = None
        bottom_right = None
        rotation_array =None
        # model = None
        for step in range(episode_length):
            print('step:', step)
            #print('has_init_task  ',env._task._scene._has_init_task)
            #print('has_init_episode ', env._task._scene._has_init_episode)
            #print(obs.keys())
            with open(text_dir, 'a') as f:
                f.write('step: '+str(step)+'\n')
            rgb = obs['front_rgb']
            
            rgb=np.transpose(rgb, (1,2,0))

            #print(rgb.shape)
            
            #prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
            #prepped_data = {k:torch.tensor([v], device=env_device) for k, v in obs_history.items()}

            ##act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
            if step == 0:
                depth_info_0 = [obs['front_depth'],obs['misc']['front_camera_extrinsics'],obs['misc']['front_camera_intrinsics']]
            depth_info = [obs['front_depth'],obs['misc']['front_camera_extrinsics'],obs['misc']['front_camera_intrinsics']]
            if 'rope' in task_name or 'plate' in task_name:
                depth_info = depth_info_0
            if aff_flag:
                
                # depth_m = obs['front_depth'][0,:,:]
                point_cloud = generate_pointcloud(obs['front_depth'])
            
            answer,idx,prompt,world_point,top_left,bottom_right = generate_answer(llama_dir, adapter_dir, rgb, prompt,img_dir,box_flag,box_flag_2,task_name_2,eval_demo_seed,step,aff_flag,target_flag,depth_info,world_point,top_left,bottom_right,depth_info_0,point_cloud)
            
            print(prompt)
            print("******",answer,"******")
            # assert(0)
            with open(text_dir, 'a') as f:
                f.write(prompt+'\n')
                f.write("******"+answer+"******"+'\n')
            #if eval_cfg.framework.dimensions==3:
            #    act_result,print_flag = deal_answer(answer,obs)
            #elif eval_cfg.framework.dimensions==2:
            #print(1111)
            with open(os.path.join(img_dir, "pred_{task_name}_{idx1}_{idx2}.json".format(task_name=task_name_2,idx1=eval_demo_seed,idx2=step)), 'w') as fout:
                json.dump(prompt+'\n'+answer, fout)
            
            act_result,print_flag,flag_2d,open_status_array,rotation_array =deal_answer(answer,obs,idx,img_dir,rgb,eval_demo_seed,step,task_name_2,rotation_array,act_result)
            
            #print(2222)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            if flag_2d==1:
                # if 'slide' in task_name:
                #     act_result.action[7] = '1.0'
                    # print(act_result.action)
                # assert(0)
                pre_act=prepose_before_contact(act_result.action[:3],act_result.action[3:7])
                if not isinstance(pre_act,np.ndarray):
                    pre_act=[np.array(pre_act)]
                pre_act_result=ActResult(np.concatenate([prepose_before_contact(act_result.action[:3],act_result.action[3:7]),act_result.action[3:7],[1.0],[0.0]]))
                print('--------', prepose_before_contact(act_result.action[:3],act_result.action[3:7]))
                transition = env.step(pre_act_result)
                # act_result = pre_act_result
                obs_pre = dict(transition.observation)
                rgb = obs_pre['front_rgb']
                rgb=np.transpose(rgb, (1,2,0))
                img = Image.fromarray(rgb)
                img.save(os.path.join(img_dir, "img_{task_name}_{idx1}_{idx2}_pre.png".format(task_name=task_name_2,idx1=eval_demo_seed,idx2=step)))
                #print('extra step')
            # print(obs_pre)
                if int(dict(transition.observation)['gripper_open']) == 1 and int(act_result.action[7]) == 0:
                    # print('1111111111111111111')
                    post_pos_before_contact = postpose_before_contact(act_result.action[:3],act_result.action[3:7],task_name)
                    print('--------', post_pos_before_contact)
                    # assert(0)
                    transition = env.step(ActResult(np.concatenate([post_pos_before_contact,act_result.action[3:7],open_status_array,[1.0]])))
                    act_result = ActResult(np.concatenate([post_pos_before_contact,act_result.action[3:7],open_status_array,[1.0]]))
                    obs_pre2 = dict(transition.observation)
                    rgb = obs_pre2['front_rgb']
                    rgb=np.transpose(rgb, (1,2,0))
                    img = Image.fromarray(rgb)
                    img.save(os.path.join(img_dir, "img_{task_name}_{idx1}_{idx2}_pre2.png".format(task_name=task_name_2,idx1=eval_demo_seed,idx2=step)))
                    # transition = env.step(act_result)
                else:
                    # print('2222222222222222222222222')
                    # if 'slide' in task_name:
                    #     post_pos_before_contact = postpose_before_contact(act_result.action[:3],act_result.action[3:7],task_name)
                    #     # print('--------', post_pos_before_contact)
                    #     # assert(0)
                    #     transition = env.step(ActResult(np.concatenate([post_pos_before_contact,act_result.action[3:7],[1.0],[1.0]])))
                    #     act_result = ActResult(np.concatenate([post_pos_before_contact,act_result.action[3:7],[1.0],[1.0]]))
                    transition = env.step(act_result)
                    # act_result = pre_act_result
            else:
                print('3333333333333333333333333')
                transition = env.step(act_result)
            #print(3333)
            
            obs_tp1 = dict(transition.observation)
            
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            #for k in obs_history.keys():
            #    obs_history[k].append(transition.observation[k])
            #    obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)
            #print('terminal:', transition.terminal)
            #print('timeout:', timeout)

            if transition.terminal or timeout:
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            pose_current = act_result.action
            position=discretize(act_result.action[:3])
            rotation=discretize(act_result.action[3:7])
            # human_flag = None
            rotmat = quaternion_to_rotation_matrix(pose_current[3:7])
            rot_z = rotmat[:, 2]
            rot_y = rotmat[:, 1]
            if rotmat_flag:
                # print(task_name,task_name_2)
                human_frame_dir=os.path.join(human_frame_dir,Name_Catlist[task_name])
                print(human_frame_dir)
                print(rot_json_list)
                if human_frame_idx < human_frame_len-1:
                    
                    # print(rot_json_list[human_frame_idx],rot_json_list[human_frame_idx+1])
                    current_human_json = rot_json_list[human_frame_idx]
                    target_human_json = rot_json_list[human_frame_idx+1]
                    if 'box' in task_name and '0025' in current_human_json:
                        target_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/close_box_frames_check/0060_extract/0060_rot.json'
                        current_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/close_box_frames_check/0050_extract/0050_rot.json'
                    elif 'box' in task_name and '0025' not in current_human_json:
                        target_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/close_box_frames_check/0050_extract/0050_rot.json'
                        current_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/close_box_frames_check/0025_extract/0025_rot.json'
                    elif 'laptop' in task_name and '0000' in current_human_json:
                        # assert(0)
                        target_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/close_laptop_lid_frames_check/0008_extract/0008_rot.json'
                        current_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/toilet_seat_down_1_frames_check/0000_extract/0000_rot.json'
                    elif 'toilet' in task_name and human_frame_idx==2:
                        target_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/toilet_seat_down_1_frames_check/0021_extract/0031_rot.json'
                        current_human_json = '/new/algo/user8/lixiaoqi/heng/keyframe6/toilet_seat_down_1_frames_check/0000_extract/0000_rot.json'
                    with open(current_human_json) as f:
                        current_human_rot = json.load(f)
                    with open(target_human_json) as f:
                        target_human_rot = json.load(f)
                    delta_z_rot = np.array(target_human_rot['z_rot']) - np.array(current_human_rot['z_rot'])
                    delta_y_rot = np.array([0,0,0])
                    if 'dustpan' in task_name:
                        delta_z_rot[2] = 0.03
                    elif 'fridge' in task_name and 'close' in task_name:
                        delta_z_rot[2] = 0
                    elif 'fridge' in task_name and 'open' in task_name:
                        delta_z_rot = np.array([0.7,-0.7,0])
                    elif 'light' in task_name:
                        delta_z_rot[0] = 0.12
                    elif 'box' in task_name and delta_z_rot[1] == 52:
                        delta_z_rot[1]=0
                    print('1111111111111', rot_json_list[human_frame_idx],rot_json_list[human_frame_idx+1], delta_z_rot)
                elif human_frame_idx >= human_frame_len-1 and human_frame_len > 1:
                    # with open(rot_json_list[-2]) as f:
                    #     current_human_rot = json.load(f)
                    # with open(rot_json_list[-1]) as f:
                    #     target_human_rot = json.load(f)
                    # human_frame_idx = 0
                    # if 'water' in task_name or 'dustpan' in task_name:
                    delta_z_rot = np.array([0,0,0])
                    delta_y_rot = np.array([0,0,0])
                else:
                    delta_z_rot = np.array([0,0,0])
                    delta_y_rot = np.array([0,0,0])
                human_frame_idx += 1
                delta_quat,_ = rotation_matrix_to_quaternion2(rotation, rot_z+delta_z_rot,rot_y+delta_y_rot)
                delta_quat = discretize(delta_quat)
                delta_quat = ', '.join(map(str, delta_quat))
            open_status=act_result.action[7]
            #if print_flag==1:
            #    print("gripper 3d position:   ",obs['gripper_position'])

            #print(obs['joint_velocities'])
            v=np.linalg.norm(obs['joint_velocities'])
            force_norm = np.linalg.norm(obs['gripper_touch_forces'])
            
            if force_norm<0.01 and abs(obs['joint_forces'][6])<0.08:
                touch_idx+=1
            if ((force_norm<0.01 and abs(obs['joint_forces'][6])<0.08) and touch_idx>=3) or ('rope' in task_name and step ==1):
                touch_idx=0
                print('-------------go to original pose--------------')
                print("force_norm: ",force_norm)
                print("joint_force: ",obs['joint_forces'][6])
                with open(text_dir, 'a') as f:
                    f.write("No force,using 2d"+'\n')
                    f.write("force_norm: "+str(force_norm) +" joint_force: "+str(obs['joint_forces'][6])+ '\n')
                if task_name_flag:
                    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format_map({'task': task_name_blank})
                else:
                    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format_map({'task': task_name_2.split(' ')[0]})
                box_flag_2 = True
                pos,rot=goto_original_pose()
                act_result = ActResult(np.concatenate([pos,rot,[1.0],[0.0]]))
                #print("act_result: ",act_result.action)
                transition = env.step(act_result)
                obs = dict(transition.observation)
                if rotmat_flag:
                    human_frame_idx=0
            #elif obs['joint_velocities'][6]<0.001:
            #    print(2)
            #    print("v: ",obs['joint_velocities'][6])
            #    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format_map({'task': get_task_name()})
            #    box_flag_2 = True                
            else:
                # print(task_name_flag,rotmat_flag)
                # assert(0)
                print("force_norm: ",force_norm)
                print("joint_force: ",obs['joint_forces'][6])
                print("v: ",obs['joint_velocities'][6])
                with open(text_dir, 'a') as f:
                    f.write("Continue using 3d"+'\n')
                    f.write("force_norm: "+str(force_norm) +" joint_force: "+str(obs['joint_forces'][6])+ "v: "+ str(obs['joint_velocities'][6])+ '\n')
                if task_name_flag:
                    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current gripper position is [{pos_str}]. The current gripper rotation is [{rot_str}]. The current gripper state is {gripper_open_now}.".format_map({'task': task_name_blank, 'pos_str': ', '.join(map(str, position)), 'rot_str': ', '.join(map(str, rotation)), 'gripper_open_now': open_status})
                    if rotmat_flag:
                        prompt = "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current gripper position is [{pos_str}]. The current gripper z-axis rotation and y-axis direction is [{z_rot}] and [{y_rot}]. The current gripper state is {gripper_open_now}. The z and y axis delta rotation in 2D human video demonstrate is [{delta_z}] and [{delta_y}].".format_map({'task': task_name_2, 'pos_str': ', '.join(map(str, position)), 'z_rot': ', '.join(map(str, rot_z)), 'y_rot': ', '.join(map(str, rot_y)), 'gripper_open_now': open_status, 'delta_z': ', '.join(map(str, delta_z_rot)), 'delta_y': ', '.join(map(str, delta_y_rot))})
                else:
                    # assert(0)
                    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current gripper position is [{pos_str}]. The current gripper rotation is [{rot_str}]. The current gripper state is {gripper_open_now}.".format_map({'task': task_name_2.split(' ')[0], 'pos_str': ', '.join(map(str, position)), 'rot_str': ', '.join(map(str, rotation)), 'gripper_open_now': open_status})
                    if rotmat_flag:
                        # assert(0)
                        prompt += ' The delta quaternion of next step and current is [{}]'.format(delta_quat)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
