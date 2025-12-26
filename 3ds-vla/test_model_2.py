from argparse import ArgumentParser
import torch
import llama
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
# import cv2
import sys
import json
import numpy as np
import torch.nn as nn
import random
import pickle
from pyrep.objects import VisionSensor
import os

sys.path.append(os.path.join(os.getcwd(),'Grounded-Segment-Anything/'))
sys.path.append(os.path.join(os.getcwd(),'3ds-vla/YARR/yarr/utils'))
# sys.path.append('/new/algo/user8/lixiaoqi/cloris-2/RLBench')
from pprint import pprint

from load_pickle import depth_to_depthscale,worldpoint_to_pixel,pixel_to_worldpoint2,find_nearest_segmentation_pixel_in_circle_target
from grounded_sam_demo_test import generate_mask,generate_target_mask


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

file_count = 0
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = None
transform_train = transforms.Compose([
    transforms.Resize(size=(336, 336), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

def discretize_1(q):
    q = [element / 0.01 for element in q]
    q = [round(element) for element in q]
    return q

def load_model(aff_flag, adapter_dir, llama_dir):
    print(llama_dir,adapter_dir)
    global model
    if model is None:
        
        # base_dir = '/'.join(os.getcwd().split('/')[:4])
        model, preprocess = llama.load(adapter_dir, llama_dir, aff_flag,device)
        model.to(device)
        model.eval()
    return model

def generate_answer_dual(llama_dir, adapter_dir, rgb, prompt,img_dir,task_name,seed,step,depth_info = None,world_point=None, top_left=None,bottom_right=None,depth_info_0=None,point_cloud=None):
    global model
    global file_count
    
    # if model == None:
    box_flag = True
    box_flag_2 = True
    target_flag = True
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "img_{task_name}_{idx1}_{idx2}.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
    
    img=Image.fromarray(rgb)
    img.save(img_path)
    
    
    if box_flag and box_flag_2 and 'current' not in prompt:
        top_left, bottom_right = generate_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step)
        draw_robot_img_path=os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_cbox.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
        if 'rope' not in task_name:
            top_left_str = ', '.join([str(x) for x in top_left])
            bottom_right_str = ', '.join([str(x) for x in bottom_right])
            prompt = prompt + " The position is within or near the bounding box: [{top_left}], [{bottom_right}].".format(top_left=top_left_str, bottom_right=bottom_right_str)
        else:
            top_left_str_1 = ', '.join([str(x) for x in top_left[0]])
            bottom_right_str_1 = ', '.join([str(x) for x in bottom_right[0]])
            top_left_str_2 = ', '.join([str(x) for x in top_left[1]])
            bottom_right_str_2 = ', '.join([str(x) for x in bottom_right[1]])
            prompt = prompt + " The position is within or near the bounding box: [{top_left}], [{bottom_right}],[{top_left2}], [{bottom_right2}].".format(top_left=top_left_str_2, bottom_right=bottom_right_str_2,top_left2=top_left_str_1, bottom_right2=bottom_right_str_1)
            # print(prompt)
            # assert(0)
        if 'slide' in task_name and 'current' not in prompt:
            
            target_top_left, target_bottom_right = generate_target_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step)
           
            target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
            target_center_2d.reverse()
            draw = ImageDraw.Draw(img)
            draw.ellipse((target_center_2d[0] - 3, target_center_2d[1] - 3, target_center_2d[0] + 3, target_center_2d[1] + 3), fill='red')
            img.save(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)))
            
            prompt = prompt + 'The target is opposite to: {target_pixel}'.format(target_pixel=[target_center_2d[1],target_center_2d[0]])
            if 'rope' in task_name:
                depth_info = depth_info_0
            death_in_meters=np.transpose(depth_info[0], (1,2,0))
            death_in_meters=np.squeeze(death_in_meters)
            world_point = discretize_1(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[0],target_center_2d[1]))
            
    if target_flag and 'current' in prompt:
        # world_point = None
        print('======================',task_name,'_'.join(task_name.split(' ')),world_point)
        if '_'.join(task_name.split(' ')) in ['bimanual_straighten_rope','bimanual_push_box','beat_the_buzz','straighten_rope','put_the_plate_between_the_red_pillars_of_the_dish_rack', 'put_the_knife_on_the_chopping_board','pick_up_the_hanger_and_place_in_on_the_rackput_the_hanger_on_the_rack','put_the_phone_on_the_base', 'put_rubbish_in_bin','put_toilet_roll_on_stand','slide_the_block_to_target','water_plant','sweep_dirt_to_dustpan','stack_the_wine_bottle_to_the_middle_of_the_rack'] or '_blocks' in '_'.join(task_name.split(' ')) or '_cup' in '_'.join(task_name.split(' ')):
            if world_point == None and ('rope' not in task_name):
                target_top_left, target_bottom_right = generate_target_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step,top_left,bottom_right)
                # target_top_left = [target_top_left[1],target_top_left[0]]
                # target_bottom_right = [target_top_left[1],target_top_left[0]]
                target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
                target_center_2d.reverse()
                draw = ImageDraw.Draw(img)
                draw.ellipse((target_center_2d[0] - 3, target_center_2d[1] - 3, target_center_2d[0] + 3, target_center_2d[1] + 3), fill='red')
                img.save(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)))
                # assert(0)
                # if 'plate' not in task_name:
                target_center_2d = find_nearest_segmentation_pixel_in_circle_target(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)), target_center_2d, radius=20)
                # if 'rope' in task_name:
                #     depth_info = depth_info_0
                # death_in_meters=np.transpose(depth_info[0], (1,2,0))
                # death_in_meters=np.squeeze(death_in_meters)
                world_point = discretize_1(depth_info[0][:,target_center_2d[1], target_center_2d[0]])
                if 'plate' in task_name:
                    world_point[2] = 84
                # print(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[1],target_center_2d[0]),pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[0],target_center_2d[1]))
                # assert(0)
                draw_robot_img_path=os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_cbox.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
                prompt = prompt + "  The target position is within or near the point: {target_point}.".format(target_point=world_point)
            if world_point == None and ('rope' in task_name):
                target_top_left, target_bottom_right = generate_target_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step,top_left,bottom_right)
                target_center_2d_1 = [(target_top_left[0][0]+target_bottom_right[0][0])//2,(target_top_left[0][1]+target_bottom_right[0][1])//2]
                target_center_2d_2 = [(target_top_left[1][0]+target_bottom_right[1][0])//2,(target_top_left[1][1]+target_bottom_right[1][1])//2]
                
                target_center_2d_1.reverse()
                target_center_2d_2.reverse()
                draw = ImageDraw.Draw(img)
                draw.ellipse((target_center_2d_1[0] - 3, target_center_2d_1[1] - 3, target_center_2d_1[0] + 3, target_center_2d_1[1] + 3), fill='red')
                draw.ellipse((target_center_2d_2[0] - 3, target_center_2d_2[1] - 3, target_center_2d_2[0] + 3, target_center_2d_2[1] + 3), fill='red')
                world_point1 = discretize_1(depth_info_0[0][:,target_center_2d_1[1], target_center_2d_1[0]])
                world_point2 = discretize_1(depth_info_0[0][:,target_center_2d_2[1], target_center_2d_2[0]])
                img.save(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)))
                prompt = prompt + "  The target position is within or near the point: {target_point}, {target_point2}.".format(target_point=world_point1,target_point2=world_point2)
            
    file_count += 1
    
    model = load_model(True, adapter_dir, llama_dir)
    img = transform_train(img.convert('RGB')).unsqueeze(0).to(device)
    prompt = llama.format_prompt(prompt)
    # result = 'The left gripper position is [467, 577], the left gripper rotation quaternion is [-70, 9, 9, -70], the left gripper open status is 0.0. The right gripper position is [484, 402], the right gripper rotation quaternion is [9, 70, 70, 9], the right gripper open status is 0.0.'
    
    with torch.no_grad():
        result = model.generate(img, [prompt],point_cloud)[0]
    return result,file_count-1,prompt,world_point,top_left,bottom_right

def generate_answer(llama_dir, adapter_dir, rgb, prompt,img_dir,box_flag,box_flag_2,task_name,seed,step,aff_flag,target_flag,depth_info = None,world_point=None, top_left=None,bottom_right=None,depth_info_0=None,point_cloud=None):
    global model
    global file_count
    # if model == None:
    
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "img_{task_name}_{idx1}_{idx2}.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
    print(img_path)
    
    #os.mkdir(img_dir)
    img=Image.fromarray(rgb)
    img.save(img_path)
    # assert(0)
    model = load_model(aff_flag, adapter_dir, llama_dir)
    if box_flag and box_flag_2 and 'current' not in prompt:
        top_left, bottom_right = generate_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step)
        draw_robot_img_path=os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_cbox.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
        # if aff_flag:
        #     #print(1)
        #     hand_frame_dict = get_match_img(img_path,task_name,step==0)
        #     if hand_frame_dict is not None:
        #         #print(2)
        #         print(hand_frame_dict)
        #         if 'human_dir_n+1' in hand_frame_dict.keys() and os.path.exists(draw_robot_img_path):
        #             print(3)
        #             human_img_path=hand_frame_dict['human_dir_n+1']
        #             aff_position=get_aff_box(draw_robot_img_path,human_img_path)
        #             top_left=[aff_position[0][1],aff_position[0][0]]
        #             bottom_right=[aff_position[0][3],aff_position[0][2]]
        top_left_str = ', '.join([str(x) for x in top_left])
        bottom_right_str = ', '.join([str(x) for x in bottom_right])
        prompt = prompt + " The position is within or near the bounding box: [{top_left}], [{bottom_right}].".format(top_left=top_left_str, bottom_right=bottom_right_str)
        if 'slide' in task_name and 'current' not in prompt:
            
            target_top_left, target_bottom_right = generate_target_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step)
            # target_top_left = [target_top_left[1],target_top_left[0]]
            # target_bottom_right = [target_top_left[1],target_top_left[0]]
            target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
            target_center_2d.reverse()
            draw = ImageDraw.Draw(img)
            draw.ellipse((target_center_2d[0] - 3, target_center_2d[1] - 3, target_center_2d[0] + 3, target_center_2d[1] + 3), fill='red')
            img.save(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)))
            # assert(0)
            # target_center_2d = find_nearest_segmentation_pixel_in_circle_target(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)), target_center_2d, radius=20)
            
            prompt = prompt + 'The target is opposite to: {target_pixel}'.format(target_pixel=[target_center_2d[1],target_center_2d[0]])
            if 'rope' in task_name:
                depth_info = depth_info_0
            death_in_meters=np.transpose(depth_info[0], (1,2,0))
            death_in_meters=np.squeeze(death_in_meters)
            world_point = discretize_1(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[0],target_center_2d[1]))
            # world_point2 = discretize_1(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[1],target_center_2d[0]))
            # print(world_point,world_point2)
            # assert(0)
    if target_flag and 'current' in prompt:
        # world_point = None
        print('======================',task_name,'_'.join(task_name.split(' ')))
        if '_'.join(task_name.split(' ')) in ['beat_the_buzz','straighten_rope','put_the_plate_between_the_red_pillars_of_the_dish_rack', 'put_the_knife_on_the_chopping_board','pick_up_the_hanger_and_place_in_on_the_rackput_the_hanger_on_the_rack','put_the_phone_on_the_base', 'put_rubbish_in_bin','put_toilet_roll_on_stand','slide_the_block_to_target','water_plant','sweep_dirt_to_dustpan','stack_the_wine_bottle_to_the_middle_of_the_rack'] or '_blocks' in '_'.join(task_name.split(' ')) or '_cup' in '_'.join(task_name.split(' ')):
            if world_point == None or ('rope' in task_name):
                target_top_left, target_bottom_right = generate_target_mask(img_path,img_dir,file_count,'_'.join(task_name.split(' ')),seed,step,top_left,bottom_right)
                # target_top_left = [target_top_left[1],target_top_left[0]]
                # target_bottom_right = [target_top_left[1],target_top_left[0]]
                target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
                target_center_2d.reverse()
                draw = ImageDraw.Draw(img)
                draw.ellipse((target_center_2d[0] - 3, target_center_2d[1] - 3, target_center_2d[0] + 3, target_center_2d[1] + 3), fill='red')
                img.save(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)))
                # assert(0)
                # if 'plate' not in task_name:
                target_center_2d = find_nearest_segmentation_pixel_in_circle_target(os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_point.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step)), target_center_2d, radius=20)
                # if 'rope' in task_name:
                #     depth_info = depth_info_0
                death_in_meters=np.transpose(depth_info[0], (1,2,0))
                death_in_meters=np.squeeze(death_in_meters)
                world_point = discretize_1(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[0],target_center_2d[1]))
                if 'plate' in task_name:
                    world_point[2] = 84
                # print(pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[1],target_center_2d[0]),pixel_to_worldpoint2(death_in_meters, depth_info[1],depth_info[2], target_center_2d[0],target_center_2d[1]))
                # assert(0)
                draw_robot_img_path=os.path.join(img_dir, "image_{task_name}_{idx1}_{idx2}_target_cbox.png".format(task_name='_'.join(task_name.split(' ')),idx1=seed,idx2=step))
            
            prompt = prompt + "  The target position is within or near the point: {target_point}.".format(target_point=world_point)
    file_count += 1
    img = transform_train(img.convert('RGB')).unsqueeze(0).to(device)
    prompt = llama.format_prompt(prompt)
    with torch.no_grad():
        if aff_flag:
            result = model.generate(img, [prompt],point_cloud)[0]
        else:
            result = model.generate(img, [prompt])[0]
    return result,file_count-1,prompt,world_point,top_left,bottom_right

def worldpoint_to_pixel(data_item):
    camera_extrinsics = data_item.misc['front_camera_extrinsics']
    camera_intrinsics = data_item.misc['front_camera_intrinsics']

    point_world = np.ones(4)
    point_world[:3] = data_item.gripper_pose[:3]
    point_cam =  np.linalg.inv(camera_extrinsics) @ point_world
    # print(final_postion_cam)
    # final_postion_cam2 = [-point_cam[1],-point_cam[2],final_postion_cam[0]]
    pixel_xy = np.dot(camera_intrinsics[:3,:3],point_cam[:3])
    pixel_xy = (pixel_xy/pixel_xy[2])[:2] # (x,y,r^2,r)
    final_pixel = (int(pixel_xy[0]),  int(pixel_xy[1]))
    print(final_pixel)
    # assert(0)
    return final_pixel

if __name__=="__main__":
    adapter_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/exp/exp-902-rlbench_dataset_0902_touch2d_4/checkpoint-7.pth"
    llama_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/xcy/llama_model_weights"
    #model=load_model("/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/adapter","/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/llama")
    model=load_model(adapter_dir,llama_dir)
    root_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672"
    data_lists=os.listdir(root_dir)
    idx=0
    for data in data_lists:
        data_dir=os.path.join(root_dir,data,"front_rgb")
        images=os.listdir(data_dir)
        
        for image in images:
            img_dir=os.path.join(data_dir,image)
            if "segmask" in img_dir and "0_0" in img_dir:
                img=Image.open(img_dir)
                #variation_dir=os.path.join(root_dir,data,"variation_descriptions.pkl")
                #with open(variation_dir, "rb") as f:
                #    variation = pickle.load(f)
                #prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.".format(task=variation[0])
                prompt_dir=os.path.join(root_dir,data,"prompt.json")
                with open(prompt_dir, "r") as f:
                    prompt = json.load(f)
                img = transform_train(img.convert('RGB')).unsqueeze(0).to(device)
                prompt = llama.format_prompt(prompt)
                with torch.no_grad():
                    result = model.generate(img, [prompt])[0]
                print(result)
                if "position is [" in result:
                    position = [int(x) for x in result.split("position is [")[1].split("]")[0].split(",")]
                    if len(position)==2:
                        position.reverse()
                        #output_dir=os.path.join(root_dir,data,"front_rgb","draw_segmask.png")
                        output_dir=os.path.join(root_dir,"outputs")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, "img_{idx}.png".format(idx=idx))
                        idx+=1
                        #img_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/output_laptop/0/raw_image.jpg"
                        img=Image.open(img_dir)
                        draw = ImageDraw.Draw(img)
                        radius = 5
                        draw.ellipse((position[0]-radius, position[1]-radius, position[0]+radius, position[1]+radius), fill=(255,0,0))
                        img.save(output_path)
    '''img_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/2024-8-19-4-15-10-0-23/front_rgb/0_0_little green light switch_segmask.png"
    img=Image.open(img_dir)
    prompt="Determine the gripper's position, orientation, and operational state for the next step in the task: turn on the light."
    img = transform_train(img.convert('RGB')).unsqueeze(0).to(device)
    prompt = llama.format_prompt(prompt)
    with torch.no_grad():
        result = model.generate(img, [prompt])[0]
    print(result)
    if "position is [" in result:     
        position = [int(x) for x in result.split("position is [")[1].split("]")[0].split(",")]
        if len(position)==2:
            position.reverse()
            output_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/2024-8-19-4-15-10-0-23/front_rgb/draw_segmask.png"
            #img_dir="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/output_laptop/0/raw_image.jpg"
            img=Image.open(img_dir)
            draw = ImageDraw.Draw(img)
            radius = 5
            draw.ellipse((position[0]-radius, position[1]-radius, position[0]+radius, position[1]+radius), fill=(255,0,0))
            img.save(output_dir)'''


