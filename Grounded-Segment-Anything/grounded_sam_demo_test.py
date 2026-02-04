import argparse
import os
import sys
import json
import numpy as np
import json
import torch
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw, ImageOps
import tqdm
sys.path.append(os.path.join(os.getcwd(),'Grounded-Segment-Anything/GroundingDINO/'))
sys.path.append(os.path.join(os.getcwd(),"Grounded-Segment-Anything/segment_anything/"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_seg_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)
def box_mask(image, dir_path,image_ori,dir_path2):
    gray_image = ImageOps.grayscale(image)
    # gray_image.save('./binary.png')
    # Convert grayscale image to numpy array
    gray_np = np.array(gray_image)

    # Apply a binary threshold
    binary_np = np.where(gray_np < 250, 1, 0).astype(np.uint8)

    # Find the coordinates of the bounding box
    non_zero_coords = np.column_stack(np.where(binary_np > 0))
    top_left = non_zero_coords.min(axis=0) #!!!!
    top_left_2 = [top_left[1],top_left[0]]
    bottom_right = non_zero_coords.max(axis=0) #!!!!
    bottom_right_2 = [bottom_right[1],bottom_right[0]]

    # Draw the bounding box on the original image
    draw = ImageDraw.Draw(image)
    draw.rectangle([tuple(top_left_2), tuple(bottom_right_2)], outline="red", width=2)
    image.save(dir_path)

    draw = ImageDraw.Draw(image_ori)
    draw.rectangle([tuple(top_left_2), tuple(bottom_right_2)], outline="red", width=2)
    image_ori.save(dir_path2)
    return np.array(top_left).tolist(),np.array(bottom_right).tolist()
    #json_data = {'top_left':np.array(top_left_2).tolist(), 'bottom_right':np.array(bottom_right_2).tolist()}
    #with open(dir_path.split('.')[0]+'.json', 'w') as f:
    #    json.dump(json_data, f)



def masked_image(mask, image, out_dir, rgb_id, category,seed,step,task_name):
    #print(mask.squeeze().squeeze().cpu().numpy().shape)
    mask_reshape = mask.squeeze().squeeze().cpu().numpy()
    image_pil=Image.fromarray(image)
    if len(mask_reshape.shape) == 2:
        mask_np = mask_reshape
        background_color = (255, 255, 255)
        new_image = np.full_like(image, background_color)
        new_image[mask_np] = image[mask_np]
        new_image_pil = Image.fromarray(new_image)
        new_image_pil2 = new_image_pil
        top_left,bottom_right=box_mask(new_image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_bbox.png".format(task_name=task_name,idx1=seed,idx2=step)),image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_cbox.png".format(task_name=task_name,idx1=seed,idx2=step)))
        new_image_pil2.save(os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_seg.png".format(task_name=task_name,idx1=seed,idx2=step)))
        return top_left,bottom_right
    else:
        top_left_list = []
        bottom_right_list = []
        for i in range(mask.squeeze().squeeze().cpu().numpy().shape[0]):
            mask_np = mask.squeeze().squeeze().cpu().numpy()[i,:,:]
            background_color = (255, 255, 255)
            new_image = np.full_like(image, background_color)
            new_image[mask_np] = image[mask_np]
            new_image_pil = Image.fromarray(new_image)
            new_image_pil2 = new_image_pil
            top_left,bottom_right=box_mask(new_image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_bbox_{idx}.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)),image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_{idx}_cbox.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)))
            
            new_image_pil2.save(os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_{idx}_seg.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)))
            top_left_list.append(top_left)
            bottom_right_list.append(bottom_right)
        return top_left_list,bottom_right_list
        # for i in range(mask.squeeze().squeeze().cpu().numpy().shape[0]):
        #     mask_np = mask.squeeze().squeeze().cpu().numpy()[i,:,:]
        #     background_color = (255, 255, 255)
        #     new_image = np.full_like(image, background_color)
        #     new_image[mask_np] = image[mask_np]
        #     new_image_pil = Image.fromarray(new_image)
        #     new_image_pil2 = new_image_pil
        #     top_left,bottom_right=box_mask(new_image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_bbox.png".format(task_name=task_name,idx1=seed,idx2=step)),image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_cbox.png".format(task_name=task_name,idx1=seed,idx2=step)))
            
        #     new_image_pil2.save(os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_seg.png".format(task_name=task_name,idx1=seed,idx2=step)))
        #     if i==0:
        #         return top_left,bottom_right

def masked_image_target(mask, image, out_dir, rgb_id, category,seed,step,task_name):
    #print(mask.squeeze().squeeze().cpu().numpy().shape)
    mask_reshape = mask.squeeze().squeeze().cpu().numpy()
    image_pil=Image.fromarray(image)
    if len(mask_reshape.shape) == 2:
        mask_np = mask_reshape
        background_color = (255, 255, 255)
        new_image = np.full_like(image, background_color)
        new_image[mask_np] = image[mask_np]
        new_image_pil = Image.fromarray(new_image)
        new_image_pil2 = new_image_pil
        top_left,bottom_right=box_mask(new_image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_bbox_target.png".format(task_name=task_name,idx1=seed,idx2=step)),image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_cbox_target.png".format(task_name=task_name,idx1=seed,idx2=step)))
        new_image_pil2.save(os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_seg_target.png".format(task_name=task_name,idx1=seed,idx2=step)))
        return top_left,bottom_right
    else:
        top_left_list = []
        bottom_right_list = []
        for i in range(mask.squeeze().squeeze().cpu().numpy().shape[0]):
            mask_np = mask.squeeze().squeeze().cpu().numpy()[i,:,:]
            background_color = (255, 255, 255)
            new_image = np.full_like(image, background_color)
            new_image[mask_np] = image[mask_np]
            new_image_pil = Image.fromarray(new_image)
            new_image_pil2 = new_image_pil
            top_left,bottom_right=box_mask(new_image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_bbox_{idx}_target.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)),image_pil,os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_{idx}_cbox_target.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)))
            
            new_image_pil2.save(os.path.join(out_dir,"image_{task_name}_{idx1}_{idx2}_{idx}_seg_target.png".format(task_name=task_name,idx1=seed,idx2=step,idx=i)))
            top_left_list.append(top_left)
            bottom_right_list.append(bottom_right)
        return top_left_list,bottom_right_list


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
def rope_endpoint(mask):
    
    # points = np.column_stack(np.where(mask))
    # if points.shape[0] < 2:
    #     raise ValueError("无法检测到足够的绳子像素，检查颜色范围或图像内容")

    # # 寻找两端点（欧几里得距离最大的点对）
    # max_distance = 0
    # endpoints = (points[0], points[0])

    # for i in range(len(points)):
    #     for j in range(i + 1, len(points)):
    #         dist = np.sum((points[i] - points[j]) ** 2)
    #         if dist > max_distance:
    #             max_distance = dist
    #             endpoints = (points[i], points[j])

    # (y1, x1), (y2, x2) = endpoints
    skeleton = skeletonize(mask)
    points = np.column_stack(np.where(skeleton))
    # print(points)
    endpoints = []
    for y, x in points:
        # Extract a 3x3 neighborhood
        neighbors = skeleton[max(0, y-1):y+4, max(0, x-1):x+4]
        # Count neighbors (exclude the point itself)
        num_neighbors = np.sum(neighbors) - skeleton[y, x]
        if num_neighbors == 1:  # Endpoint has exactly one neighbor
            endpoints.append((y, x))
    print(endpoints)
    try:
        return endpoints[0], endpoints[1]
    except:
        return None, None
def generate_mask(img_path, out_dir, rgb_id,task_name,seed,step):
    config='Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint='Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
    device='cpu'
    box_threshold=0.3
    text_threshold=0.25
    use_sam_hq=False
    sam_checkpoint='Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    sam_version='vit_h'
    #CAT_LIST = {'lamp on':'little green light switch', 'turn on the light':'little green light switch', 'close fridge': 'fridge', 'close microwave': 'microwave with the whole body', 'drawer': 'drawer', 'close laptop lid': 'laptop', 'toilet seat': 'toilet seat with two parts', 'close box':'box with two parts', 'put toilet roll on stand': 'the smallest cylindrical toilet roll that is on the box', 'take usb out of computer':'This is a small usb connected to a big computer. Find the small usb.', 'take shoes out of box':'box with two parts', 'lamp on':'little green light switch', 'unplug charger':'charger on vertical board', 'water plants':'watering can','sweep dirt to dustpan':'the dark brown long pole with two parts'}
    CAT_LIST = {'turn_on':'little green light switch','slide_the_block_to_target':'red block','button':'round button','rack':'wine','put_rubbish_in_bin':'the smallest white rubbish at the bottom','open_microwave':'small handle on microwave','jar': 'small grey circle on the table', 'close_fridge': 'fridge', 'close_microwave': 'microwave with the whole body', 'close_laptop_lid': 'laptop', 'toilet_seat_down': 'toilet seat with two parts', 'close_box':'box with two parts', 'put_toilet_roll': 'the smallest cylindrical toilet roll that is on the box', 'take_usb_out_of_computer':'This is a small usb connected to a big computer. Find the small usb.', 'unplug_charger':'the smallest black charger on vertical board', 'water_plants':'watering can','dustpan':'the dark brown long pole with two parts'}
    CAT_LIST.update({'bimanual_straighten_rope':'red rope','bimanual_lift_ball':'ball','bimanual_pick_laptop':'laptop','bimanual_push_box':'box','bimanual_lift_tray':'tray','straighten_rope':'red rope','put_the_phone_on_the_base':'small white phone near the red base on the desk','place_hanger_on_rack':'handle of hanger','press_switch':'switch button', 'put_the_knife_on_the_chopping_board':'knife','put_plate_in_colored_dish_rack':'white dish','take_frame_off_hanger':'frame','remove_the_green_pepper_from_the_weighing_scales_and_place_it_on_the_table':'green pepper','take_umbrella_out_of_umbrella_stand':'umbrella_handle','beat_the_buzz':'orange handle','change_the_clock_to_show_time_12.15':'wooden button'})
    # CAT_LIST.update({'blocks': task_name.split('_')[2]+' block','jar':'small grey circle on the table','push_the_maroon_button':'round_button','wine':'wine','turn_on_the_light':'little green light switch'})
    task_name = task_name.replace(' ','_')
    print(task_name)
    if 'stack' in task_name and 'blocks' in task_name:
        category = task_name.split('_')[2] + 'block'
    elif 'drawer' in task_name:
        category = task_name.split('_')[1] + ' drawer'
    elif 'cup' in task_name:
        category = task_name.split('_cup')[0].split('_')[-1] + ' cup'
    elif 'tap' in task_name:
        category = task_name.split('_')[1] + 'cross-shaped cross on faucet'
    else:
        for task in list(CAT_LIST.keys()):
            if task.lower() in task_name.lower() or task_name.lower() in task.lower():
                #print(task)
                category = CAT_LIST[task]
    #print(category)
    model=load_seg_model(config,checkpoint,device)
    # load image
    
    image_pil, image = load_image(img_path)
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, category, box_threshold, text_threshold, device=device
    )
    sorted_indices = sorted(range(len(pred_phrases)), key=lambda k: pred_phrases[k], reverse=True)

    # Step 2: 使用排序后的索引重新排列 pred_phrases 和 boxes_filt
    pred_phrases = [pred_phrases[i] for i in sorted_indices]
    boxes_filt = boxes_filt[sorted_indices]
    # print(boxes_filt, pred_phrases,category)


    if 'usb' in category:
        for i,item in enumerate(pred_phrases):
            if 'usb' in item:
                # print('-------------')
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    # print(sorted(pred_phrases, reverse=True))
    # assert(0)
    if 'charger' in category:
        for i,item in enumerate(pred_phrases):
            if 'charger' in item:
                # print('-------------')
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    if 'handle' in category and 'microwave' in category:
        for i,item in enumerate(pred_phrases):

            if 'handle' in item:
                # assert(0)
                # print('-------------')
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    
    if 'grey circle' in category:
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if 'grey circle' in item:
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    # print(boxes_filt)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    print(boxes_filt)
    if 'phone' in category:
        
        print('1111111111111')
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if 'phone' in item and (boxes_filt[i][1] + boxes_filt[i][3])//2 > 336:
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                # print(i)
                # assert(0)
                break
    # if 'rope' in category:
    #     for i,item in enumerate(pred_phrases):

    #         if 'rope' in item  and step ==0:
    #             box_head = torch.Tensor([boxes_filt[i][0],boxes_filt[i][1],boxes_filt[i][0]+30,boxes_filt[i][3]])
    #             boxes_filt = box_head.unsqueeze(0)
    #             pred_phrases = [pred_phrases[i]]
    #             break
    #         if 'rope' in item  and step > 0:
    #             box_head = torch.Tensor([boxes_filt[i][2]-30,boxes_filt[i][1],boxes_filt[i][2],boxes_filt[i][3]])
    #             boxes_filt = box_head.unsqueeze(0)
    #             pred_phrases = [pred_phrases[i]]
    #             break
    if 'orange handle' in category:
        
        print('1111111111111')
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if 'handle' in item and boxes_filt[i][1] > 20 and abs(boxes_filt[i][3]-boxes_filt[i][1])>abs(boxes_filt[i][2]-boxes_filt[i][0]):
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                # print(i)
                # assert(0)
                break
    if 'knife' in category:
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if abs(boxes_filt[i][3]-boxes_filt[i][1])>abs(boxes_filt[i][2]-boxes_filt[i][0]) and boxes_filt[i][1]>10 and abs(boxes_filt[i][2]-boxes_filt[i][0])<110:
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                # print(i)
                # assert(0)
                break
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    if 'rope' in category:
        for i,item in enumerate(pred_phrases):
            if 'rope' in item  and step ==0:
                # print(masks.shape)
                # print(masks[i].shape,masks[i].cpu().numpy().shape)
                endpoint0, endpoint1 = rope_endpoint(masks[i].cpu().numpy()[0])
                print(endpoint0,endpoint1)
                if endpoint0 != None and endpoint1 != None:
                    if 'bimanual' in task_name:
                        boxes_filt = torch.Tensor([[endpoint0[0]-20,endpoint0[1]-20,endpoint0[0]+20,endpoint0[1]+20],[endpoint1[0]-20,endpoint1[1]-20,endpoint1[0]+20,endpoint1[1]+20]])
                        new_mask0 = np.zeros_like(masks[0].cpu().numpy()[0], dtype=bool)
                        new_mask0[endpoint0[0]-20:endpoint0[0]+20, endpoint0[1]-20:endpoint0[1]+20] = masks[0].cpu().numpy()[0][endpoint0[0]-20:endpoint0[0]+20, endpoint0[1]-20:endpoint0[1]+20]
                        

                        new_mask1 = np.zeros_like(masks[0].cpu().numpy()[0], dtype=bool)
                        new_mask1[endpoint1[0]-20:endpoint1[0]+20, endpoint1[1]-20:endpoint1[1]+20] = masks[0].cpu().numpy()[0][endpoint1[0]-20:endpoint1[0]+20, endpoint1[1]-20:endpoint1[1]+20]

                        masks = torch.cat((torch.from_numpy(new_mask0).unsqueeze(0).unsqueeze(0),torch.from_numpy(new_mask1).unsqueeze(0).unsqueeze(0)),dim=0)
                        print(masks.shape)
                        pred_phrases = [pred_phrases[0],pred_phrases[0]]
                        break
                    print('endpoint0, endpoint1',endpoint0, endpoint1)
                    if endpoint0[1]> endpoint1[1]:
                        endpoint = endpoint0
                    else:
                        endpoint = endpoint1
                    box_head = torch.Tensor([endpoint[0]-10,endpoint[1]-10,endpoint[0]+10,endpoint[1]+10])
                    boxes_filt = box_head.unsqueeze(0)
                    new_mask = np.zeros_like(masks[i].cpu().numpy()[0], dtype=bool)
                    print(masks.shape,new_mask.shape)
                    new_mask[endpoint[0]-10:endpoint[0]+10, endpoint[1]-10:endpoint[1]+10] = masks[i].cpu().numpy()[0][endpoint[0]-10:endpoint[0]+10, endpoint[1]-10:endpoint[1]+10]
                    masks[i][0] = torch.from_numpy(new_mask)
                    print(masks.shape)
                    # print(endpoint,boxes_filt)
                    pred_phrases = [pred_phrases[i]]
                    break
                else:
                    print('no endpoint')
                    box_head = torch.Tensor([boxes_filt[i][2]-10,boxes_filt[i][1],boxes_filt[i][2],boxes_filt[i][3]])
                    
                    new_mask = np.zeros_like(masks[i].cpu().numpy()[0], dtype=bool)
                    print(masks.shape,new_mask.shape)
                    new_mask[int(boxes_filt[i][1]):int(boxes_filt[i][3]),int(boxes_filt[i][2])-10:int(boxes_filt[i][2])] = masks[i].cpu().numpy()[0][int(boxes_filt[i][1]):int(boxes_filt[i][3]),int(boxes_filt[i][2])-10:int(boxes_filt[i][2])]
                    boxes_filt = box_head.unsqueeze(0)
                    masks[i][0] = torch.from_numpy(new_mask)
                    pred_phrases = [pred_phrases[i]]
                    print(masks.shape)
                    break
                    
            if 'rope' in item  and step >0:
                endpoint0, endpoint1 = rope_endpoint(masks[i].cpu().numpy()[0])
                print('endpoint0, endpoint1',endpoint0, endpoint1)
                if endpoint0 != None and endpoint1 != None:
                    if endpoint0[1] < endpoint1[1]:
                        endpoint = endpoint0
                    else:
                        endpoint = endpoint1
                    box_head = torch.Tensor([endpoint[0]-10,endpoint[1]-10,endpoint[0]+10,endpoint[1]+10])
                    boxes_filt = box_head.unsqueeze(0)
                    new_mask = np.zeros_like(masks[i].cpu().numpy()[0], dtype=bool)
                    print(masks.shape,new_mask.shape)
                    new_mask[endpoint[0]-10:endpoint[0]+10, endpoint[1]-10:endpoint[1]+10] = masks[i].cpu().numpy()[0][endpoint[0]-10:endpoint[0]+10, endpoint[1]-10:endpoint[1]+10]
                    masks[i][0] = torch.from_numpy(new_mask)
                    pred_phrases = [pred_phrases[i]]
                    print(masks.shape)
                    break
                else:
                    print('no endpoint')
                    box_head = torch.Tensor([boxes_filt[i][0],boxes_filt[i][1],boxes_filt[i][0]+10,boxes_filt[i][3]])
                    
                    new_mask = np.zeros_like(masks[i].cpu().numpy()[0], dtype=bool)
                    print(masks.shape,new_mask.shape)
                    new_mask[int(boxes_filt[i][1]):int(boxes_filt[i][3]),int(boxes_filt[i][0]):int(boxes_filt[i][0])+10] = masks[i].cpu().numpy()[0][ int(boxes_filt[i][1]):int(boxes_filt[i][3]),int(boxes_filt[i][0]):int(boxes_filt[i][0])+10]
                    boxes_filt = box_head.unsqueeze(0)
                    masks[i][0] = torch.from_numpy(new_mask)
                    pred_phrases = [pred_phrases[i]]
                    print(masks.shape)
                    break
                        
    # print(masks,pred_phrases)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # # print(pred_phrases)
    # # assert(0)
    print(boxes_filt)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # print(pred_phrases)
    # assert(0)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(out_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(out_dir, masks, boxes_filt, pred_phrases)
    # assert(0)
    
    if 'bimanual_straighten_rope' not in task_name:
        top_left, bottom_right=masked_image(masks[0:1,:,:,:],np.array(image_pil),out_dir, rgb_id, category,seed,step,task_name)
        
    else:
        top_left, bottom_right=masked_image(masks,np.array(image_pil),out_dir, rgb_id, category,seed,step,task_name)
        print('output',top_left,bottom_right)
    # if 'rope' in category:
    #     top_left = boxes_filt[0].numpy()[:2]
    #     bottom_right = boxes_filt[0].numpy()[2:]
    
    return top_left,bottom_right

def generate_target_mask(img_path, out_dir, rgb_id,task_name,seed,step,tl_pre=None,br_pre=None):
    config='Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint='Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
    device='cpu'
    box_threshold=0.3
    text_threshold=0.25
    use_sam_hq=False
    sam_checkpoint='Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    sam_version='vit_h'
    TARGET_CAT_LIST = {'put_rubbish_in_bin':'black bin','blocks':'light green','put_toilet_roll_on_stand':'the thin standing stand','slide_the_block_to_target':'green square','water_plants':'green plant','dustpan':'bigger dark dustpan','water_plants':'green plant','stack_the_wine_bottle_to_the_middle_of_the_rack':'rack'}
    TARGET_CAT_LIST.update({'bimanual_straighten_rope':'green block','bimanual_push_box':'red square','beat_the_buzz':'orange handle','put_the_phone_on_the_base':'orange phone', 'pick_up_the_hanger_and_place_in_on_the_rackput_the_hanger_on_the_rack':'grey long rack line','put_the_knife_on_the_chopping_board':'small chopping board on table','put_the_plate_between_the_red_pillars_of_the_dish_rack':'red lines','straighten_rope':'green block'})
    #print(task_name)
    if 'cup' in task_name:
        category = task_name.split('_cup')[1].split('_')[-1] + ' cup'
    else:
        for task in list(TARGET_CAT_LIST.keys()):
            if task.lower() in task_name.lower() or task_name.lower() in task.lower():
                #print(task)
                category = TARGET_CAT_LIST[task]
    #print(category)
    model=load_seg_model(config,checkpoint,device)
    # load image
    if 'green block' in category or 'chopping' in category or category == 'red':
        img_path = img_path[:-5]+'0.png'
    image_pil, image = load_image(img_path)
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, category, box_threshold, text_threshold, device=device
    )
    sorted_indices = sorted(range(len(pred_phrases)), key=lambda k: pred_phrases[k], reverse=True)

    # Step 2: 使用排序后的索引重新排列 pred_phrases 和 boxes_filt
    pred_phrases = [pred_phrases[i] for i in sorted_indices]
    boxes_filt = boxes_filt[sorted_indices]
    # print(boxes_filt, pred_phrases,category)

    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    print(boxes_filt)
    idx_mid = 10000
    if 'light green' in category:
        y_mid = 10000
        for i,item in enumerate(boxes_filt):
            # print(i)
            print(i,(item[0]+item[2])//2)
            if (item[0]+item[2])//2 > 305 and (item[0]+item[2])//2 < 370:
                # break
                # break
                print(i)
                width = (item[3]+item[1])//2
                print(width)
                if width < y_mid and width>470:
                    print('--------------')
                    y_mid = width
                    idx_mid = i
        if idx_mid == 10000:
            idx_mid = 0
        # print(idx_mid)
        # assert(0)
        boxes_filt = boxes_filt[idx_mid].unsqueeze(0)
        pred_phrases = [pred_phrases[idx_mid]]
    print(boxes_filt)
    if 'chopping' in category:
        # print(boxes_filt)
        # assert(0)
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if 'chopping' in item and boxes_filt[i][0] > 20:
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                # print(i)
                # assert(0)
                break
    if category == 'red lines':
        x_min = 10000
        y_min = 10000
        x_max = 0
        y_max = 0
        count = 0
        for i,item in enumerate(pred_phrases):
    #         # print(i,item)
            if count >=2:
                break
            if abs(boxes_filt[i][2]-boxes_filt[i][0]) < 20:
                count += 1
                if boxes_filt[i][0] < x_min:
                    print(boxes_filt[i][0],x_min)
                    x_min = boxes_filt[i][0]
                if boxes_filt[i][1] < y_min:
                    y_min = boxes_filt[i][1]
                if boxes_filt[i][2] > x_max:
                    # print(x_max)
                    x_max = boxes_filt[i][2]
                if boxes_filt[i][3] > y_max:
                    y_max = boxes_filt[i][3]
                print(boxes_filt[i])
                # print(x_min,y_min,x_max,y_max)
        if x_min != 10000 and y_min != 10000 and x_max != 0 and y_max != 0:
            boxes_filt[i][0] = x_min
            boxes_filt[i][1] = y_min
            boxes_filt[i][2] = x_max
            boxes_filt[i][3] = y_max
            boxes_filt = boxes_filt[i].unsqueeze(0)
            pred_phrases = [pred_phrases[i]]
    if 'green block' in category and 'bimanual' not in task_name:
        dis_min = 10000
        min_idx = -1
        for i,item in enumerate(pred_phrases):
            
            center_x = (boxes_filt[i][2]+boxes_filt[i][0])//2
            center_y = (boxes_filt[i][3]+boxes_filt[i][1])//2
            print(center_x,torch.Tensor(tl_pre),center_y,torch.Tensor(br_pre))
            center_x_contact = (torch.Tensor(tl_pre)[0]+torch.Tensor(br_pre)[0]) // 2
            center_y_contact = (torch.Tensor(tl_pre)[1]+torch.Tensor(br_pre)[1]) // 2
            if abs(center_y-center_x_contact) + abs(center_x-center_y_contact) < dis_min:
                dis_min = abs(center_y-center_x_contact) + abs(center_x-center_y_contact)
                min_idx = i
        boxes_filt = boxes_filt[min_idx].unsqueeze(0)
        pred_phrases = [pred_phrases[min_idx]]
    if 'orange handle' in category:
        
        print('1111111111111')
        for i,item in enumerate(pred_phrases):
            # print(i,item)
            if 'handle' in item and boxes_filt[i][1] > 20 and abs(boxes_filt[i][3]-boxes_filt[i][1])>abs(boxes_filt[i][2]-boxes_filt[i][0]):
                # print(item)
                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                # print(i)
                # assert(0)
                break
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # print(pred_phrases)
    # assert(0)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(out_dir, "grounded_sam_output_target.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(out_dir, masks, boxes_filt, pred_phrases)
    if 'bimanual_straighten_rope' not in task_name: 
        top_left, bottom_right=masked_image_target(masks[0:1,:,:,:],np.array(image_pil),out_dir, rgb_id, category,seed,step,task_name)
    else:
        top_left, bottom_right=masked_image_target(masks,np.array(image_pil),out_dir, rgb_id, category,seed,step,task_name)
    
    return top_left,bottom_right
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)

    # plt.axis('off')
    # plt.savefig(
    #     os.path.join(out_dir, "grounded_sam_output.jpg"),
    #     bbox_inches="tight", dpi=300, pad_inches=0.0
    # )

    # save_mask_data(out_dir, masks, boxes_filt, pred_phrases)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default = './groundingdino_swint_ogc.pth', required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='sam_vit_h_4b8939.pth', required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, default='./', help="path to image file")
    # parser.add_argument("--input_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    image_dir = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    # load model
    model = load_seg_model(config_file, grounded_checkpoint, device=device)
    CAT_LIST = {'lamp on':'little green light switch', 'turn on the light':'little green light switch', 'close fridge': 'fridge', 'close microwave': 'microwave with the whole body', 'drawer': 'drawer', 'close laptop lid': 'laptop', 'toilet seat': 'toilet seat with two parts', 'close box':'box with two parts', 'put toilet roll on stand': 'the smallest cylindrical toilet roll that is on the box', 'take usb out of computer':'This is a small usb connected to a big computer. Find the small usb.', 'take shoes out of box':'box with two parts', 'lamp on':'little green light switch', 'unplug charger':'charger on vertical board', 'water plants':'watering can','sweep dirt to dustpan':'the dark brown long pole with two parts'}

    data_list = os.listdir(image_dir)
    for data_item in tqdm.tqdm(data_list):
        print(data_item)
        if '2024-' not in data_item:
            continue
        
        FLAG = 'False'
        # try:
        front_rgb_list = os.listdir(os.path.join(image_dir,data_item,'front_rgb'))
        # for rgb_image_name in front_rgb_list:
        #     if 'v4' in rgb_image_name:
        #         FLAG = 'True'
        #         break
        # # print(FLAG)
        # if FLAG == 'True':
        #     continue
        draw_id = []
        for rgb_image_name in front_rgb_list:
            if 'draw_' in rgb_image_name and 'segmask' not in rgb_image_name:
                # print(rgb_image_name)
                rgb_id = rgb_image_name.split('.')[0].split('_')[1]
                # print(rgb_id)
                if rgb_id not in draw_id:
                    draw_id.append(rgb_id)
        # print(draw_id)
        for rgb_id in sorted(draw_id):
            json_dir = os.path.join(text_prompt,data_item+'_'+rgb_id+'.json')
            with open(json_dir, 'r') as file:
                json_data = json.load(file)
            task = json_data['variation_description']
            if 'usb' not in task:
                break
            # print(task)
            for task_name in list(CAT_LIST.keys()):
                if task.lower() in task_name.lower() or task_name.lower() in task.lower():
                    category = CAT_LIST[task_name]
            # print(category)
            img_path = os.path.join(image_dir,data_item,'front_rgb',rgb_id+'.png')
            output_path = os.path.join(image_dir,data_item,'front_rgb')
            generate_mask(img_path, output_path, rgb_id, task)
        # assert(0)
            # print(output_path)
        # exit()
        # except:
        #     continue
        #     # assert(0)
        # assert(0)

    # make dir
    # os.makedirs(output_dir, exist_ok=True)
    