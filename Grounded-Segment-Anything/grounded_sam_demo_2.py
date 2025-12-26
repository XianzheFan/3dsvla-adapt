import argparse
import os
import sys
import json
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageOps
import tqdm
sys.path.append(os.path.join(os.getcwd(), "./Grounded-Segment-Anything/GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "./Grounded-Segment-Anything/segment_anything"))


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


def load_model(model_config_path, model_checkpoint_path, device):
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
def box_mask(image, dir_path, id):
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
    json_data = {'top_left':np.array(top_left_2).tolist(), 'bottom_right':np.array(bottom_right_2).tolist()}
    with open(dir_path.split('.')[0]+'.json', 'w') as f:
        json.dump(json_data, f)



def masked_image(mask, image, out_dir, rgb_id, category):
    print(mask.squeeze().squeeze().cpu().numpy().shape)
    mask_reshape = mask.squeeze().squeeze().cpu().numpy()
    if len(mask_reshape.shape) == 2:
        mask_np = mask_reshape
        background_color = (255, 255, 255)
        new_image = np.full_like(image, background_color)
        new_image[mask_np] = image[mask_np]
        new_image_pil = Image.fromarray(new_image)
        new_image_pil2 = new_image_pil
        box_mask(new_image_pil,os.path.join(out_dir,rgb_id+'_{}_{}_box_target_1022.png'.format(0,category)),0)
        new_image_pil2.save(os.path.join(out_dir,rgb_id+'_{}_{}_segmask_target_1022.png'.format(0,category)))
    else:
        for i in range(mask.squeeze().squeeze().cpu().numpy().shape[0]):
            mask_np = mask.squeeze().squeeze().cpu().numpy()[i,:,:]
            background_color = (255, 255, 255)
            new_image = np.full_like(image, background_color)
            new_image[mask_np] = image[mask_np]
            new_image_pil = Image.fromarray(new_image)
            new_image_pil2 = new_image_pil
            box_mask(new_image_pil,os.path.join(out_dir,rgb_id+'_{}_{}_box_target_1022.png'.format(i,category)),i)
            
            new_image_pil2.save(os.path.join(out_dir,rgb_id+'_{}_{}_segmask_target_1022.png'.format(i,category)))



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

def generate_mask(img_path, model, out_dir, rgb_id, category):
    # load image
    image_pil, image = load_image(img_path)
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, category, box_threshold, text_threshold, device=device
    )
    sorted_indices = sorted(range(len(pred_phrases)), key=lambda k: pred_phrases[k], reverse=True)

    # Step 2: 使用排序后的索引重新排列 pred_phrases 和 boxes_filt
    pred_phrases = [pred_phrases[i] for i in sorted_indices]
    boxes_filt = boxes_filt[sorted_indices]

    if 'usb' in category:
        for i,item in enumerate(pred_phrases):
            if 'usb' in item:

                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break

    if 'charger' in category:
        for i,item in enumerate(pred_phrases):
            if 'charger' in item:

                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    if 'handle' in category:
        for i,item in enumerate(pred_phrases):

            if 'handle' in item:

                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break
    if 'grey circle' in category:
        for i,item in enumerate(pred_phrases):

            if 'grey circle' in item:

                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]
                break

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

    
    if 'light green' in category:
        y_mid = 10000
        for i,item in enumerate(boxes_filt):

            if (item[0]+item[2])//2 > 310 and (item[0]+item[2])//2 < 370:

                print(i)
                width = (item[3]+item[1])//2
                print(width)
                if width < y_mid and width>470:
                    y_mid = width
                    idx_mid = i

        boxes_filt = boxes_filt[idx_mid].unsqueeze(0)
        pred_phrases = [pred_phrases[idx_mid]]
    if 'chopping' in category:

        for i,item in enumerate(pred_phrases):

            if 'chopping' in item and boxes_filt[i][0] > 20:

                boxes_filt = boxes_filt[i].unsqueeze(0)
                pred_phrases = [pred_phrases[i]]

                break
    if category == 'red lines':
        x_min = 10000
        y_min = 10000
        x_max = 0
        y_max = 0
        count = 0
        for i,item in enumerate(pred_phrases):

            if count >=4:
                break
            if abs(boxes_filt[i][2]-boxes_filt[i][0]) < 20:
                count += 1
                if boxes_filt[i][0] < x_min:
                    print(boxes_filt[i][0],x_min)
                    x_min = boxes_filt[i][0]
                if boxes_filt[i][1] < y_min:
                    y_min = boxes_filt[i][1]
                if boxes_filt[i][2] > x_max:

                    x_max = boxes_filt[i][2]
                if boxes_filt[i][3] > y_max:
                    y_max = boxes_filt[i][3]
                print(boxes_filt[i])

        if x_min != 10000 and y_min != 10000 and x_max != 0 and y_max != 0:
            boxes_filt[i][0] = x_min
            boxes_filt[i][1] = y_min
            boxes_filt[i][2] = x_max
            boxes_filt[i][3] = y_max
            boxes_filt = boxes_filt[i].unsqueeze(0)
            pred_phrases = [pred_phrases[i]]
            
        
        
                # break
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    masked_image(masks,np.array(image_pil),out_dir, rgb_id, category)
   
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
    parser.add_argument("--text_prompt", type=str, default="cat", help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
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
    model = load_model(config_file, grounded_checkpoint, device=device)
    CAT_LIST = {'lamp_on':'little green light switch','slide_block_to_target':'block','light_bulb_in':'small grey circle','push_button':'round button','turn_tap':'small cross-shaped cross on faucet','stack_blocks':'blocks' ,'place_wine_at_rack_location':'wine','turn on the light':'little green light switch', 'open_fridge': 'fridge','put_rubbish_in_bin':'small white rubbish','open_microwave':'small handle on microwave','close_jar': 'small grey circle on the table', 'close_fridge': 'fridge', 'close_microwave': 'microwave with the whole body', 'drawer': 'drawer', 'close_laptop_lid': 'laptop', 'seat': 'toilet seat with two parts', 'close_box':'box with two parts', 'put_toilet_roll': 'the smallest cylindrical toilet roll that is on the box', 'take_usb':'This is a small usb connected to a big computer. Find the small usb.','lamp_on':'little green light switch', 'unplug_charger':'the smallest black charger on vertical board', 'water_plants':'watering can','sweep':'the dark brown long pole with two parts'}
    
    TARGET_CAT_LIST = {'stack_blocks':'light green'}
    TARGET_CAT_LIST.update({'put_knife_on_chopping_board':'chopping board','put_plate_in_colored_dish_rack':'red lines','straighten_rope':'green block'})
    
    data_list = os.listdir(image_dir)
    for data_item in tqdm.tqdm(sorted(data_list)):
        
        category = None
        # if 'plate' not in data_item:
        #     continue
        
        FLAG = 'False'

        front_rgb_list = os.listdir(os.path.join(image_dir,data_item,'front_rgb'))

        draw_id = []
        for rgb_image_name in front_rgb_list:
            if 'draw3.' in rgb_image_name and 'segmask' not in rgb_image_name:
                # print(rgb_image_name)
                rgb_id = rgb_image_name.split('/')[-1].split('.')[0].split('_')[0]
                # print(rgb_id)
                if rgb_id not in draw_id:
                    draw_id.append(rgb_id)
        if len(draw_id) == 0:
            continue

        task = data_item.split('-')[0]
        for task_name in list(TARGET_CAT_LIST.keys()):
            if task_name.lower() in task.lower() or task.lower() in task_name.lower():
                category = TARGET_CAT_LIST[task_name]
                break
        
        if category == None:
            continue
        print(data_item)
        for rgb_id in sorted(draw_id):

            img_path = os.path.join(image_dir,data_item,'front_rgb',rgb_id+'.png')
            output_path = os.path.join(image_dir,data_item,'front_rgb')
            generate_mask(img_path, model, output_path, rgb_id, category)
            