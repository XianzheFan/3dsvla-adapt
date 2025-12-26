import os
import urllib
import hashlib
import warnings
import numpy as np

from tqdm import tqdm
import torch


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def format_prompt(instruction, input=None, lang_type='EN'):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    CH_PROMPT_DICT = {
        "prompt_input": (
            "Below is a chinese instruction that describes a task, paired with a chinese input that provides further context. "
            "Write a chinese response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is a chinese instruction that describes a task. "
            "Write a chinese response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    }
    if input is None or input == '':
        if lang_type == 'EN':
            return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
        else:
            return CH_PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        if lang_type == 'EN':
            return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})
        else:
            return CH_PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})

def discretize(q):
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
        rot = discretize(rot)
        pos = discretize(pos)

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
    rot = discretize(rot)
    pos = discretize(pos)
    
    gripper_pose_future_str = ', '.join(map(str, gripper_pose_future))
    pos_str = ', '.join(map(str, pos))
    rot_str = ', '.join(map(str, rot))
    prompt=("The gripper position is [{pos_str}]. The gripper rotation quaternion is [{rot_str}]. "
            "The gripper open status is {gripper_open_future} ").format_map({'pos_str': pos_str, 'rot_str': rot_str, 'gripper_open_future': gripper_open_future})
    return prompt
        


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    # assume the url is https://some/path/sha256_model.pth
    expected_sha256 = url.split("/")[-1].split('_')[0]
    # expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target
