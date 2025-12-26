import os
import json
from pathlib import Path
import numpy as np
import re
import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from .helpers import Point_PN_scan, PCViews
from .llama import ModelArgs, Transformer, BERTTransformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download
import torch.nn.functional as F
import argparse
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,args=None,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14@336px',
                 v_embed_dim=1024, v_depth=16,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=577, query_layer=32, phase="finetune"):
        super().__init__()
        # llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        bias_lora = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params     ###llaama参数
        ) # max_batch_size only affects inferenc
        
        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model,download_root='/qiyuan_research_vepfs_001/lixiaoqi/3ds-vla/3ds-vla/pretrain')
        if args is not None:
            if args.clip_only:
            
                self.patch_embed = Point_PN_scan(k_neighbors=64)
                pc_views = PCViews()
                self.get_pos_2d = pc_views.get_pos_2d
                self.pos_embed_2d = self.clip.visual.positional_embedding.unsqueeze(0)
                self.img_size = 336
                self.patch_size=14
                self.cls_pos = nn.Parameter(torch.randn(1, 1, 1024))
                self.fc_patch = nn.Sequential(
                nn.Linear(768, 1024),  # First layer: 768 -> 1024
                nn.ReLU(),                   # Activation function
                # You can add more layers here if needed
            )
                self.token_mlp = nn.Sequential(
                nn.Linear(1154, 577),  # Fully connected layer (1154 -> 577)
                nn.ReLU(),  # ReLU activation
            )

        
        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector

        visual_model_args = ModelArgs(dim=1024, n_layers=16, n_heads=8, max_seq_len=577)
        visual_model_args.vocab_size = 1024
        # visual_model_args.vocab_size = 64
        self.visual_blocks = BERTTransformer(visual_model_args)
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama 
        model_args.w_bias = bias_lora
        model_args.w_lora = bias_lora
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = ['/qiyuan_research_vepfs_001/lixiaoqi/3ds-vla/3ds-vla/pretrain/llama_model_weights/7B/consolidated.00.pth']
        for ckpt in ckpts:
            print('load_ckpt_path:', ckpt)
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
        
        for name, param in self.named_parameters():
            param.requires_grad = False
            if args.clip_only:
                if 'patch_embed' in name:
                    param.data = param.data.float()
                    param.requires_grad = True
                    
                if 'fc_patch' in name:
                    param.data = param.data.float()
                    param.requires_grad = True
                    
                if 'token_mlp' in name:
                    param.data = param.data.float()
                    param.requires_grad = True
                    

        for name, para in self.llama.named_parameters():
            if 'norm' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'bias' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'lora' in name:
                para.data = para.data.float()
                para.requires_grad = True
            
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
               count += 1
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        # print(count)
        # exit()
        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    def bilinear_interpolation_3d_to_2d(self, x, y, pos_embed):
        grid_x = (2.0 * x / (self.img_size - 1)) - 1
        grid_y = (2.0 * y / (self.img_size - 1)) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(2)

        pos_embed_reshaped = (
            pos_embed.permute(0, 2, 1)
            .view(
                1,
                -1,
                int(self.img_size / self.patch_size),
                int(self.img_size / self.patch_size),
            )
            .repeat(grid.shape[0], 1, 1, 1)
        )
        pos_embed_reshaped = pos_embed_reshaped.to(x.device) #24, 1024, 24, 24
        # print(pos_embed_reshaped.shape)
        interpolated_pos_embed = F.grid_sample(
            pos_embed_reshaped,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )#24, 1024, 128, 1]
        # print(interpolated_pos_embed.shape)
        return interpolated_pos_embed.squeeze()
    def clip_encode_image(self, x,group_input_tokens=None,interpolated_pos_embed=None):
        # modified from CLIP
        if group_input_tokens is not None:
            img_clip_token = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid] 4, 1024, 24, 24
            # shape = [*, width, grid ** 2]

            img_clip_token = img_clip_token.reshape(img_clip_token.shape[0], img_clip_token.shape[1], -1).permute(0, 2, 1)
            x = self.fc_patch(group_input_tokens) #[4, 576, 768]
            x = (0.1*x + 0.9*img_clip_token) 
            
            # x = img_clip_token
            # interpolated_pos_embed = None
            x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                        x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
            img_clip_token = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                        x.shape[-1], dtype=x.dtype, device=x.device), img_clip_token], dim=1) 
            # x = x+ self.clip.visual.positional_embedding
            
            # img_clip_token = img_clip_token + torch.cat([interpolated_pos_embed,self.cls_pos.expand(group_input_tokens.shape[0], -1, -1)],dim=1)
            x = x+ torch.cat([interpolated_pos_embed,self.cls_pos.expand(group_input_tokens.shape[0], -1, -1)],dim=1)
            
            img_clip_token = img_clip_token + self.clip.visual.positional_embedding
            # print(self.clip.visual.positional_embedding.shape)
            # assert(0)
            ##cat
            # print('-------add with mlp-------')
            cat_result = torch.cat((x, img_clip_token), dim=1)
            cat_result = cat_result.permute(0, 2, 1)  # Change shape to [4, 1024, 1154]
            x = self.token_mlp(cat_result)
            x = x.permute(0, 2, 1)
            ##add
            # x = 0.5*x+0.5*img_clip_token
            
            # if interpolated_pos_embed != None:
            #     print('------with pos emb----')
            #     cls_pos = self.cls_pos.expand(group_input_tokens.shape[0], -1, -1)
            #     # print(cls_pos.shape)
            #     interpolated_pos_embed = torch.cat([interpolated_pos_embed,cls_pos],dim=1)
            #     assert(0)
            # else:
            #     print('------without pos emb----')
            #     interpolated_pos_embed = self.clip.visual.positional_embedding
            # x = x + interpolated_pos_embed.to(x.dtype)#4, 577, 1024
            # print(x.shape)
            # assert(0)
        else:
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid] 4, 1024, 24, 24
            # shape = [*, width, grid ** 2]

            x = x.reshape(x.shape[0], x.shape[1], -1)
            
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                        x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] 4, 577, 1024

            x = x + self.clip.visual.positional_embedding.to(x.dtype)#4, 577, 1024

        x = self.clip.visual.ln_pre(x)

        # print(x.shape)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])
        # print(x.shape)
        
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj #4, 577, 768
        # print(x.shape)
        # assert(0)
        #bx577x768
        return x

    # def oppo_loss(self, pos_pred, bbox_1, bbox_2, target_pixel):
    #     # print(pos_pred, bbox_1, bbox_2, target_pixel)
    #     oppo_loss = 0
    #     for i in range(len(pos_pred)):
    #         if target_pixel[i] > max(bbox_1[i], bbox_2[i]):
    #             upper_bound_penalty = np.clip(pos_pred[i]-min(bbox_1[i],bbox_2[i]), 0, None)
    #             # print('1111111', upper_bound_penalty)
    #             oppo_loss += upper_bound_penalty
    #         elif target_pixel[i] < min(bbox_1[i], bbox_2[i]):
    #             lower_bound_penalty = np.clip(max(bbox_1[i],bbox_2[i])-pos_pred[i], 0, None)
    #             # print('22222222',lower_bound_penalty)
    #             oppo_loss += lower_bound_penalty
    #         else:
    #             lower_bound_penalty = np.clip(min(bbox_1[i],bbox_2[i]) - pos_pred[i], 0, None)
    #             upper_bound_penalty = np.clip(pos_pred[i] - max(bbox_1[i],bbox_2[i]), 0, None)
    #             # Combine penalties and apply penalty factor
    #             penalty = (lower_bound_penalty + upper_bound_penalty)/2
    #             # print('33333333',penalty)
    #             oppo_loss += penalty
    #     oppo_loss_norm = oppo_loss / 672
    #     return oppo_loss_norm
    
    def cosine_similarity(self, q1, q2):
        # Convert quaternions to numpy arrays if they aren't already
        q1 = np.array(q1)
        q2 = np.array(q2)
        
        # Compute the dot product of the two quaternions
        dot_product = np.dot(q1, q2)
        
        # Compute the magnitudes of the quaternions
        norm_q1 = np.linalg.norm(q1)
        norm_q2 = np.linalg.norm(q2)
        
        # Compute the cosine similarity
        cosine_sim = dot_product / (norm_q1 * norm_q2)
        cosine_sim_loss = np.clip((1-cosine_sim),0,None)
        return cosine_sim_loss
    def oppo_loss(self, pos_pred, bbox_1, bbox_2, target_pixel):
        THRESHHOLD = 60
        oppo_loss = 0
        # print(bbox_1,bbox_2,pos_pred)
        box_center = [(bbox_1[0]+bbox_2[0])//2,(bbox_1[1]+bbox_2[1])//2]
        
        xB, yB = box_center
        xA, yA = target_pixel
        xC, yC = pos_pred
        # Calculate vectors AB and AC
        vectorAB = np.array([xB - xA, yB - yA])
        vectorAC = np.array([xC - xA, yC - yA])
        
        # Normalize the vectors
        normAB = np.linalg.norm(vectorAB)
        normAC = np.linalg.norm(vectorAC)
        
        if normAB == 0 or normAC == 0:
            # Avoid division by zero in case the vectors are degenerate (e.g., A and B are the same point)
            return 0.0  # Infinite loss if the vectors cannot be computed
        
        # Normalized vectors
        vectorAB_normalized = vectorAB / normAB
        vectorAC_normalized = vectorAC / normAC
    
        # Cross product of normalized vectors in 2D
        cross_product = vectorAB_normalized[0] * vectorAC_normalized[1] - vectorAB_normalized[1] * vectorAC_normalized[0]
        collinearity_loss = np.abs(cross_product)  # Should be close to 0
        
        # Step 2: Ensure B is inside the segment, not on the sides
        AB_squared = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)  # Squared distance between A and B
        BC_squared = np.sqrt((xC - xB) ** 2 + (yC - yB) ** 2)  # Squared distance between B and C
        AC_squared = np.sqrt((xC - xA) ** 2 + (yC - yA) ** 2)  # Squared distance between A and C
        normalized_AB = AB_squared / AC_squared
        normalized_BC = BC_squared / AC_squared
        if AC_squared == 0:
            segment_loss =  2.0  
        # We want AB + BC = AC (but using squared distances to avoid square root)
        if AB_squared >= AC_squared:  # AB 不应大于 AC
            segment_loss = (normalized_AB - 1.0) 
        elif AB_squared <= 0.7 * AB_squared:  # AB 不应小于 0.8 * AC
            segment_loss = (1 - normalized_AB) 
        else:
            segment_loss = 0
            

        # print('segment loss',segment_loss)
        # assert(0)
        # # The segment loss should ensure AB + BC equals AC (in normalized terms)
        # segment_loss = np.abs(normalized_AB + normalized_BC - 1)
        
        # # # Step 3: Ensure B is not at A or C (penalize if B coincides with A or C)
        # distance = np.linalg.norm(np.array(box_center)-np.array(pos_pred))
        # if BC_squared > THRESHHOLD:
        #     boundary_loss_A = 0
        # else:
        #     boundary_loss_A = (THRESHHOLD - distance) /336
        # print(target_pixel,pos_pred,box_center, collinearity_loss, segment_loss, normalized_AB)
        # # assert(0)
        # # Total loss: Collinearity + Segment Loss + Boundary Loss
        oppo_loss_norm = collinearity_loss + segment_loss
        # print('--------------------',oppo_loss_norm)
        # assert(0)
        return oppo_loss_norm

    def range_loss(self, pos_pred, bbox_1, bbox_2):
        
        range_loss = 0
        for i in range(len(pos_pred)):
            lower_bound_penalty = np.clip(min(bbox_1[i],bbox_2[i]) - pos_pred[i], 20, None)
            upper_bound_penalty = np.clip(pos_pred[i] - max(bbox_1[i],bbox_2[i]), 20, None)
            # Combine penalties and apply penalty factor
            penalty = (lower_bound_penalty + upper_bound_penalty)/2
            # print('33333333',penalty)
            range_loss += penalty
        range_loss_norm = range_loss / 336
        # print(pos_pred, bbox_1, bbox_2,range_loss_norm)
        return range_loss_norm
        



    def forward_visual(self, imgs,group_input_tokens=None,interpolated_pos_embed=None,args=None):
        clip_feats = self.clip_encode_image(imgs,group_input_tokens,interpolated_pos_embed)
        
        
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))
        
        
        visual_query = clip_feats
        visual_query = self.visual_blocks(visual_query, 0)
        
        visual_query = self.visual_proj(visual_query)
        
        visual_query = self.visual_proj_norm(visual_query)
        

        return visual_query

    def forward(self, tokens, labels, imgs, prompts,point_cloud,args):
        
        if args.clip_only:
            #point cloud:bsx1024x3
            pts = point_cloud[:, :, :3].half().cuda()
            batch_size = pts.shape[0]
            pts_trans = pts.clone().transpose(1, 2).contiguous()
            
            center, group_input_tokens = self.patch_embed(pts_trans, pts)
            group_input_tokens = group_input_tokens.transpose(1, 2) #[4, 576, 768
            
            
            pos_x, pos_y, _ = self.get_pos_2d(center)
            
            
            self.patch_pos_embed_2D = self.pos_embed_2d[:, 1:]
            
            interpolated_pos_embed = self.bilinear_interpolation_3d_to_2d(
                pos_x, pos_y, self.patch_pos_embed_2D
            )
            interpolated_pos_embed = interpolated_pos_embed.reshape(
                center.shape[0], -1, center.shape[1], 1024
            )
            interpolated_pos_embed = interpolated_pos_embed.mean(dim=1) #4, 576, 1024
            # interpolated_pos_embed = None
            visual_proj = self.forward_visual(imgs,group_input_tokens,interpolated_pos_embed,args)
            # assert(0)
        else:
            visual_proj = self.forward_visual(imgs,args=args)
        
        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask, visual_proj + adapter[adapter_index])             ####调用前向传播forward
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        # output_de = torch.argmax(output, dim=-1)
        # output_list = output_de.cpu().numpy().tolist()
        
        oppo_loss_mean = 0
        range_loss_mean = 0
        simi_loss_mean = 0
        if args.simi_loss == True:

            output_de = torch.argmax(output, dim=-1)
            output_list = output_de.cpu().numpy().tolist()
            
            
            output_decoded = [str(self.tokenizer.decode(i)) for i in output_list]
            # print('---------',len(output_decoded))
            pattern_quat = r'quaternion is \[(-?\d+), (-?\d+), (-?\d+), (-?\d+)]'
            simi_loss_batch = []
            for i,answer in enumerate(output_decoded):
                # print(answer)
                quat_output = re.findall(pattern_quat, answer)
                
                if quat_output != [] and 'delta' in prompts[i]:
                    quat_pred = [int(quat_output[0][0]),int(quat_output[0][1]),int(quat_output[0][2]),int(quat_output[0][3])]
                    target_quat =  prompts[i].split('[')[3].split(']')[0].split(',')
                    target_quat = [int(re.findall(r'\d+', i)[0]) for i in target_quat]
                    cos_simi_loss = self.cosine_similarity(quat_pred,target_quat)
                    simi_loss_batch.append(cos_simi_loss)
            if simi_loss_batch != []:
                simi_loss_mean = sum(simi_loss_batch)/len(output_decoded)
                print('--------------',simi_loss_mean)
                # assert(0)
            return c_loss + simi_loss_mean, c_loss
                    
        if args.oppo_loss == True and args.range_loss != True:
            
            output_de = torch.argmax(output, dim=-1)
            # output_list = output_de.cpu().numpy().tolist()
            output_list = output_de.detach().cpu().tolist()
            
            output_decoded = [str(self.tokenizer.decode(i)) for i in output_list]
            # print('---------',len(output_decoded))
            pattern_pos = r'position is \[(-?\d+), (-?\d+)]'
            oppo_loss_batch = []
            
            for i,answer in enumerate(output_decoded):
                pos_output = re.findall(pattern_pos, answer)
                if pos_output != [] and len(pos_output[0])==2 and 'opposite' in prompts[i]:
                    keyword = "The gripper position is ["
                    start_index = answer.find(keyword)

                    if start_index != -1:
                        # 定位到 '[' 后第一个数字的起始位置
                        number_start = start_index + len(keyword)
                        number_end = answer.find(']', number_start)
                        coordinates_str = answer[number_start:number_end]
    
                        # 分割成两个数字
                        x, y = map(int, coordinates_str.split(', '))
                    pos_pred = [x,y]
                    bbox_1 = prompts[i].split('[')[1].split(']')[0].split(',')
                    bbox_1 = [int(i) for i in bbox_1]
                    bbox_2 = prompts[i].split('[')[2].split(']')[0].split(',')
                    bbox_2 = [int(i) for i in bbox_2]
                    # print(prompts[i])
                    target_pixel = prompts[i].split('[')[3].split(']')[0].split(',')
                    target_pixel = [int(re.findall(r'\d+', i)[0]) for i in target_pixel]
                    # print(pos_pred, bbox_1, bbox_2, target_pixel)
                    
                    oppo_loss = self.oppo_loss(pos_pred, bbox_1, bbox_2, target_pixel)
                    oppo_loss_batch.append(oppo_loss)
                
            if oppo_loss_batch != []:
                oppo_loss_mean = sum(oppo_loss_batch)/len(oppo_loss_batch)
            
                print(oppo_loss_mean, c_loss + oppo_loss_mean)
                # assert(0)
            return c_loss + oppo_loss_mean, c_loss
        elif args.oppo_loss == True and args.range_loss == True:
            output_de = torch.argmax(output, dim=-1)
            output_list = output_de.cpu().numpy().tolist()
            
            
            output_decoded = [str(self.tokenizer.decode(i)) for i in output_list]
            # print('---------',len(output_decoded))
            pattern_pos = r'position is \[(-?\d+), (-?\d+)]'
            oppo_loss_batch = []
            range_loss_batch = []
            for i,answer in enumerate(output_decoded):
                pos_output = re.findall(pattern_pos, answer)
                if pos_output != [] and len(pos_output[0])==2 and 'opposite' in prompts[i]:
                    pos_pred = [int(pos_output[0][0]),int(pos_output[0][1])]
                    bbox_1 = prompts[i].split('[')[1].split(']')[0].split(',')
                    bbox_1 = [int(i) for i in bbox_1]
                    bbox_2 = prompts[i].split('[')[2].split(']')[0].split(',')
                    bbox_2 = [int(i) for i in bbox_2]
                    print(prompts[i])
                    target_pixel = prompts[i].split('[')[3].split(']')[0].split(',')
                    target_pixel = [int(re.findall(r'\d+', i)[0]) for i in target_pixel]
                    oppo_loss = self.oppo_loss(pos_pred, bbox_1, bbox_2, target_pixel)
                    oppo_loss_batch.append(oppo_loss)
                if pos_output != [] and len(pos_output[0])==2 and 'opposite' not in prompts[i]:
                    pos_pred = [int(pos_output[0][0]),int(pos_output[0][1])]
                    bbox_1 = prompts[i].split('[')[1].split(']')[0].split(',')
                    bbox_1 = [int(i) for i in bbox_1]
                    bbox_2 = prompts[i].split('[')[2].split(']')[0].split(',')
                    bbox_2 = [int(i) for i in bbox_2]
                    range_loss = self.range_loss(pos_pred, bbox_1, bbox_2)
                    range_loss_batch.append(range_loss)
            if oppo_loss_batch != []:
                oppo_loss_mean = sum(oppo_loss_batch)/len(oppo_loss_batch)
                
            if range_loss_batch != []:
                range_loss_mean = sum(range_loss_batch)/len(range_loss_batch)
                
            return c_loss + oppo_loss_mean + range_loss_mean, c_loss
        else:
            return c_loss, c_loss

    @torch.no_grad()
    def forward_inference(self, visual_proj, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)


        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0

        for layer in self.llama.layers:
            h = layer(h, start_pos, freqs_cis, mask, visual_proj + adapter[adapter_index].repeat(_bsz, 1, 1))
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    #@torch.inference_mode()
    @torch.no_grad()
    def generate(
        self, imgs, prompts,pc=None,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            if pc is None:
                visual_query = self.forward_visual(imgs)
            else:
                print('---------use 3d tokens -----------')
                pc = torch.tensor(pc)
                if len(pc.shape) == 2:
                    pc = pc.unsqueeze(0)
                pts = pc[:, :, :3].half().cuda()
                batch_size = pts.shape[0]
                pts_trans = pts.clone().transpose(1, 2).contiguous()
                
                center, group_input_tokens = self.patch_embed(pts_trans, pts)
                group_input_tokens = group_input_tokens.transpose(1, 2) #[4, 576, 768
                
                
                pos_x, pos_y, _ = self.get_pos_2d(center)
                
                self.patch_pos_embed_2D = self.pos_embed_2d[:, 1:]

                interpolated_pos_embed = self.bilinear_interpolation_3d_to_2d(
                    pos_x, pos_y, self.patch_pos_embed_2D
                )
                interpolated_pos_embed = interpolated_pos_embed.reshape(
                    center.shape[0], -1, center.shape[1], 1024
                )
                interpolated_pos_embed = interpolated_pos_embed.mean(dim=1) #4, 576, 1024
                # interpolated_pos_embed = None
                visual_query = self.forward_visual(imgs,group_input_tokens,interpolated_pos_embed)
                # visual_query = self.forward_visual(pc)
        
        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded


_MODELS = {
    "BIAS-7B": "https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, aff_flag=False, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}")
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = llama_dir
    # llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    print("ckpt get===================")
    if aff_flag == True:
        parser = argparse.ArgumentParser(description="Script with a 'clip_only' argument.")

        # Add the 'clip_only' flag
        parser.add_argument(
            "--clip_only",type=bool, default=True
        )

        # Parse arguments
        args = parser.parse_args()
        model = LLaMA_adapter(
            llama_ckpt_dir, llama_tokenzier_path,args,
            max_seq_len=max_seq_len, max_batch_size=1,
            clip_model='ViT-L/14@336px',
            v_embed_dim=1024, v_depth=16,
            v_num_heads=16, v_mlp_ratio=4.0,
            query_len=577, query_layer=32,
            phase=phase)
    else:
        model = LLaMA_adapter(
            llama_ckpt_dir, llama_tokenzier_path,
            max_seq_len=max_seq_len, max_batch_size=1,
            clip_model='ViT-L/14@336px',
            v_embed_dim=1024, v_depth=16,
            v_num_heads=16, v_mlp_ratio=4.0,
            query_len=577, query_layer=32,
            phase=phase)

    print("model get===================")

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform