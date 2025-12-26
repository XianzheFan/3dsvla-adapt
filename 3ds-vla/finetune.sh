#!/usr/bin/env bash
# /new/algo/user8/lixiaoqi/xcy/Manipllm_tta/ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth
OUTPUT_DIR='./exp/oxe_pretrain_0125'
mkdir -p "$OUTPUT_DIR"
#5e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --master_port=10033 --nproc_per_node=8 --use_env main_finetune.py --batch_size 3 \
   --epochs 12 --warmup_epochs 0 --blr 1e-4 --weight_decay 0.05 \
   --output_dir "$OUTPUT_DIR" \
   --pretrained_path /new/algo/user8/lixiaoqi/cloris-2/checkpoint-478000.pth \
   --llama_path /new/algo/user8/lixiaoqi/xcy/llama_model_weights \
   --data_config /new/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/data/rlbench_single_dual_1221 \
   --clip_only