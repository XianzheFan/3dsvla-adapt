

OUTPUT_DIR='./3ds-vla/exp/train_model'
mkdir -p "$OUTPUT_DIR"
# #5e-3
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --master_port=10033 --nproc_per_node=1 --use_env ./3ds-vla/main_finetune.py --batch_size 3 \
   --epochs 12 --warmup_epochs 0 --blr 1e-4 --weight_decay 0.05 \
   --output_dir "$OUTPUT_DIR" \
   --pretrained_path ./3ds-vla/pretrain/checkpoint-478000.pth \
   --llama_path ./3ds-vla/pretrain/llama_model_weights \
   --data_config ./3ds-vla/data/train_json_single \
   --clip_only