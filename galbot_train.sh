export http_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
export https_proxy=http://'galbot:sK0aZ5bZ9v'@10.119.176.202:3128
apt update
apt install -y ffmpeg

export GX_STORAGE_PATH=$CODE_DIR/storage
export GX_REGISTER=sim_data_gen.franka.register.GRASPVLA_REGISTER
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
  NUM_GPUS=$(ls /dev | grep -E 'nvidia[0-9]+' | wc -l)
fi

grep -R "TORCH_CUDA_ARCH_LIST" -n \
  $HOME/.bashrc $HOME/.bash_profile $HOME/.profile \
  $CONDA_PREFIX/etc/conda/activate.d 2>/dev/null || true

export NCCL_IB_TIMEOUT=10000
export NCCL_IB_RETRY_CNT=10000
export NCCL_IB_AR_THRESHOLD=0
export SDL_AUDIODRIVER=dummy
export HOME=/mnt/afs/fanxianzhe
sed -i 's/\/root/\/mnt\/afs\/fanxianzhe/g' /etc/passwd
source $HOME/.bashrc
conda activate 3ds-vla

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
env | grep -E 'TORCH_CUDA_ARCH_LIST|CUDA_HOME|CUDA_VISIBLE_DEVICES' -n
cd /mnt/project/public/public_datasets/mix-crop-front-depth-side-290cat-200w/3dsvla-adapt
export CUDA_HOME=/usr/local/cuda-12.8

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
echo "[INFO] Using LD_PRELOAD=$LD_PRELOAD"
ls -l "$LD_PRELOAD"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

OUTPUT_DIR='3ds-vla/exp/train_model'
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 
python -u -m torch.distributed.launch --master_port=10033 --nproc_per_node=1 --use_env 3ds-vla/main_finetune.py --batch_size 3 \
  --epochs 12 --warmup_epochs 0 --blr 1e-4 --weight_decay 0.05 \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_path 3ds-vla/pretrain/checkpoint-478000.pth \
  --llama_path 3ds-vla/pretrain/llama_model_weights \
  --data_config 3ds-vla/data/train_json_single \
  --clip_only \
  --use_rlds \
  --rlds_root /mnt/project/simvla/data/deploy-data/front-depth-side-290cat-200w/graspsim_train \
  --rlds_use_depth \
  --rlds_pc_num 2304 \
  --rlds_prefetch 1 \
  --rlds_stride 2 \
  --max_words 512 \





# USER_HOME=/mnt/afs/fanxianzhe
# CODE_DIR=/mnt/afs/fanxianzhe/galbot_vla
# ENV_DIR=galbotvla_env
# TRAIN_SETS="/mnt/project/simvla/data/deploy-data/front-depth-side-290cat-200w/graspsim_train,rlds,3;/mnt/project/simvla/data/deploy-data/front-depth-side-290cat-200w/graspsim_train,rlds_bbox,1;/mnt/project/public/public_datasets/grit,grit,6"
# VAL_SETS="/mnt/project/simvla/data/deploy-data/front-depth-side-290cat-200w/graspsim_train,rlds,1"
# ROBOT=epiclab_franka
# export PYTHONPATH=$PYTHONPATH:/mnt/afs/fanxianzhe/galbot_vla
# export WANDB_API_KEY="b3aae75a6a8394afed0b7a95ea1fe3c4c5903f5f"

# export GX_STORAGE_PATH=$CODE_DIR/storage
# export GX_REGISTER=sim_data_gen.franka.register.GRASPVLA_REGISTER
# if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
#   NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# else
#   NUM_GPUS=$(ls /dev | grep -E 'nvidia[0-9]+' | wc -l)
# fi
# GLOBAL_BATCH_SIZE=128
# export WANDB__SERVICE_WAIT=3600
# export WANDB_INIT_TIMEOUT=3600
# WORLD_SIZE=${WORLD_SIZE:=1}
# RANK=${RANK:=0}
# MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
# MASTER_PORT=${MASTER_PORT:=9265}
# export NCCL_IB_TIMEOUT=10000
# export NCCL_IB_RETRY_CNT=10000
# export NCCL_IB_AR_THRESHOLD=0
# export SDL_AUDIODRIVER=dummy

# export HOME=/mnt/afs/fanxianzhe
# sed -i 's/\/root/\/mnt\/afs\/fanxianzhe/g' /etc/passwd
# source $HOME/.bashrc

# conda activate $ENV_DIR
# export CUDA_HOME=/usr/local/cuda-12.8

# cd $CODE_DIR/src/vla_network

# git config --global --add safe.directory '*'

# export LD_PRELOAD=$USER_HOME/.conda/envs/$ENV_DIR/lib/libstdc++.so.6
# echo "[INFO] Using LD_PRELOAD=$LD_PRELOAD to fix ABI conflicts."

# torchrun --nnodes $WORLD_SIZE \
#          --nproc-per-node $NUM_GPUS \
#          --node-rank $RANK \
#          --master-addr $MASTER_ADDR \
#          --master-port $MASTER_PORT \
#          --module vla_network.scripts.train_vla \
#          --exp_name=$EXP_NAME \
#          --train_datasets=$TRAIN_SETS \
#          --val_datasets=$VAL_SETS \
#          --global_bs=$GLOBAL_BATCH_SIZE \
#          --device_bs=$((GLOBAL_BATCH_SIZE / NUM_GPUS / WORLD_SIZE)) \
#          --num_workers=1 \
#          --count_num=10000 \
#          --robot=$ROBOT \
#          --save_step 20000 \
#          --max_steps 500000 \
#          --deepspeed none \
#          --grad_ckpt 0 \
#          --fsdp full_shard \
#          --use_bbox 1 \
#          --use_depth 0 \
#          --pred cot_flow_matching \
#          --action_expert 1 \
#          --attn_impl flex_attention \
#          --backbone_2d dinosiglipmono

sleep inf