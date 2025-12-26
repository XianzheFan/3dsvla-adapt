# export export CUDA_HOME=/usr/local/cuda
# export CUDA_VISIBLE_DEVICES=0
python ./Grounded-Segment-Anything/grounded_sam_demo_2.py \
  --config ./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ./Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
  --sam_checkpoint ./Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
  --input_image ./RLBench/train_dataset \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device "cpu" 