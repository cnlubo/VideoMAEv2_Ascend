cd /data/lubo/VideoMAEv2_Ascend

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 run_single_video_inference.py \
--video ./video/1AlH6EMWtgg_000003_000013.mp4 \
--checkpoint /data/lubo/checkpoints/VideoMAE2/distill/vit_b_k710_dl_from_giant.pth \
--labels /data/lubo/VideoMAEv2_Ascend/misc/label_map_k710.txt \
--model vit_base_patch16_224 \
--num-classes 710 \
--device npu:0 \
--num-frames 16 \
--sampling-rate 4 \
--input-size 224 \
--short-side-size 224 \
--tubelet-size 2 \
--top-k 5 \
--drop-path 0.0 \
--init-scale 0.001 \
--use-mean-pooling
