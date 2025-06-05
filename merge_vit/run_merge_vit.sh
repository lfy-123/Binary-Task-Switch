#!/bin/bash
source /mnt/workspace/gaojunqi/anaconda3/etc/profile.d/conda.sh
conda activate merge
cd /mnt/workspace/gaojunqi/lifangyuan/model_merge/ours_sim/merge_vit



python merge_vit.py --model-type "ViT-B-32" \
--drop-ratio 0.6 \
--group_size 4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256 > test_ViT-B-32.log 2>&1



python merge_vit.py --model-type "ViT-L-14" \
--drop-ratio 0.6 \
--group_size 4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256 > test_ViT-L-14.log 2>&1


conda deactivate
conda activate merge_lora

python merge_vit_lora.py --model-type "ViT-B-32" \
--drop-ratio 0.4 \
--group_size 4 \
--lora-rank 64 --lr 1e-4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256 > test_ViT-B-32_lora.log 2>&1


