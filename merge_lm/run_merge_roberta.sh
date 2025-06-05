#!/bin/bash
source /mnt/workspace/gaojunqi/anaconda3/etc/profile.d/conda.sh
conda activate merge_lm
cd /mnt/workspace/gaojunqi/lifangyuan/model_merge/ours_lm/merge_lm


python merge_roberta.py \
    --drop_ratio 0.7 \
    --group_size 4 \
    --batch_size 16 \
    --eval_subset_size 500 \
    --knn_samples_per_dataset 100 \
    --knn_neighbors 5
    
