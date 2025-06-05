#!/bin/bash



python merge_roberta.py \
    --drop_ratio 0.6 \
    --group_size 4 \
    --batch_size 256 \
    --eval_subset_size 500 \
    --knn_samples_per_dataset 100
    
    
    
    