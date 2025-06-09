# Merging Language Models (LMs)

## Checkpoints

**RoBERTa**: You can download the fine-tuned checkpoints from Hugging Face [here](https://huggingface.co/lfy-hg/roberta_base_on_glue).

Place the checkpoints as described in the [documentation](./ckpts/README.md).

## Data

The [**data**](./data) folder should contain two subfolders:
- One for the [evaluate](https://github.com/huggingface/evaluate) dataset.
- One for the [GLUE dataset](https://huggingface.co/datasets/nyu-mll/glue/tree/main).

Place these two folders following the instructions in the [documentation](./data/README.md).

## Dependencies

You can create an environment **merge_lm** using the following command:

```bash
conda env create -f environment.yml
```

## Get Started

The `run_merge_roberta.sh` script contains commands for merging the RoBERTa models, or you can run the code using the following commands:

```python
conda activate merge_lm
python merge_roberta.py \
    --drop_ratio 0.7 \
    --group_size 4 \
    --batch_size 16 \
    --eval_subset_size 500 \
    --knn_samples_per_dataset 100 \
    --knn_neighbors 5
```

### Parameters:

**--drop_ratio:** Set dropout rate.

**--group_size:** Specifies the size of the groups during task vector compression.

**--eval_subset_size:** Represents the maximum size of the evaluation dataset.

**--knn_samples_per_dataset:** Defines the number of samples collected for each dataset in KNN.

**--knn_neighbors:** Specifies the number of neighbors to use in the KNN algorithm.

