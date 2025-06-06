# Merging Language Models (LMs)

## Checkpoints

**RoBERTa**: You can download the fine-tuned checkpoints from Hugging Face [here](https://huggingface.co/lfy-hg/roberta_base_on_glue).

Place the checkpoints as described in the [documentation](./ckpts/README.md).

## Data

The [**data**](<insert link>) folder contains two subfolders:
- One for the [evaluate](https://github.com/huggingface/evaluate) dataset.
- One for the [GLUE dataset](https://huggingface.co/datasets/nyu-mll/glue/tree/main).

Place these two folders following the instructions in the [documentation](./data/README.md).

## Dependencies

Please follow the instructions in the [DARE repository](https://github.com/yule-BUAA/MergeLM) to install the dependencies.

Additionally, install the following Python packages:
- `scipy`
- `sklearn`
- `torchmetrics`
- `evaluate`

## Get Started

The `run_merge_roberta.sh` script contains instructions for fusing the RoBERTa model.
To get started, run the script:
```bash
bash run_merge_roberta.sh
```
### Parameters:

**--drop_ratio:** Set dropout rate.

**--group_size:** Specifies the size of the groups during task vector compression.

**--eval_subset_size:** Represents the maximum size of the evaluation dataset.

**--knn_samples_per_dataset:** Defines the number of samples collected for each dataset in KNN.

**--knn_neighbors:** Specifies the number of neighbors to use in the KNN algorithm.

