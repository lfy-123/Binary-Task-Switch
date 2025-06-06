# Merging Vision Transformers (ViTs)

## Datasets
Please follow [Adamerging](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets) to download the datasets. And put datasets in the **data** folder

Please follow [doc](./data/README.md) to place these datasets.

## Checkpoints
You can download the fine-tuned checkpoints from the [huggingface](https://huggingface.co/lfy-hg/ViTs-on-multi-dataset).

Please follow [doc](./ckpts/README.md) to place these checkpoints.

## Dependencies
Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.

**For LORA Experiment:**
After installing the above environment, it is necessary to install the PEFT.
```bash
pip install peft
```

## Get Started
You can find the instructions for merging ViT-B-32, ViT-L-14, ViT-B-32+LoRA from the [run_merge_vit.sh](./run_merge_vit.sh).
```bash
bash run_merge_vit.sh
```

### Parameters:

**--drop-ratio:** Set dropout rate.

**--group_size:** Specifies the size of the groups during task vector compression.

**--knn-pre-sample-num:** Defines the number of samples collected for each dataset in KNN.

**--knn-neighbors:** Specifies the number of neighbors to use in the KNN algorithm.





