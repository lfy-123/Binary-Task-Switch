# Merging Vision Transformers (ViTs)

## Datasets
Please follow [Adamerging](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets) to download the datasets. And put datasets in the **data** folder

Please follow [doc](./data/README.md) to place these datasets.

## Checkpoints
You can download the fine-tuned checkpoints from the [Google Drive folder]()

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
> bash run_merge_vit.sh






