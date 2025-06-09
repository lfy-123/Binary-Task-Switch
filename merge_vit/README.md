# Merging Vision Transformers (ViTs)

## Datasets
Please follow [Adamerging](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets) to download the datasets. And put datasets in the **data** folder

Please follow [doc](./data/README.md) to place these datasets.

## Checkpoints
You can download the fine-tuned checkpoints from the huggingface [here](https://huggingface.co/lfy-hg/ViTs-on-multi-dataset).

Please follow [doc](./ckpts/README.md) to place these checkpoints.

## Dependencies

You can create an environment **merge** using the following command:

```bash
conda env create -f environment.yml
```

**For LORA Experiment:**
After installing the above environment, it is necessary to install the PEFT.
```bash
conda create -n merge_lora --clone merge
conda activate merge_lora
pip install peft
```

## Get Started
You can find the all commands for merging ViT-B-32, ViT-L-14 and ViT-B-32+LoRA from the [run_merge_vit.sh](./run_merge_vit.sh).

Merging eight ViT-B-32 models:
```python
conda activate merge
python merge_vit.py --model-type "ViT-B-32" \
--drop-ratio 0.6 \
--group_size 4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256
```
Merging eight ViT-L-14 models:
```python
conda activate merge
python merge_vit.py --model-type "ViT-L-14" \
--drop-ratio 0.6 \
--group_size 4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256
```
Merging eight ViT-B-32 LoRA models:
```python
conda activate merge_lora
python merge_vit_lora.py --model-type "ViT-B-32" \
--drop-ratio 0.4 \
--group_size 4 \
--lora-rank 64 --lr 1e-4 \
--knn-neighbors 10 \
--knn-pre-sample-num 100 \
--batch-size 256
```

### Parameters:

**--drop-ratio:** Set dropout rate.

**--group_size:** Specifies the size of the groups during task vector compression.

**--knn-pre-sample-num:** Defines the number of samples collected for each dataset in KNN.

**--knn-neighbors:** Specifies the number of neighbors to use in the KNN algorithm.





