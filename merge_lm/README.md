 # Merging Language Models (LMs)

## Checkpoints

**RoBERTa**: You can download the fine-tuned checkpoints from huggingface [here]().

Place the checkpoints as follows:

```
│cktps/
├──roberta/
│  ├── cola/
│  │  ├── config.json
│  │  ├──......
│  ├── sst2/
│  │  ├── config.json
│  │  ├──......
│  ├── ......
├──roberta-base/
│  ├── config.json
│  ├──......
```

## Data
The [**data**]() folder contains two folders, one for evaluation and one for the glue dataset
https://github.com/huggingface/evaluate

## Dependencies

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to install the dependencies.

Additionally, install scipy, sklearn, torchmetrics, evaluate.



You can modify the `cache_dir` in the `utils/load_config.py` file to specify your own path to save the datasets.

## Eval

#### Merge RoBERTa models

> python merge_roberta_glue.py

#### Merge GPT-2 models

> python merge_gpt_glue.py

## Results

Results for our EMR-Merging will be saved in ./save_merge_logs.
