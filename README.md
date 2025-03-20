# Binary-Task-Switch
This is the official implementation of our CVPR2025 Paper：[Less is More: Efficient Model Merging with Binary Task Switch.](https://arxiv.org/abs/2412.00054), by Biqing Qi, Fangyuan Li, Zhen Wang, Junqi Gao, Dong Li, Peng Ye, Bowen Zhou.

# News!
1. On March 20, 2025, we are building our code repository.

# Abstract
As an effective approach to equip models with multi-task capabilities without additional training, model merging has garnered significant attention. However, existing merging methods face challenges of redundant parameter conflicts and the excessive storage burden of fine-tuned parameters. In this work, through controlled experiments, we reveal that for fine-tuned task vectors, only those parameters with magnitudes above a certain threshold contribute positively to the task, exhibiting a pulse-like characteristic. We then attempt leveraging this pulse-like characteristic to binarize the task vectors and reduce storage overhead. Further controlled experiments show that the binarized task vectors incur almost no decrease in fine-tuning and merging performance, and even exhibit stronger performance improvements as the proportion of redundant parameters increases. Based on these insights, we propose Task Switch (T-Switch), which decomposes task vectors into three components: 1) an activation switch instantiated by a binarized mask vector, 2) a polarity switch instantiated by a binarized sign vector, and 3) a scaling knob instantiated by a scalar coefficient. By storing task vectors in a binarized form, T-Switch alleviates parameter conflicts while ensuring efficient task parameter storage. Furthermore, to enable automated switch combination in T-Switch, we further introduce Auto-Switch, which enables training-free switch combination via retrieval from a small query set. Experiments indicate that our methods achieve significant performance improvements over existing baselines, requiring only 1-3$\%$ of the storage space of full-precision parameters.

# Figure Explanation

<img src="https://github.com/lfy-123/Binary-Task-Switch/blob/main/jpg/merge_illustration.jpg" width="500px">

**Figure1**: Left: Challenges of model merging: conflicts in task vectors and the burden of parameter storage. Right: Our method eliminates redundancy while enabling the storage of binarized, lightweight task vectors.

<img src="https://github.com/lfy-123/Binary-Task-Switch/blob/main/jpg/merge_method.jpg" width="800px">

**Figure2**: Overview of our method: T-Switch and Auto-Switch. The left side illustrates the construction process of the task switch, where noise parameters in the task vectors are discarded, and the remaining parameters are binarized to form the task switch. The upper right corner shows the inference process of our T-Switch using the task switch. The lower right corner demonstrates how our Auto-Switch automatically selects the task switch based on data features.

# Installation

**For Merging ViT Experiment:**
```bash
Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.
```

**For LORA Experiment:**
After installing the previous step, install peft.
```bash
pip install peft
```

**For Merging Roberta Experiment:**

Please follow [DARE](https://github.com/yule-BUAA/MergeLM) to install the dependencies.
Additionally, install scipy, sklearn, torchmetrics, evaluate.





# Acknowledgement

Our implementation references the code below, thanks to them.

EMR-Merging: https://github.com/harveyhuang18/EMR_Merging

FusionBench: https://github.com/tanganke/fusion_bench/tree/main/fusion_bench/method

Task Arithmetic: https://github.com/mlfoundations/task_vectors

TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

DARE: https://github.com/yule-BUAA/MergeLM



