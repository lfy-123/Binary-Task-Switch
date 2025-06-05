import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from utils.glue_data_loader import glue_data_num_labels_map, rev_glue_data_id_map

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path


import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import Repository, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    # is_fairscale_available,
)
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.dependency_versions_check import dep_version_check
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    # ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod


def get_gpu_used_memory(print_pre_name=""):
    # 获取当前使用的 GPU 内存（单位是字节）
    used_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    print(f"{print_pre_name} Current GPU used memory: {used_memory:.2f} MB", flush=True)
    

class CustomizedTrainer(Trainer):

    def __init__(self, use_multitask_setting: bool = False, *args, **kwargs):
        """
        Customized trainer with user-defined train loss function.
        :param use_multitask_setting: boolean, whether to use multitask setting
        """
        super(CustomizedTrainer, self).__init__(*args, **kwargs)
        self.use_multitask_setting = use_multitask_setting

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False):
        """
        how the loss is computed by CustomizedTrainer
        :param model: nn.Module
        :param inputs: dict, model inputs
        :param return_outputs: boolean, whether return the outputs or not
        :return:
        """
        assert "labels" in inputs, "labels are not involved in inputs!"
        labels = inputs.pop("labels")
        if self.use_multitask_setting:
            assert "dataset_ids" in inputs.keys(), "key dataset_ids is missing in the inputs!"
            # Tensor
            dataset_ids = inputs["dataset_ids"]
            outputs = model(**inputs)
            logits = outputs["logits"]
            total_loss = None
            for dataset_id in dataset_ids.unique():
                single_dataset_indices = dataset_ids == dataset_id
                single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
                # cross-entropy loss for classification
                if single_dataset_num_labels > 1:
                    loss = F.cross_entropy(input=logits[single_dataset_indices][:, :single_dataset_num_labels], target=labels[single_dataset_indices].long())
                # mse loss for regression
                else:
                    assert single_dataset_num_labels == 1, "wrong number of labels!"
                    loss = F.mse_loss(input=logits[single_dataset_indices][:, 0], target=labels[single_dataset_indices])
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            return (total_loss, outputs) if return_outputs else total_loss
        else:
            outputs = model(**inputs)
            logits = outputs["logits"]
            if logits.shape[1] > 1:
                # cross-entropy loss for classification
                loss = F.cross_entropy(input=logits, target=labels.long())
            else:
                # mse loss for regression
                assert logits.shape[1] == 1, "wrong number of labels!"
                loss = F.mse_loss(input=logits.squeeze(dim=1), target=labels)
            return (loss, outputs) if return_outputs else loss
