import copy
import os
from tqdm import tqdm
import sys
import argparse
from functools import partial
import time
import logging
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from collections import OrderedDict
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from utils.utils import set_random_seed
from sklearn.neighbors import KNeighborsClassifier
from utils.compress_util import compress_tensors
from transformers import AutoConfig
from customized_roberta import KNNEnhancedRobertaForSequenceClassification


# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Interface for merging roberta models on glue")
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["roberta-base"])
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
parser.add_argument("--drop_ratio", type=float, default=0.0, help="Task vector drop ratio")
parser.add_argument("--group_size", type=int, default=4, help="The size of the group during compression")
parser.add_argument("--eval_subset_size", type=int, default=500, help="Set the maximum number of evaluation sets")
parser.add_argument("--knn_samples_per_dataset", type=int, default=100, help="The number of samples per dataset for knn")
parser.add_argument("--knn_neighbors", type=int, default=10, help="The number of neighbors for knn")



args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"



def process_matrix(matrix, k, name):
    if k > 1:
        k = k / 100
    positive_matrix = torch.where(matrix > 0, matrix, torch.zeros_like(matrix))
    negative_matrix = torch.where(matrix < 0, matrix, torch.zeros_like(matrix))

    positive_non_zero = positive_matrix[positive_matrix != 0]
    if len(positive_non_zero) > 0:
        num_elements_to_keep = int(len(positive_non_zero) * (1 - k))
        if num_elements_to_keep > 0:
            positive_threshold = torch.topk(positive_non_zero, num_elements_to_keep, largest=True).values.min()
            positive_matrix = torch.where(positive_matrix >= positive_threshold, positive_matrix, torch.zeros_like(positive_matrix))
        else:
            logger.info(f"Too few positive elements in {name}.")
    else:
        logger.info(f"No non-zero positive elements found in matrix {name}.")

    negative_abs_matrix = torch.abs(negative_matrix)
    negative_non_zero = negative_abs_matrix[negative_abs_matrix != 0]
    if len(negative_non_zero) > 0:
        num_elements_to_keep = int(len(negative_non_zero) * (1 - k))
        if num_elements_to_keep > 0:
            negative_threshold = torch.topk(negative_non_zero, num_elements_to_keep, largest=True).values.min()
            negative_matrix = torch.where(negative_abs_matrix >= negative_threshold, negative_matrix, torch.zeros_like(negative_matrix))
        else:
            logger.info(f"Too few negative elements in {name}.")
    else:
        logger.info(f"No non-zero negative elements found in matrix {name}.")
    return positive_matrix, negative_matrix


def drop_elements_separately(pretrained_model, finetuned_model, drop_ratio):
    task_vector = {}
    for (name, param_p), (_, param_f) in zip(pretrained_model.named_parameters(), finetuned_model.named_parameters()):
        if "classifier" not in name:
            difference_matrix = param_f.data - param_p.data
            if difference_matrix.dim() >= 1:       
                positive_elements_result, negative_elements_result = process_matrix(difference_matrix, drop_ratio, name)
                task_vector[name] = positive_elements_result + negative_elements_result
            else:
                task_vector[name] = difference_matrix
        else:
            pass
    return task_vector


def binarize_matrix(task_vector):
    binary_matrix_dict = {}
    scaling_factor_dict = {}
    for name, differ in task_vector.items():
        if differ.dim() >= 1:
            binary_matrix = torch.where(differ > 0, 1.0, torch.where(differ < 0, -1.0, 0.0))
            l2_norm_original = torch.norm(differ, p=2)
            l2_norm_binary = torch.norm(binary_matrix, p=2)
            scaling_factor = l2_norm_original / l2_norm_binary
            
            task_vector[name] = binary_matrix * scaling_factor

            binary_matrix_dict[name] = binary_matrix
            scaling_factor_dict[name] = scaling_factor
    return task_vector, binary_matrix_dict, scaling_factor_dict


def load_task_vector(pretrained_model, task_vector):
    pretrained_model_copy = copy.deepcopy(pretrained_model)
    for name, param in pretrained_model_copy.named_parameters():
        if name in task_vector and "classifier" not in name:
            param.data += task_vector[name].to(param.device)
    return pretrained_model_copy


def eval_model_on_dataset(args, eval_model, dataset_name, classifier_dict, trainer, tokenizer):
    eval_model_training_args = TrainingArguments(
        output_dir=f"./output/roberta/{dataset_name}",
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        report_to="none"
    )

    eval_model.classifier = classifier_dict[dataset_name]
    
    evaluator = CustomizedTrainer(
        model=eval_model,  # final merged model
        args=eval_model_training_args,  # training arguments
        eval_dataset=trainer.eval_dataset.select(range(min(args.eval_subset_size, len(trainer.eval_dataset)))),  # evaluation dataset
        compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
        tokenizer=tokenizer  # tokenizer
    )
    with torch.no_grad():
        metrics = evaluator.evaluate()
    return {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in metrics.items()}


def reconstruct_binary_and_length(binary_list_dict: list[dict], length_list_dict: list[dict]):
    """
    The purpose of this function is to reformat the obtained binary matrix (dictionary) and length information (dictionary) 
    for easy embedding and use in subsequent code
    """
    binary_dict_list_dict = {}
    length_dict_list_dict = {}
    mean_dict_list_dict = {}
    for idx, (binary_dict, length_dict) in enumerate(zip(binary_list_dict, length_list_dict)):
        keys_list = list(binary_dict.keys())
        for model_key in keys_list:
            if "layer" in model_key:
                key_part = model_key.split(".")
                layer_idx = key_part[3]
                index = model_key.find(layer_idx)
                outside_key = model_key[:index+len(layer_idx)]
                if outside_key not in binary_dict_list_dict:
                    binary_dict_list_dict[outside_key] = []
                    length_dict_list_dict[outside_key] = []
                    for _ in range(len(binary_list_dict)):
                        binary_dict_list_dict[outside_key].append({})
                        length_dict_list_dict[outside_key].append({})
                inside_key = model_key[index+len(layer_idx)+1:]
                binary_dict_list_dict[outside_key][idx][inside_key] = binary_dict[f"{outside_key}.{inside_key}"]
                length_dict_list_dict[outside_key][idx][inside_key] = length_dict[f"{outside_key}.{inside_key}"]
            elif "embeddings" in model_key:
                index = model_key.find("embeddings")
                outside_key = model_key[:index+len("embeddings")]
                if outside_key not in binary_dict_list_dict:
                    binary_dict_list_dict[outside_key] = []
                    length_dict_list_dict[outside_key] = []
                    for _ in range(len(binary_list_dict)):
                        binary_dict_list_dict[outside_key].append({})
                        length_dict_list_dict[outside_key].append({})
                inside_key = model_key[index+len("embeddings")+1:]
                binary_dict_list_dict[outside_key][idx][inside_key] = binary_dict[f"{outside_key}.{inside_key}"]
                length_dict_list_dict[outside_key][idx][inside_key] = length_dict[f"{outside_key}.{inside_key}"]
    return binary_dict_list_dict, length_dict_list_dict


def compute_avg(metrics):
    if len(metrics) == 0:
        return 0.0
    all_score = 0.0
    for dataset_name, data_dict in metrics.items():
        if "eval_accuracy" in data_dict.keys():
            all_score += data_dict["eval_accuracy"]
        elif "eval_matthews_correlation" in data_dict.keys():
            all_score += data_dict["eval_matthews_correlation"]
        elif "eval_averaged_scores" in data_dict.keys():
            all_score += data_dict["eval_averaged_scores"]
        else:
            raise("wrong")
    return all_score / len(metrics)


def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, tokenizer: transformers.AutoTokenizer):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path).to(args.device) 
    
    set_random_seed(seed=0)
    classifier_dict = {}
    for dataset_name, finetuned_model in zip(args.dataset_names, models_to_merge):
        classifier_dict[dataset_name] = copy.deepcopy(finetuned_model.classifier)
        
    logger.info(f"------------------------------------ Start positive and negative dropout - sparsification operation ------------------------------------")
    task_vectors_dict = {}
    test_metrics = {}
    for dataset_name, finetuned_model_temp, trainer in zip(args.dataset_names, models_to_merge, trainers):

        task_vector = drop_elements_separately(pretrained_model, finetuned_model_temp, args.drop_ratio)

        task_vectors_dict[dataset_name] = task_vector
        
        # Evaluate the discarded task vector
        sparse_by_drop_model = load_task_vector(pretrained_model, task_vector)
        test_metric = eval_model_on_dataset(args, sparse_by_drop_model, dataset_name, classifier_dict, trainer, tokenizer)
        test_metrics[dataset_name] = test_metric
        torch.cuda.empty_cache()        
    logger.info(f"After discarding {args.drop_ratio}, the result is {test_metrics}")
    logger.info(f"After discarding {args.drop_ratio}, the evaluation average result is {compute_avg(test_metrics)}")
    
    
    binary_task_vectors_dict = {}
    binary_list_dict = []
    length_list_dict = []
    test_metrics = {}
    all_times = {}
    
    logger.info(f"------------------------------------ Start binarization - sparse operation again ------------------------------------")
    for dataset_name, trainer in zip(args.dataset_names, trainers):
        binary_task_vector, binary_matrix_dict, scaling_factor_dict = binarize_matrix(task_vectors_dict[dataset_name])
        
        binary_task_vectors_dict[dataset_name] = binary_task_vector
        binary_list_dict.append(binary_matrix_dict)
        length_list_dict.append(scaling_factor_dict)
        
        sparse_by_binary_model = load_task_vector(pretrained_model, binary_task_vector)
        
        # Evaluate binary task vectors
        start_time = time.time()
        test_metric = eval_model_on_dataset(args, sparse_by_binary_model, dataset_name, classifier_dict, trainer, tokenizer)
        end_time = time.time()
        test_metrics[dataset_name] = test_metric
        torch.cuda.empty_cache()
        all_times[dataset_name] = end_time - start_time

        # Save and compress binary task vectors
        compressed_dict = compress_tensors(binary_matrix_dict, group_size=args.group_size)
        bin_path = f"./output/roberta_binary_drop_{args.drop_ratio}_group_{args.group_size}/{dataset_name}"
        os.makedirs(bin_path, exist_ok = True)
        for name, ldict in compressed_dict.items():
            bit_param = ldict['tensor']
            if not isinstance(bit_param, torch.Tensor):
                bin_filename = f"{bin_path}/{dataset_name}_{name}_compressed.bin"
                with open(bin_filename, 'wb') as f:
                     bit_param.tofile(f)
        torch.save(scaling_factor_dict, f"./output/roberta_binary_drop_{args.drop_ratio}_group_{args.group_size}/scaling_factor_dict.pt")
    
    logger.info(f"Evaluate reasoning time：{all_times}")
    logger.info(f"After binarizing, the result is {test_metrics}")
    logger.info(f"After binarizing, the average result is {compute_avg(test_metrics)}")


    logger.info(f"------------------------------------ Start using KNN merge model for inference ------------------------------------")
    binary_dict_list_dict, length_dict_list_dict = reconstruct_binary_and_length(binary_list_dict, length_list_dict)
    test_metrics, all_elapsed_time = infer_with_knn_and_task_features(pretrained_model, binary_dict_list_dict, length_dict_list_dict, classifier_dict, trainers, args)

    logger.info(f"The average of the KNN inference result is {compute_avg(test_metrics)}")

    
    all_time = [all_elapsed_time[dataset] for dataset in all_elapsed_time]
    mean_time = sum(all_time) / len(all_time)
    logger.info(f"The total evaluation time for KNN is {sum(all_time):.2f}, average time is {mean_time:.2f}")
    
    
    
def prepare_mixed_dataset_knn(pretrained_model, trainers, args):    
    pretrained_model.eval()
    all_features = []
    all_labels = []

    for dataset_index, dataset_name in enumerate(args.dataset_names):
        logger.info(f"Collect samples from {dataset_name} as knn seed data")
        dataloader = trainers[dataset_index].get_train_dataloader()
        collected = 0
        features_list = []

        for inputs in tqdm(dataloader, desc=f"采样 {dataset_name}", leave=False):
            if collected >= args.knn_samples_per_dataset:
                break

            with torch.no_grad():
                inputs_no_labels = {k: v.to(args.device) for k, v in inputs.items() if k != "labels"}
                outputs = pretrained_model.roberta(**inputs_no_labels)
                feature_batch = outputs[0][:,0,:]

            remaining = args.knn_samples_per_dataset - collected
            batch_features = feature_batch[:remaining].detach().cpu()  # [B, H]
            features_list.append(batch_features)
            all_labels.extend([dataset_index] * batch_features.size(0))
            collected += batch_features.size(0)

            del feature_batch, outputs
            torch.cuda.empty_cache()

        if features_list:
            all_features.append(torch.cat(features_list, dim=0))  # [N, H]
    logger.info(f"All dataset samples have been sampled, with a total of {len(all_labels)} feature samples")
    return torch.cat(all_features, dim=0), all_labels


def infer_with_knn_and_task_features(pretrained_model, binary_dict_list_dict, length_dict_list_dict, classifier_dict, trainers, args):    
    mixed_data, mixed_dataset_labels = prepare_mixed_dataset_knn(pretrained_model, trainers, args)
    knn = KNeighborsClassifier(n_neighbors=args.knn_neighbors)
    knn.fit(mixed_data, mixed_dataset_labels)
    metrics = {}
    
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        output_hidden_states=True
    )
    
    base_model = KNNEnhancedRobertaForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        config=config,
        binary_dict_list_dict=binary_dict_list_dict,
        length_dict_list_dict=length_dict_list_dict,
        knn=knn,
        dataset_names=args.dataset_names,
    )
    
    
    test_metrics={}
    all_elapsed_time = {}
    for dataset_name, trainer in zip(args.dataset_names, trainers):
        eval_model_training_args = TrainingArguments(
            output_dir=f"./output/roberta/{dataset_name}",
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            report_to="none"
        )

        base_model.classifier = classifier_dict[dataset_name]

        eval_model_evaluator = CustomizedTrainer(
            model=base_model,  # final merged model
            args=eval_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset.select(range(min(args.eval_subset_size, len(trainer.eval_dataset)))),  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer,  # tokenizer
        )

        start_time = time.time()
        with torch.no_grad():
            test_metric = eval_model_evaluator.evaluate()
            test_metric = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metric.items()}
            logger.info(f"infer with knn on dataset {dataset_name}: {test_metric}")
            test_metrics[dataset_name] = test_metric
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Knn evaluation dataset {dataset_name} takes {elapsed_time:.2f} seconds")
        all_elapsed_time[dataset_name] = elapsed_time
    return test_metrics, all_elapsed_time


if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    assert all([dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
        'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]!'
    
    args.pretrained_model_path = "./ckpts/roberta-base"

    load_model_paths = []
    for dataset_name in args.dataset_names:
        load_model_paths.append(f"./ckpts/roberta/{dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    # load the checkpoint of each individual model that needs to be merged
    models_to_merge, trainers, = [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=f"./output/roberta/{dataset_name}", 
            per_device_train_batch_size=args.batch_size,       # batch size per device during training
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
            report_to="none"
        )

        model_to_merge = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=load_model_path,
            num_labels=num_labels,
            output_hidden_states=True
        ).to(args.device)

        model_to_merge.config.output_hidden_states = True
        trainer = CustomizedTrainer(
            model=model_to_merge,               # model to be merged
            args=training_args,                 # training arguments
            train_dataset=train_dataset,        # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        
        models_to_merge.append(model_to_merge.to(args.device))
        trainers.append(trainer)
    
    logger.info(f"********** Run starts. **********")
    get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, tokenizer=tokenizer)
    
    
    
    