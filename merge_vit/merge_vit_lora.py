import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from peft import LoraConfig, get_peft_model, PeftModel
import torchvision
from args import parse_arguments
from tqdm import tqdm
from heads import get_classification_head
from src.modeling import ImageClassifier
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
import os
from sklearn.neighbors import KNeighborsClassifier
import utils
import sys
import time
import json
from compress_util import compress_tensors, decompress_tensors
from safetensors.torch import load_file
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_single_dataset(model, dataset_name, args):
    model.eval()
    if hasattr(model, 'val_preprocess'):
        val_preprocess = model.val_preprocess
    elif hasattr(model.model, 'val_preprocess'):
        val_preprocess = model.model.val_preprocess

    # Prepare for other versions of torchvision to prevent errors
    if val_preprocess and hasattr(val_preprocess, 'transforms'):
        for t in val_preprocess.transforms:
            if isinstance(t, torchvision.transforms.RandomResizedCrop):
                try:
                    t.antialias = False
                except AttributeError:
                    pass

    dataset = get_dataset(
        dataset_name,
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=12
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"evaling on {dataset_name}")):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    logger.info(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics


def add_lora_to_model(pretrained_model, lora_rank, lora_alpha):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            'conv1',
            'c_fc',
            'c_proj'
        ],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(pretrained_model, lora_config)
    return lora_model


def train_lora_model(lora_model, dataloader, device, num_epochs=5, lr=1e-5):
    lora_model = lora_model.to(device)
    lora_model.train()
    
    for name, param in lora_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            inputs, labels = data['images'].to(device), data['labels'].to(device)

            optimizer.zero_grad()
            outputs = lora_model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)            
            total += labels.size(0)

        epoch_loss = running_loss / total

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        scheduler.step()
    logger.info("LoRA training completed.")
    return lora_model


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
            logger.warning(f"Too few positive elements in {name}.")
    else:
        logger.warning(f"No non-zero positive elements found in matrix {name}.")

    negative_abs_matrix = torch.abs(negative_matrix)
    negative_non_zero = negative_abs_matrix[negative_abs_matrix != 0]
    if len(negative_non_zero) > 0:
        num_elements_to_keep = int(len(negative_non_zero) * (1 - k))
        if num_elements_to_keep > 0:
            negative_threshold = torch.topk(negative_non_zero, num_elements_to_keep, largest=True).values.min()
            negative_matrix = torch.where(negative_abs_matrix >= negative_threshold, negative_matrix, torch.zeros_like(negative_matrix))
        else:
            logger.warning(f"Too few negative elements in {name}.")
    else:
        logger.warning(f"No non-zero negative elements found in matrix {name}.")
    return positive_matrix, negative_matrix


def drop_elements_separately_lora(finetuned_model, drop_ratio):
    task_vector = {}
    for name_f, param_f in finetuned_model.named_parameters():
        if "lora" not in name_f:
            continue
        if param_f.dim() >= 1:
            positive_elements_result, negative_elements_result = process_matrix(param_f, drop_ratio, name_f)
            task_vector[name_f] = positive_elements_result + negative_elements_result
        else:
            task_vector[name_f] = difference_matrix
    return task_vector


def load_task_vector_lora(lora_model, task_vector):
    for name, param in lora_model.named_parameters():
        if name in task_vector:
            print(f"加载模型参数:{name}")
            param.data.copy_(task_vector[name].to(param.device))


def binarize_matrix(task_vector):
    binary_matrix_dict = {}
    scaling_factor_dict = {}
    for name, differ in task_vector.items():
        if torch.all(differ.eq(0)):
            continue
        if differ.dim() >= 1:
            binary_matrix = torch.where(differ > 0, 1.0, torch.where(differ < 0, -1.0, 0.0))
            l2_norm_original = torch.norm(differ, p=2)
            l2_norm_binary = torch.norm(binary_matrix, p=2)
            scaling_factor = l2_norm_original / l2_norm_binary

            task_vector[name] = binary_matrix * scaling_factor
            
            binary_matrix_dict[name] = binary_matrix.to(torch.int8)
            scaling_factor_dict[name] = scaling_factor
    return task_vector, binary_matrix_dict, scaling_factor_dict


def prepare_mixed_dataset_knn(datasets_name, shared_image_encoder, args):
    """
    Extract image features using the shared_image_encoder and assign dataset labels for k-NN.
    """
    logger.info("Start extracting features from image datasets for k-NN...")

    shared_image_encoder = shared_image_encoder.to(args.device)
    shared_image_encoder.eval()

    all_features = []
    all_labels = []

    train_preprocess = getattr(shared_image_encoder, 'train_preprocess', None)
    if train_preprocess is None and hasattr(shared_image_encoder, 'model'):
        train_preprocess = getattr(shared_image_encoder.model, 'train_preprocess', None)

    # Prepare for other versions of torchvision to prevent errors
    if train_preprocess and hasattr(train_preprocess, 'transforms'):
        for t in train_preprocess.transforms:
            if isinstance(t, torchvision.transforms.RandomResizedCrop):
                try:
                    t.antialias = False
                except AttributeError:
                    pass

    for dataset_index, dataset_name in enumerate(datasets_name):
        logger.info(f"Collect samples from {dataset_name} as knn seed data")

        dataset = get_dataset(
            dataset_name,
            train_preprocess,
            location=args.data_location,
            batch_size=args.knn_pre_sample_num,
            num_workers=12
        )

        dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

        collected = 0
        features_list = []

        for images, _ in dataloader:
            if collected >= args.knn_pre_sample_num:
                break

            images = images.to(args.device)

            with torch.no_grad():
                features = utils.get_logits(images, shared_image_encoder)  # [B, D]

            remaining = args.knn_pre_sample_num - collected
            batch_features = features[:remaining].detach().cpu()  # [B, D]
            features_list.append(batch_features)
            all_labels.extend([dataset_index] * batch_features.size(0))
            collected += batch_features.size(0)

            del features
            torch.cuda.empty_cache()

        if features_list:
            all_features.append(torch.cat(features_list, dim=0))  # [N, D]

    logger.info(f"All datasets sampled, total feature samples: {len(all_labels)}")
    return torch.cat(all_features, dim=0), all_labels


def add_weighted_specific_task_vectors_lora(lora_model, specific_task_vectors, specific_weights_mean, datasets_name, args=None):
    for name, param in lora_model.named_parameters():
        if name in specific_task_vectors[datasets_name[0]]:
            specific_tv = torch.zeros_like(param)
            for dataset_name in datasets_name:
                specific_tv += specific_task_vectors[dataset_name][name] * specific_weights_mean[dataset_name]
            param.data.copy_(specific_tv)


def infer_with_knn_and_task_features_lora(pretrained_image_encoder, lora_model, all_specific_task_vectors, datasets_name, args):
    # Step 1: Prepare the mixed data required for KNN
    mixed_data, mixed_dataset_labels = prepare_mixed_dataset_knn(datasets_name, pretrained_image_encoder, args)
    
    # Step 2: fit the KNN, with n_neighbors set as the number of neighbors
    knn = KNeighborsClassifier(n_neighbors=args.knn_neighbors)  # 这个参数设置为多少比较合适？？？
    knn.fit(mixed_data, mixed_dataset_labels)
    metrics = {}

    all_elapsed_time = {}
    for dataset_name in datasets_name:
        # classification_head = 
        # merge_model = ImageClassifier(lora_model.image_encoder, classification_head)
        lora_model.model.classification_head = get_classification_head(args, dataset_name)
        
        if hasattr(lora_model.image_encoder, 'train_preprocess'):
            val_preprocess = lora_model.image_encoder.val_preprocess
        elif hasattr(lora_model.image_encoder.model, 'train_preprocess'):
            val_preprocess = lora_model.image_encoder.model.val_preprocess
        
        # Prepare for other versions of torchvision to prevent errors
        if val_preprocess and hasattr(val_preprocess, 'transforms'):
            for t in val_preprocess.transforms:
                if isinstance(t, torchvision.transforms.RandomResizedCrop):
                    try:
                        t.antialias = False
                    except AttributeError:
                        pass
        
        # Prepare dataset and dataloader
        dataset = get_dataset(
            dataset_name,  # Only need one dataset to preprocess
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=12
        )
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=None
        )
        device = args.device
        
        start_time = time.time()
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Evaluating {dataset_name}")):
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                y = data['labels'].to(device)
                
                # Step 3: Extract features from the current batch of images
                feature_vector = pretrained_image_encoder(x)

                # Step 4: Use KNN to batch infer the probability of the dataset
                knn_probabilities_batch = knn.predict_proba(feature_vector.cpu())
                specific_weights_batch = []
                # Generate a specific weight dictionary for the probability results of each sample
                for sample_idx in range(knn_probabilities_batch.shape[0]):
                    specific_weight = {datasets_name[i]: knn_probabilities_batch[sample_idx][i] for i in range(len(datasets_name))}
                    specific_weights_batch.append(specific_weight)
                
                # Take the average probability of data in a batch
                specific_weights_mean = {dataset: 0.0 for dataset in datasets_name}
                for specific_weight in specific_weights_batch:
                    for dataset, weight in specific_weight.items():
                        specific_weights_mean[dataset] += weight
                for dataset in specific_weights_mean:
                    specific_weights_mean[dataset] /= len(specific_weights_batch)

                logger.info(f"The weight of each data during inference: {specific_weights_mean}")
                # Step 5: Merging Specific Task Vectors Based on KNN Probability
                add_weighted_specific_task_vectors_lora(lora_model, all_specific_task_vectors, specific_weights_mean, datasets_name, args)

                # Step 6: Batch reasoning and calculating logits
                logits = utils.get_logits(x, lora_model)

                # Step 7: Calculate the prediction and calculate the accuracy rate
                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            top1 = correct / n
            end_time = time.time()
            elapsed_time = end_time - start_time

            logger.info(f"The time taken by Auto_Switch to evaluate the dataset {dataset_name}: {elapsed_time:.2f} 秒")

        metrics[dataset_name] = {'top1': top1}
        all_elapsed_time[dataset_name] = elapsed_time
        logger.info(f'Done Auto_Switch evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics, all_elapsed_time


if __name__ == '__main__':
    args, _ = parse_arguments()
    args.data_location = './data'
    lora_save_dir = "./ckpts/lora_model"
    args.model_save = f"./ckpts/{args.model_type}"
    pretrained_checkpoint = f'{args.model_save}/zeroshot.pt'
    pretrained_image_encoder = torch.load(pretrained_checkpoint)
    pretrained_image_encoder.to(args.device)

    lora_rank=args.lora_rank
    lora_alpha=lora_rank*2

    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    total_lora_dict = {}
    finetuned_image_encoders_dict = {}
    
    for idx, dataset_name in enumerate(exam_datasets):
        lora_image_encoder_save_path = lora_save_dir + f"/lora_{dataset_name}"
        if not os.path.exists(lora_image_encoder_save_path):
            logger.info(f"******************* The lora model for the {dataset_name} dataset has not been trained yet, start training!!! *******************")
            
            if hasattr(pretrained_image_encoder, 'train_preprocess'):
                train_preprocess = pretrained_image_encoder.train_preprocess
            elif hasattr(pretrained_image_encoder.model, 'train_preprocess'):
                train_preprocess = pretrained_image_encoder.model.train_preprocess
            
            # Prepare for other versions of transformers to prevent errors
            if train_preprocess and hasattr(train_preprocess, 'transforms'):
                for t in train_preprocess.transforms:
                    if isinstance(t, torchvision.transforms.RandomResizedCrop):
                        try:
                            t.antialias = False
                        except AttributeError:
                            pass
                
            # Load the Lora module on the pre trained model
            pretrained_image_encoder_copy = deepcopy(pretrained_image_encoder)
            classification_head = get_classification_head(args, dataset_name)
            final_model = ImageClassifier(pretrained_image_encoder_copy, classification_head)
            final_lora_model = add_lora_to_model(final_model, lora_rank=lora_rank, lora_alpha=lora_alpha)
            
            dataset = get_dataset(
                dataset_name,
                train_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=12
            )
            dataloader = get_dataloader(
                dataset, is_train=True, args=args, image_encoder=None)

            final_lora_model = train_lora_model(final_lora_model, dataloader, args.device, num_epochs=args.epochs, lr=args.lr)
            
            os.makedirs(lora_image_encoder_save_path, exist_ok=True)
            final_lora_model.save_pretrained(lora_image_encoder_save_path)
            metrics = eval_single_dataset(final_lora_model, dataset_name, args)
            logger.info(f"After lora trainning, the accuracy of lora_model on {dataset_name}: {metrics['top1'] * 100:.2f}%")
        else:
            logger.info(f"******************* The lora model for dataset {dataset_name} has been trained!!! *******************")
            pretrained_image_encoder_copy = deepcopy(pretrained_image_encoder)
            classification_head = get_classification_head(args, dataset_name)
            final_model = ImageClassifier(pretrained_image_encoder_copy, classification_head)
            final_lora_model = PeftModel.from_pretrained(final_model, lora_image_encoder_save_path)
        finetuned_image_encoders_dict[dataset_name] = final_lora_model
        
        
    
    # Create a model for subsequent inference
    pretrained_image_encoder_copy = deepcopy(pretrained_image_encoder)
    classification_head = get_classification_head(args, dataset_name) 
    inference_model = ImageClassifier(pretrained_image_encoder_copy, classification_head)
    lora_inference_model = add_lora_to_model(inference_model, lora_rank=lora_rank, lora_alpha=lora_alpha)
    
    
    
    task_vectors_dict = {}
    all_acc_drop = []
    logger.info(f"------------------------------------ Start positive and negative dropout - sparsification operation ------------------------------------")
    for dataset_name in exam_datasets:
        finetuned_image_encoder = finetuned_image_encoders_dict[dataset_name]
        finetuned_image_encoder.to(args.device)
        
        task_vector = drop_elements_separately_lora(finetuned_image_encoder, args.drop_ratio)

        task_vectors_dict[dataset_name] = task_vector
        
        # Evaluate droped task vectors
        logger.info("Evaluate the discarded model")
        load_task_vector_lora(lora_inference_model, task_vector)
        lora_inference_model.model.classification_head = get_classification_head(args, dataset_name)
        metrics = eval_single_dataset(lora_inference_model, dataset_name, args)
        
        
        
        logger.info(f"After dropping {args.drop_ratio}, the accuracy of model on {dataset_name}: {metrics['top1'] * 100:.2f}%")
        all_acc_drop.append(metrics['top1'] * 100)
    logger.info(f"======================= After dropping separately, average accuracy: {sum(all_acc_drop)/len(all_acc_drop)}% =======================")

    del finetuned_image_encoders_dict
    
    
        
    binary_task_vectors_dict = {}
    all_acc = []
    all_times = {}
    logger.info(f"------------------------------------ Start binarization - sparse operation again ------------------------------------")    
    for dataset_name in exam_datasets:
        binary_task_vector, binary_matrix_dict, scaling_factor_dict = binarize_matrix(task_vectors_dict[dataset_name])
        binary_task_vectors_dict[dataset_name] = binary_task_vector
        
        
        # Evaluate binaried task vectors
        logger.info("Evaluate the binaried model")
        load_task_vector_lora(lora_inference_model, binary_task_vector)
        lora_inference_model.model.classification_head = get_classification_head(args, dataset_name)
        start_time = time.time()

        metrics = eval_single_dataset(lora_inference_model, dataset_name, args)

        end_time = time.time()
        all_times[dataset_name] = end_time - start_time
        
        all_acc.append(metrics['top1'] * 100)
        logger.info(f"After binarizing, the accuracy of model on {dataset_name}: {metrics['top1'] * 100:.2f}%")
        
        
        compressed_dict = compress_tensors(binary_matrix_dict, group_size=args.group_size)
        bin_path = f"./output/lora_{args.model_type}_drop_{args.drop_ratio}_group_{args.group_size}/{dataset_name}"
        os.makedirs(bin_path, exist_ok = True)
        for name, ldict in compressed_dict.items():
            bit_param = ldict['tensor']
            if not isinstance(bit_param, torch.Tensor):
                bin_filename = f"{bin_path}/{dataset_name}_{name}_compressed.bin"
                with open(bin_filename, 'wb') as f:
                    bit_param.tofile(f)
    
    
    logger.info(f"======================= Reasoning time:{all_times}s =======================")
    logger.info(f"======================= After binarizing, average accuracy:{sum(all_acc)/len(all_acc)}% =======================")




    logger.info(f"--------------------------------- Using Auto_Switch for inference evaluation ---------------------------------")
    metrics, all_elapsed_time = infer_with_knn_and_task_features_lora(pretrained_image_encoder, lora_inference_model, binary_task_vectors_dict, exam_datasets, args)
    top1_values = [metrics[dataset]['top1'] for dataset in metrics]
    top1_mean = sum(top1_values) / len(top1_values)
    logger.info(f"The average result of Auto_Switch is {top1_mean*100}%")
    
    all_time = [all_elapsed_time[dataset] for dataset in all_elapsed_time]
    mean_time = sum(all_time) / len(all_time)
    logger.info(f"The total time for Auto_Switch evaluation is {sum(all_time):.2f}s, average time:{mean_time:.2f}s")
    
    
    
    