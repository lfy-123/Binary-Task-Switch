import torch
import numpy as np
from bitarray import bitarray

def compress_to_bitarray(tensor, group_size=4):
    tensor = tensor.flatten().cpu().numpy()  
    
    n = len(tensor) // group_size  
    if group_size > tensor.size:
        group_size = tensor.size
    reshaped_tensor = tensor.reshape(-1, group_size) 
    mark_array = np.any(reshaped_tensor != 0, axis=1) 
    mark_str = ''.join('1' if mark else '0' for mark in mark_array) 

    compressed_bits = bitarray(mark_str)
    mask_len = len(compressed_bits)

    mapping = {0: '00', 1: '01', -1: '10'}
    flattened_group = reshaped_tensor.flatten()
    compressed_group = np.array([mapping[x] for x in flattened_group]) 

    all_ones = 0
    for i, mark in enumerate(mark_array):
        if mark: 
            all_ones += 1
            group = reshaped_tensor[i]
            
            non_zero_indices = [index for index, value in enumerate(group) if value != 0]
            num_non_zero = len(non_zero_indices)
            
            compressed_group = ''.join([str(int(abs(group[idx]))) for idx in range(group_size)])
            
            signs = ''.join(['1' if group[idx] == 1 else '0' for idx in non_zero_indices]) 
            compressed_bits.extend(compressed_group)
            compressed_bits.extend(signs) 

    print(f"Proportion of non-zero elements: {all_ones}/{n}")
    return compressed_bits


def decompress_from_bitarray(compressed_bits, tensor_shape, group_size=4):
    if tensor_shape[0] * tensor_shape[1] < group_size:
        group_size = tensor_shape[0] * tensor_shape[1]
    mark_len = tensor_shape[0] * tensor_shape[1] // group_size
    mark_bits = compressed_bits[:mark_len]
    
    decompressed_tensor = np.zeros(tensor_shape[0] * tensor_shape[1])
    reshaped_tensor = decompressed_tensor.reshape(-1, group_size)

    compressed_data = compressed_bits[mark_len:]

    num_idx = 0
    for i, mark in enumerate(mark_bits):
        if mark == 1:
            group_bits = compressed_data[num_idx:num_idx+group_size]
            non_zero_indices = [index for index, value in enumerate(group_bits) if value != 0]
            num_non_zero = len(non_zero_indices)
            num_idx += group_size

            signs = compressed_data[num_idx:num_idx+num_non_zero]
            num_idx += num_non_zero

            group = []
            non_idx = 0
            for idx, bit in enumerate(group_bits):
                if bit == 1 and signs[non_idx] == 1:
                    group.append(1)
                    non_idx += 1
                elif bit == 1 and signs[non_idx] == 0:
                    group.append(-1)
                    non_idx += 1
                elif bit == 0:
                    group.append(0)
                else:
                    group.append(0)
                
            reshaped_tensor[i] = group

    return decompressed_tensor.reshape(tensor_shape)


def compress_tensors(tensor_dict, group_size=4):
    compressed_dict = {}
    for name, tensor in tensor_dict.items():       
        pre_length = torch.norm(tensor.float(), p=2)
        binary_tensor = torch.where(
                tensor > 0, 
                torch.tensor(1, dtype=torch.int8, device=tensor.device), 
                torch.where(tensor < 0, torch.tensor(-1, dtype=torch.int8, device=tensor.device), torch.tensor(0, dtype=torch.int8, device=tensor.device))
                )
        now_length = torch.norm(binary_tensor.float(), p=2)
        scaling = pre_length / now_length
        print(f"Compress parameter: {name}")
        shape = binary_tensor.shape
        if binary_tensor.dim() >= 1:
            compressed_tensor = compress_to_bitarray(binary_tensor, group_size=group_size) 
        else:
            print(f"Skip scalar")
            compressed_tensor = binary_tensor.to(torch.int8)
        compressed_dict[name] = {
            'tensor': compressed_tensor,
            'shape': shape
        }
    return compressed_dict

def decompress_tensors(compressed_dict, group_size=4):
    decompressed_dict = {}
    for name, matrices in compressed_dict.items():
        compressed_tensor = matrices['tensor']
        shape = matrices['shape']

        tensor = torch.tensor(decompress_from_bitarray(compressed_tensor, shape, group_size=group_size))
        decompressed_dict[name] = tensor.to(torch.bfloat16)
    return decompressed_dict


if __name__ == "__main__":
    rows = 400
    cols = rows
    tensor_dict = {
        'param1': torch.randint(low=-1, high=2, size=(rows, cols), dtype=torch.float32),
        'param2': torch.randint(low=-1, high=2, size=(rows, cols), dtype=torch.float32)
    }
    print(f"tensor_dict:{tensor_dict}")
    compressed_dict = compress_tensors(tensor_dict)
    print("Compressed Dictionary:")
    
    decompressed_dict = decompress_tensors(compressed_dict)
    print("Decompressed Dictionary:")
    print(decompressed_dict)

    for (name1,param1), (name2,param2) in zip(tensor_dict.items(), decompressed_dict.items()):
        if torch.equal(param1, param2):
            print(f"Before and after compression, the parameter {name1} remains consistent")
        else:
            print(f"Before and after compression, the parameter {name1} is inconsistent")

