import torch
import numpy as np

def print_tensor_shapes(data_dict):
    """
    Print the keys and shapes of tensor/array values in a dictionary.
    
    Args:
        data_dict (dict): Dictionary containing tensors or numpy arrays
    """
    for key, value in data_dict.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: not a tensor/array (type: {type(value)})")