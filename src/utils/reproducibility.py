import os
import random
import numpy as np

def set_all_seeds(seed):
    """
    Sets seeds for reproducibility across different libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("Warning: PyTorch not found. Skipping PyTorch seed setting.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)