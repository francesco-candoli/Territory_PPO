import random
import numpy as np
import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)