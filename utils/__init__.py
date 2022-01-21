import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if  torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)
