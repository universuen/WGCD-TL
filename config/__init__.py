import random

import torch
import numpy as np

from config import (
    data,
    dataset,
    logger,
    path,
    training,
)

# training device
device = 'auto'

# multi-processing
num_data_loader_workers = 0

# random seed
seed = 1

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


def set_random_state():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_state()
