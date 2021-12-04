import torch

from . import (
    training,
    data,
    dataset,
    path,
)

# random seed
seed: int = 0

# pytorch device
device = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
