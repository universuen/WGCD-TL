import torch

from . import (
    gans,
    logger, 
    path,
    vae,
)

# random seed
seed = 0

# number of sample features (dynamically set during running)
x_size: int = None

# device used for training
device: str = 'cpu'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
