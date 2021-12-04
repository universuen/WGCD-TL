import context

import torch

from src import config, utils
from src.sngan import SNGAN

if __name__ == '__main__':
    utils.set_x_size()
    sngan = SNGAN()
    sngan.load_model()
    z = torch.randn(1, config.data.z_size, device=config.device)
    print(sngan.g(z))
