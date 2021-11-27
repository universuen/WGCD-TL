import context

import torch
import numpy as np

from src.VAE.models import Encoder
from config import path


if __name__ == '__main__':
    e = Encoder()
    e.load_state_dict(torch.load(path.data / 'encoder.pt'))
    e.eval()

    x = torch.Tensor(
        np.array([np.load(path.data / 'test_features.npy')[0]])
    )
    _, mu, sigma = e(x)
    print(mu.mean().item(), sigma.mean().item())
