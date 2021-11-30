import context

import torch
import numpy as np

from src.vae.models import EncoderModel
from config import path


if __name__ == '__main__':
    e = EncoderModel()
    e.load_state_dict(torch.load(path.data / 'encoder.pt'))
    e.eval()

    x = torch.Tensor(
        np.array([np.load(path.data / 'test_features.npy')[0], np.load(path.data / 'test_features.npy')[0], np.load(path.data / 'test_features.npy')[0]])
    )
    z = e(x)[0]
    print(z.shape)
    print(z[0].sum(), z[1].sum(), z[2].sum())

    # _, mu, sigma = e(x)
    # print(mu.mean().item(), sigma.mean().item())
