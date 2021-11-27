import context

import torch

from src.VAE.models import Encoder
from src.GAN.models import Generator
from src.dataset import MinorityTrainingDataset
from config import path


if __name__ == '__main__':
    e = Encoder()
    e.load_state_dict(torch.load(path.data / 'encoder.pt'))
    e.eval()

    g = Generator()
    g.load_state_dict(torch.load(path.data / 'generator.pt'))
    g.eval()

    x = MinorityTrainingDataset()[:10][0]
    z, _, _ = e(x)
    x_hat = g(z)
    print(x.mean().mean().item())
    print(x_hat.mean().mean().item())
    print(torch.norm(x - x_hat, 2, dim=1).mean().item())
