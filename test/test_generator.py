import context

import torch
from torch.nn.functional import softmax, log_softmax

from src.vae.models import EncoderModel
from src.egan.models import GeneratorModel
from src.dataset import MinorityDataset
from config import path

def cal_kl_div(x, x_hat):
    x_hat = softmax(x_hat, dim=1)
    x = log_softmax(x, dim=1)
    return -(x_hat * x).sum() / x.shape[0]


if __name__ == '__main__':
    e = EncoderModel()
    e.load_state_dict(torch.load(path.data / 'encoder.pt'))
    e.eval()

    g = GeneratorModel()
    g.load_state_dict(torch.load(path.data / 'EGAN_generator.pt'))
    g.eval()

    x = MinorityDataset()[:100][0]
    z, _, _ = e(x)
    x_hat = g(z)
    print(x.mean().mean().item())
    print(x_hat.mean().mean().item())
    print(torch.norm(x - x_hat, 2, dim=1).mean().item())
    print(cal_kl_div(x, x_hat).item())
