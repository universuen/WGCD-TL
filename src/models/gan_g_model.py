import torch
from torch import nn

from src import config
from src.utils import init_weights


class GANGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, config.x_size),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
