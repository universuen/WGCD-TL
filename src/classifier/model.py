import torch
from torch import nn

from src.utils import init_weights
from config.data import x_size


class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(x_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Linear(8, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
