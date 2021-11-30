import torch
from torch import nn

import config
from src.utils import init_weights


class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(config.data.x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
