import torch
from torch import nn

from src import models
from src.utils import init_weights


class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(models.x_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.process(x)
