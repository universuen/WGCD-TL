import torch
from torch import nn

from src.utils import init_weights
from config.data import x_size


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(x_size, 64),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(64, 256),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(256, 1024),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(1024, 256),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(256, 64),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(64, 16),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(16, 4),
            ),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
