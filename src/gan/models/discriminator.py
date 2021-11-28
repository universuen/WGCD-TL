import torch
from torch import nn

from src.utils import init_weights
from config.data import x_size


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(x_size, 256),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(256, 128),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(128, 64),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(64, 32),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(32, 16),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(16, 8),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(8, 4),
            ),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(4, 2),
            ),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
