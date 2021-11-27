import torch
from torch import nn

from src.utils import init_weights
from config.data import x_size, z_size


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(x_size, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

        )

        self.calculate_mu = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 8, bias=False),
            nn.LeakyReLU(),
            nn.Linear(8, z_size),
        )
        self.calculate_log_variance = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 8, bias=False),
            nn.LeakyReLU(),
            nn.Linear(8, z_size),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        temp = self.preprocess(x)
        mu = self.calculate_mu(temp)
        log_variance = self.calculate_log_variance(temp)
        sigma = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(mu)
        z = epsilon * sigma + mu
        return z, mu, sigma
