import torch
from torch import nn

import config
from src.utils import init_weights


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(config.data.x_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

        self.calculate_mu = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, config.data.z_size),
        )
        self.calculate_log_variance = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, config.data.z_size),
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