from torch import nn

from src import models
from src.models.model_like import ModelLike


class WGANDModel(ModelLike):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(models.x_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
        )
