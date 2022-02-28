from torch import nn

from src import models
from src.config import model_config
from src.models.model_like import ModelLike


class GANGModel(ModelLike):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(model_config.z_size, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32, bias=False),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, models.x_size),
        )
