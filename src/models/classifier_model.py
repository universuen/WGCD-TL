import torch
from torch import nn

from src import models
from src.models._model import Model


class ClassifierModel(Model):
    def __init__(self):
        super().__init__()
        self.main_model = nn.Sequential(
            nn.Linear(models.x_size, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
        )
        self.last_layer = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.apply(self._init_weights)
            self.initialized = True
        x = self.main_model(x)
        x = self.last_layer(x)
        return torch.sigmoid(x)
