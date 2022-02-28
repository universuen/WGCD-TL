import torch
from torch import nn


def init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif type(layer) == nn.BatchNorm1d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class ModelLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: nn.Module = None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.model(x)
