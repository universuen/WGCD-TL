import torch
from torch import nn

from AEGAN.utils import init_weights


class Decoder(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            hidden_sizes: list[int],
    ):
        super().__init__()

        modules = []
        current_dim = in_size

        for i in hidden_sizes:
            modules.extend(
                [
                    nn.Linear(current_dim, i, bias=False),
                    nn.BatchNorm1d(i),
                    nn.LeakyReLU(),
                ]
            )
            current_dim = i
        modules.append(nn.Linear(current_dim, out_size))

        self.process = nn.Sequential(*modules)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.process(x)
