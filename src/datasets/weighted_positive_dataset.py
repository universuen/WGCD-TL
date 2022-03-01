import random
from typing import Callable

# import numpy as np
import torch

from src.datasets import PositiveDataset, NegativeDataset


class WeightedPositiveDataset(PositiveDataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        super().__init__(training, transform, target_transform)
        # calculate fits
        pos_samples = PositiveDataset(training, transform, target_transform).samples
        neg_samples = NegativeDataset(training, transform, target_transform).samples
        self.dists = torch.zeros([len(pos_samples) + len(neg_samples)] * 2)
        self.weights = torch.zeros(len(pos_samples))
        # calculate distances
        all_samples = torch.cat([pos_samples, neg_samples])
        for i, sample_a in enumerate(all_samples):
            for j, sample_b in enumerate(all_samples):
                if i > j:
                    self.dists[i][j] = self.dists[j][i]
                elif i < j:
                    self.dists[i][j] = torch.norm(sample_a - sample_b, p=2)
                else:
                    continue
        # calculate weights
        all_labels = torch.cat(
            [
                torch.ones(len(pos_samples)),
                torch.zeros(len(neg_samples))
            ]
        )
        k = int(len(all_samples) / 10)
        instabilities = torch.zeros(len(pos_samples))
        for i, _ in enumerate(pos_samples):
            indices = torch.topk(self.dists[i], k, largest=False).indices
            labels = all_labels[indices]
            instabilities[i] = 1 - torch.cos(2 * torch.pi * sum(labels) / k)
        self.weights = instabilities / (sum(instabilities) + 1e-5)

    def get_n_samples(self, size: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples,
                weights=self.weights,
                k=size,
            )
        )
