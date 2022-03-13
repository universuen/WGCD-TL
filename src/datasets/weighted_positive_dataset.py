import random

import torch

from src.datasets import PositiveDataset, NegativeDataset


class WeightedPositiveDataset(PositiveDataset):
    def __init__(
            self,
            test: bool = False,
    ):
        super().__init__(test)
        self.is_weighted = True
        # calculate fits
        pos_samples = PositiveDataset(test).samples
        neg_samples = NegativeDataset(test).samples
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
        self.entropy = torch.zeros(len(pos_samples))
        for i, _ in enumerate(pos_samples):
            indices = torch.topk(self.dists[i], k, largest=False).indices
            labels = all_labels[indices]
            p = sum(labels) / k - 1e-5  # make sure 0 < p < 1
            self.entropy[i] = -(p * torch.log(p) + (1-p) * torch.log(1-p))
        self.weights = self.entropy / (sum(self.entropy) + 1e-5)

    def _get_weighted_samples(self, size: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples,
                weights=self.weights,
                k=size,
            )
        )
