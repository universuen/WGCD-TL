from typing import Callable

import torch
from torch.utils.data import Dataset

from src import datasets


class FullDataset(Dataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        self.samples = datasets.training_samples if training else datasets.test_samples
        self.samples = torch.from_numpy(self.samples).float()
        self.labels = datasets.training_labels if training else datasets.test_labels
        self.labels = torch.from_numpy(self.labels).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = self.labels[item]
        sample = self.transform(sample) if self.transform else sample
        label = self.target_transform(label) if self.target_transform else label
        return sample, label
