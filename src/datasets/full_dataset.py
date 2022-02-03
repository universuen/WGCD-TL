from typing import Callable

import torch
import numpy as np
from torch.utils.data import Dataset as Base

from src import config


class FullDataset(Base):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        if training is True:
            samples_path = config.path.processed_datasets / 'training_samples.npy'
            labels_path = config.path.processed_datasets / 'training_labels.npy'
        else:
            samples_path = config.path.processed_datasets / 'test_samples.npy'
            labels_path = config.path.processed_datasets / 'test_labels.npy'

        self.samples = torch.from_numpy(
            np.load(str(samples_path))
        ).float()
        self.labels = torch.from_numpy(
            np.load(str(labels_path))
        ).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = self.labels[item]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label
