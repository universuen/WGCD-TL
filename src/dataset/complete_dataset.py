from typing import Callable

import torch
import numpy as np
from torch.utils.data import Dataset as Base

from config import path


class CompleteDataset(Base):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        if training is True:
            labels_path = path.data / 'training_labels.npy'
            features_path = path.data / 'training_features.npy'
        else:
            labels_path = path.data / 'test_labels.npy'
            features_path = path.data / 'test_features.npy'

        self.features = torch.from_numpy(
            np.load(features_path)
        ).float()
        self.labels = torch.from_numpy(
            np.load(labels_path)
        ).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label
