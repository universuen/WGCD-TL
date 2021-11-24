from pathlib import Path
from typing import Callable

import torch
import numpy as np
from torch.utils.data import Dataset as Base

from config import device


class Dataset(Base):
    def __init__(
            self,
            features_path: Path,
            labels_path: Path,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        self.features = torch.from_numpy(
            np.load(features_path)
        ).float().to(device)
        self.labels = torch.from_numpy(
            np.load(labels_path)
        ).float().to(device)
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
