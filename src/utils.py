import random

import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from src import config, datasets, models


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(file_name: str) -> (np.ndarray, np.ndarray):
    set_random_state()
    # concatenate the file path
    file_path = config.path.datasets / file_name
    # calculate skip rows
    skip_rows = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '@':
                break
            else:
                skip_rows += 1
    # read raw data
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    # partition labels and samples
    labels = np_array[:, -1].copy()
    samples = np_array[:, :-1].copy()
    # digitize labels
    for i, _ in enumerate(labels):
        labels[i] = labels[i].strip()
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')
    # normalize samples
    samples = minmax_scale(samples)
    models.x_size = samples.shape[1]
    return samples, labels


def prepare_dataset(name: str, training_test_ratio: float = 0.8):
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        random_state=config.seed,
    )
    datasets.training_samples = training_samples
    datasets.training_labels = training_labels
    datasets.test_samples = test_samples
    datasets.test_labels = test_labels
