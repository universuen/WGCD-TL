import random

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(file_name: str) -> (np.ndarray, np.ndarray):
    set_random_state()
    # concatenate the file path
    file_path = src.config.path_config.datasets / file_name
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
    samples = minmax_scale(samples.astype('float32'))
    src.models.x_size = samples.shape[1]
    return samples, labels


def prepare_dataset(name: str, training_test_ratio: float = 0.8) -> None:
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        random_state=src.config.seed,
    )
    src.datasets.training_samples = training_samples
    src.datasets.training_labels = training_labels
    src.datasets.test_samples = test_samples
    src.datasets.test_labels = test_labels


def get_final_test_metrics(statistics: dict) -> dict:
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


# def normalize(x: torch.Tensor) -> torch.Tensor:
#     return (x - x.min()) / (x.max() - x.min())


# def get_knn_indices(sample: torch.Tensor, all_samples: torch.Tensor, k: int = 5) -> torch.Tensor:
#     dist = torch.empty(len(all_samples))
#     for i, v in enumerate(all_samples):
#         dist[i] = torch.norm(sample - v, p=2)
#     return torch.topk(dist, k, largest=False).indices


def get_gan_dataset(gan: src.types.GANLike) -> src.types.DatasetLike:
    gan.fit()
    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    target_sample_num = total_neg_cnt - total_pos_cnt
    if target_sample_num <= 0:
        return full_dataset
    z = torch.rand(target_sample_num, src.config.model_config.z_size, device=src.config.device)
    new_samples = gan.generate_samples(z)
    new_labels = torch.ones(target_sample_num, device=src.config.device)
    target_dataset.samples = torch.cat(
        [
            target_dataset.samples,
            new_samples,
        ],
    )
    target_dataset.labels = torch.cat(
        [
            target_dataset.labels,
            new_labels,
        ]
    )
    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    return target_dataset


