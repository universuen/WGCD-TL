import context

import random

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import src

if __name__ == '__main__':
    x, y = make_blobs(600, n_features=3, centers=2, random_state=src.config.seed)
    src.datasets.training_samples, src.datasets.training_labels = x, y
    r_dataset = src.datasets.WeightedPositiveDataset()
    r_dataset.samples = torch.tensor(list(range(len(r_dataset))))
    sample_cnt = dict()
    for i in range(len(r_dataset)):
        sample_cnt[r_dataset.samples[i].item()] = 0
    assert len(sample_cnt) == len(r_dataset)
    for i in r_dataset.get_samples(10000):
        sample_cnt[i.item()] += 1

    x = list(range(len(r_dataset)))
    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, r_dataset.entropy, 'tab:blue')
    ax2.plot(x, sample_cnt.values(), 'tab:orange')
    ax1.set_xlabel('Sample ID')
    ax2.set_xlabel('Sample ID')
    ax1.set_ylabel('Entropy', color='tab:blue')
    ax2.set_ylabel('Selected times', color='tab:orange')
    plt.savefig(src.config.path_config.test_results / 'weighted_statistics.jpg')
    plt.show()

