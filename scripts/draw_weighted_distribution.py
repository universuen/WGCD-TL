import context

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

from src import config, utils, datasets
from src.datasets import FullDataset, WeightedPositiveDataset


if __name__ == '__main__':
    x, y = make_blobs(1000, n_features=2, centers=2, random_state=config.seed)
    datasets.training_samples, datasets.training_labels = x, y

    full_dataset = FullDataset()

    features = full_dataset.samples.numpy()
    types = []

    for i in full_dataset.labels:
        if i.item() == 0:
            types.append('Class A')
        else:
            types.append('Class B')

    r_dataset = WeightedPositiveDataset()
    utils.set_random_state()
    r_features = r_dataset.get_samples()
    features = np.append(features, r_features, axis=0)
    types.extend(['Chosen' for _ in r_features])

    features = TSNE(
        learning_rate='auto',
        init='random',
        random_state=config.seed,
    ).fit_transform(features)

    df = pd.DataFrame(
        {
            'f_x': features[:, 0],
            'f_y': features[:, 1],
            'Type': types,
        }
    )

    sns.set_style('white')
    ax = sns.scatterplot(
        data=df,
        x='f_x',
        y='f_y',
        hue='Type',
        hue_order=['Class A', 'Chosen', 'Class B'],
    )
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.set(yticklabels=[])
    ax.set(ylabel=None)
    plt.savefig(config.path_config.test_results / 'weighted_distribution.jpg')
    plt.show()