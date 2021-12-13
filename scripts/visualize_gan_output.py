import context

import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import src

TARGET_GAN = src.GAN

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    datasets = random.choices(src.utils.get_all_datasets(), k=6)

    for dataset_name in datasets:
        src.utils.prepare_dataset(dataset_name)
        dataset = src.dataset.CompleteDataset(training=True)
        raw_x, raw_y = dataset[:]
        raw_x = raw_x.numpy()
        raw_y = raw_y.numpy()

        src.utils.set_random_state()
        gan = TARGET_GAN()
        gan.train(dataset=src.dataset.MinorityDataset(training=True))
        gan.load_model()
        z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.data.z_size], device=src.config.device)
        x = np.concatenate(
            [raw_x, gan.generator(z).detach().cpu().numpy()],
        )
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[dataset_name] = [embedded_x, y]

    sns.set_theme()
    _, axes = plt.subplots(2, 3)
    for (key, value), axe in zip(result.items(), axes.flat):
        axe.set(xticklabels=[])
        axe.set(yticklabels=[])
        axe.set(title=key)
        majority = []
        minority = []
        generated_data = []
        for i, j in zip(value[0], value[1]):
            if j == 0:
                majority.append(i)
            elif j == 1:
                minority.append(i)
            else:
                generated_data.append(i)
        minority = np.array(minority)
        majority = np.array(majority)
        generated_data = np.array(generated_data)
        sns.scatterplot(
            x=majority[:, 0],
            y=majority[:, 1],
            ax=axe,
            alpha=0.3,
            label='majority',
        )
        sns.scatterplot(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            ax=axe,
            alpha=0.3,
            label='generated_data',
        )
        sns.scatterplot(
            x=minority[:, 0],
            y=minority[:, 1],
            ax=axe,
            alpha=1.0,
            label='minority',
        )
        axe.get_legend().remove()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()