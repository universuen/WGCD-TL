import context

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler
from tqdm import tqdm

import src
from scripts.datasets import ALL_DATASETS

# DATASET = 'wisconsin.dat'

TRADITIONAL_METHODS = [
    SMOTE,
    ADASYN,
    SVMSMOTE,
]

GAN_MODELS = [
    src.gans.ClassicGAN,
    src.gans.WGAN,
    src.gans.SNGAN,
]


def tsne(dataset_name: str) -> None:
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(dataset_name)
    dataset = src.datasets.FullDataset()

    raw_x, raw_y = dataset.samples, dataset.labels
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    for M in TRADITIONAL_METHODS:
        x, _ = M(random_state=src.config.seed).fit_resample(raw_x, raw_y)
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[M.__name__] = [embedded_x, y]

    for GAN in GAN_MODELS:
        src.utils.set_random_state()
        gan = GAN()
        gan.fit(src.datasets.PositiveDataset())
        z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.model_config.z_size], device=src.config.device)
        x = np.concatenate([raw_x, gan.g(z).detach().cpu().numpy()])
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[GAN.__name__] = [embedded_x, y]

    for GAN in GAN_MODELS:
        src.utils.set_random_state()
        gan = GAN()
        gan.fit(src.datasets.WeightedPositiveDataset())
        z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.model_config.z_size], device=src.config.device)
        x = np.concatenate([raw_x, gan.g(z).detach().cpu().numpy()])
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)
        result[f'{GAN.__name__}_W'] = [embedded_x, y]

    sns.set_style('white')
    fig, axes = plt.subplots(3, 3)
    for (key, value), axe in zip(result.items(), axes.flat):
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
            alpha=0.8,
            label='majority',
        )
        sns.scatterplot(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            ax=axe,
            alpha=0.8,
            label='generated_data',
        )
        sns.scatterplot(
            x=minority[:, 0],
            y=minority[:, 1],
            ax=axe,
            alpha=0.8,
            label='minority',
        )
        axe.get_legend().remove()
    fig.set_size_inches(18, 10)
    fig.set_dpi(100)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(src.config.path_config.tsne_plots / f'{dataset_name}.jpg')
    plt.close()


if __name__ == '__main__':
    src.config.logging_config.level = 'CRITICAL'
    successful_datasets = []
    for dataset_name in tqdm(ALL_DATASETS):
        try:
            tsne(dataset_name)
            successful_datasets.append(dataset_name)
        except (RuntimeError, ValueError):
            pass
    with open(src.config.path_config.test_results / 'tsne_datasets.txt', 'w') as f:
        for i in successful_datasets:
            f.write(f'{i}\n')
