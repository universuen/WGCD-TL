import context

import random

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

import src

DATASET = 'pima.dat'

if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(DATASET)
    dataset = src.dataset.CompleteDataset(training=True)

    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    # SMOTE
    x, _ = SMOTE(random_state=src.config.seed).fit_resample(raw_x, raw_y)
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['SMOTE'] = [embedded_x, y]

    # ADASYN
    x, _ = ADASYN(random_state=src.config.seed).fit_resample(raw_x, raw_y)
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['ADASYN'] = [embedded_x, y]

    # BorderlineSMOTE
    x, _ = BorderlineSMOTE(random_state=src.config.seed).fit_resample(raw_x, raw_y)
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['BorderlineSMOTE'] = [embedded_x, y]

    # WGAN
    src.utils.set_random_state()
    gan = src.WGAN()
    gan.train(src.dataset.MinorityDataset())
    z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.data.z_size], device=src.config.device)
    x = np.concatenate([raw_x, gan.generator(z).detach().cpu().numpy()])
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['WGAN'] = [embedded_x, y]

    # WGANGP
    src.utils.set_random_state()
    gan = src.WGANGP()
    gan.train(src.dataset.MinorityDataset())
    z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.data.z_size], device=src.config.device)
    x = np.concatenate([raw_x, gan.generator(z).detach().cpu().numpy()])
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['WGANGP'] = [embedded_x, y]

    # SNGAN
    src.utils.set_random_state()
    gan = src.SNGAN()
    gan.train(src.dataset.MinorityDataset())
    z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.data.z_size], device=src.config.device)
    x = np.concatenate([raw_x, gan.generator(z).detach().cpu().numpy()])
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['SNGAN'] = [embedded_x, y]

    # HLGAN
    src.utils.set_random_state()
    gan = src.HLGAN()
    gan.train(src.dataset.MinorityDataset())
    z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.config.data.z_size], device=src.config.device)
    x = np.concatenate([raw_x, gan.generator(z).detach().cpu().numpy()])
    y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    result['HLGAN'] = [embedded_x, y]

    sns.set_theme()
    _, axes = plt.subplots(3, 3)
    for (key, value), axe in zip(result.items(), axes.flat):
        print(key)
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
            alpha=0.5,
            label='majority',
        )
        sns.scatterplot(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            ax=axe,
            alpha=0.5,
            label='generated_data',
        )
        sns.scatterplot(
            x=minority[:, 0],
            y=minority[:, 1],
            ax=axe,
            alpha=1.0,
            s=10,
            label='minority',
        )
        axe.get_legend().remove()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.show()
