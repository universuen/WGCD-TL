import context

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import src
from scripts.datasets import DATASETS

TEST_NAME = '3-7'

GANS = [
    src.gans.ClassicGAN,
    src.gans.WGAN,
    src.gans.SNGAN,
]
K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]


def highlight_higher_cells(s: pd.Series) -> list[str]:
    result_ = []
    for i_1, i_2 in zip(s[0::2], s[1::2]):
        if i_1 > i_2:
            result_.append('background-color: yellow')
            result_.append('')
        elif i_1 < i_2:
            result_.append('')
            result_.append('background-color: yellow')
        else:
            result_.append('')
            result_.append('')
    return result_


if __name__ == '__main__':
    src.config.logging_config.level = 'WARNING'
    result_file = src.config.path_config.test_results / f'applicability_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    all_models = []
    for i in GANS:
        all_models.append(i.__name__)
        all_models.append(f'{i.__name__}_imp')
    result = {
        k: pd.DataFrame(
            {
                kk:
                    {
                        kkk: 0.0 for kkk in [*DATASETS, 'mean']
                    } for kk in all_models
            }
        ) for k in METRICS
    }

    for dataset_name in tqdm(DATASETS):
        # prepare data
        src.utils.set_random_state()
        samples, labels = src.utils.preprocess_data(dataset_name)
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=src.config.seed)
        temp_result = {
            k: {
                kk: [] for kk in all_models
            } for k in METRICS
        }
        # k-fold test
        for training_indices, test_indices in skf.split(samples, labels):
            src.datasets.training_samples = samples[training_indices]
            src.datasets.training_labels = labels[training_indices]
            src.datasets.test_samples = samples[test_indices]
            src.datasets.test_labels = labels[test_indices]
            training_dataset = src.datasets.FullDataset()
            test_dataset = src.datasets.FullDataset(test=True)
            for GAN in GANS:
                # test GAN
                src.utils.set_random_state()
                gan = GAN()
                gan.fit(training_dataset)
                gan_classifier = src.classifier.Classifier(GAN.__name__)
                balanced_dataset = src.utils.get_balanced_dataset(training_dataset, gan)
                gan_classifier.fit(balanced_dataset)
                gan_classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][GAN.__name__].append(gan_classifier.metrics[metric_name])
                # test GAN-W-TL-BL
                src.utils.set_random_state()
                w_gan = GAN()
                w_gan.fit(src.datasets.WeightedPositiveDataset())
                tl_classifier = src.transfer_learner.TransferLearner()
                tl_classifier.fit(
                    dataset=training_dataset,
                    gan=w_gan,
                )
                tl_classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][f'{GAN.__name__}_imp'].append(tl_classifier.metrics[metric_name])
            # calculate final metrics
            for model_name in all_models:
                for metric_name in METRICS:
                    result[metric_name][model_name][dataset_name] = np.mean(temp_result[metric_name][model_name])
            # calculate average metrics on all datasets
            for model_name in all_models:
                for metric_name in METRICS:
                    result[metric_name][model_name]['mean'] = np.mean(
                        [
                            i for i in result[metric_name][model_name].values
                        ]
                    )
            # write down current result
            occupied = True
            while occupied:
                try:
                    with pd.ExcelWriter(result_file) as writer:
                        for metric_name in METRICS:
                            df = result[metric_name]
                            df.to_excel(writer, metric_name)
                            df.style.apply(highlight_higher_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')
                    occupied = False
                except PermissionError:
                    pass
