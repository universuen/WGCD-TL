import context

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import src
from scripts.datasets import DATASETS

TEST_NAME = 'final'

K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]


def highlight_legal_cells(s: pd.Series) -> list[str]:
    result_ = ['']
    for idx in range(1, len(s)):
        result_.append('background-color: yellow' if s[idx] > s[idx - 1] else '')
    return result_


if __name__ == '__main__':
    src.config.logging_config.level = 'WARNING'
    result_file = src.config.path_config.test_results / f'ablation_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    methods = [
        'Baseline',
        'G',
        'W-G',
        'W-G-CSL',
        'W-G-CSL-TL',
    ]
    # [metric][method][dataset]
    result = {
        k: pd.DataFrame(
            {
                kk: {
                    kkk: 0.0 for kkk in [*DATASETS, 'mean']
                } for kk in methods
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
                kk: [] for kk in methods
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
            sngan = src.gans.SNGAN()
            sngan.fit(training_dataset)
            w_sngan = src.gans.SNGAN()
            w_sngan.fit(src.datasets.WeightedPositiveDataset())
            classifier = None

            for method_name in methods:
                src.utils.set_random_state()
                if method_name == 'Baseline':
                    classifier = src.classifier.Classifier('Baseline')
                    classifier.fit(training_dataset)
                elif method_name == 'G':
                    classifier = src.classifier.Classifier('G')
                    balanced_dataset = src.utils.get_balanced_dataset(training_dataset, sngan)
                    classifier.fit(balanced_dataset)
                elif method_name == 'W-G':
                    classifier = src.classifier.Classifier('W-G')
                    balanced_dataset = src.utils.get_balanced_dataset(training_dataset, w_sngan)
                    classifier.fit(balanced_dataset)
                elif method_name == 'W-G-CSL':
                    classifier = src.classifier.Classifier('W-G-CSL')
                    classifier.fit(
                        dataset=training_dataset,
                        gan=w_sngan,
                    )
                elif method_name == 'W-G-CSL-TL':
                    classifier = src.transfer_learner.TransferLearner()
                    classifier.fit(
                        dataset=training_dataset,
                        gan=w_sngan,
                    )

                classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][method_name].append(classifier.metrics[metric_name])

            # calculate final metrics
            for method_name in methods:
                for metric_name in METRICS:
                    result[metric_name][method_name][dataset_name] = np.mean(temp_result[metric_name][method_name])
            # calculate average metrics on all datasets
            for method_name in methods:
                for metric_name in METRICS:
                    result[metric_name][method_name]['mean'] = np.mean(
                        [i for i in result[metric_name][method_name].values])
            # write down current result
            occupied = True
            while occupied:
                try:
                    with pd.ExcelWriter(result_file) as writer:
                        for metric_name in METRICS:
                            df = result[metric_name]
                            df.to_excel(writer, metric_name)
                            df.style.apply(highlight_legal_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')
                    occupied = False
                except PermissionError:
                    pass
