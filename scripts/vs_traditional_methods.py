import context

import os

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

import src
from scripts.datasets import ALL_DATASETS as DATASETS

TEST_NAME = 'auc_fixed'

TRADITIONAL_METHODS = [
    SMOTE,
    ADASYN,
    SVMSMOTE,
]

GAN = src.gans.SNGAN

K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]

if __name__ == '__main__':
    src.config.logging_config.level = 'WARNING'
    result_file = src.config.path_config.test_results / f'vstm_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    all_methods = ['Baseline', *[i.__name__ for i in TRADITIONAL_METHODS], 'WGCSLTL']
    result = {
        k: pd.DataFrame(
            {
                kk:
                    {
                        kkk: 0.0 for kkk in [*DATASETS, 'mean']
                    } for kk in all_methods
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
                kk: [] for kk in all_methods
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
            # test baseline classifier
            src.utils.set_random_state()
            o_classifier = src.classifier.Classifier('Baseline')
            o_classifier.fit(training_dataset)
            o_classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['Baseline'].append(o_classifier.metrics[metric_name])
            # test traditional methods
            for METHOD in TRADITIONAL_METHODS:
                try:
                    x, y = training_dataset.samples, training_dataset.labels
                    x = x.cpu().numpy()
                    y = y.cpu().numpy()
                    x, y = METHOD(random_state=src.config.seed).fit_resample(x, y)
                    balanced_dataset = src.datasets.Dataset()
                    balanced_dataset.samples = torch.from_numpy(x)
                    balanced_dataset.labels = torch.from_numpy(y)
                    src.utils.set_random_state()
                    tm_classifier = src.classifier.Classifier(METHOD.__name__)
                    tm_classifier.fit(balanced_dataset)
                    tm_classifier.test(test_dataset)
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(tm_classifier.metrics[metric_name])
                except (RuntimeError, ValueError):
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(0)
            # test WGCSLTL
            src.utils.set_random_state()
            gan = src.gans.SNGAN()
            gan.fit(src.datasets.WeightedPositiveDataset())
            tl_classifier = src.transfer_learner.TransferLearner()
            tl_classifier.fit(
                dataset=training_dataset,
                gan=gan,
            )
            tl_classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['WGCSLTL'].append(tl_classifier.metrics[metric_name])
        # calculate final metrics
        for method_name in all_methods:
            for metric_name in METRICS:
                result[metric_name][method_name][dataset_name] = np.mean(temp_result[metric_name][method_name])
        # calculate average metrics on all datasets
        for gan_name in all_methods:
            for metric_name in METRICS:
                result[metric_name][gan_name]['mean'] = np.mean([i for i in result[metric_name][gan_name].values])
        # write down current result
        occupied = True
        while occupied:
            try:
                with pd.ExcelWriter(result_file) as writer:
                    for metric_name in METRICS:
                        df = result[metric_name]
                        df.to_excel(writer, metric_name)
                        df.style.highlight_max(axis=1).to_excel(writer, metric_name, float_format='%.4f')
                occupied = False
            except PermissionError:
                pass
