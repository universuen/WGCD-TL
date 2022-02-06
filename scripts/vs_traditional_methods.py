import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

import src

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

RGAN = src.gans.RSNGAN

K = 5

DATASETS = [
    'dermatology-6.dat',
    'ecoli-0-1-3-7_vs_2-6.dat',
    'ecoli-0-1-4-6_vs_5.dat',
    'ecoli-0-1-4-7_vs_2-3-5-6.dat',
    'ecoli-0-1-4-7_vs_5-6.dat',
    'ecoli-0-1_vs_2-3-5.dat',
    'ecoli-0-1_vs_5.dat',
    'ecoli-0-2-3-4_vs_5.dat',
    'ecoli-0-2-6-7_vs_3-5.dat',
    'ecoli-0-3-4-6_vs_5.dat',
    'ecoli-0-3-4-7_vs_5-6.dat',
    'ecoli-0-3-4_vs_5.dat',
    'ecoli-0-4-6_vs_5.dat',
    'ecoli-0-6-7_vs_3-5.dat',
    'ecoli-0-6-7_vs_5.dat',
    'ecoli-0_vs_1.dat',
    'ecoli1.dat',
    'ecoli2.dat',
    'ecoli3.dat',
    'ecoli4.dat',
    'glass-0-1-2-3_vs_4-5-6.dat',
    'glass-0-1-4-6_vs_2.dat',
    'glass-0-1-5_vs_2.dat',
    'glass-0-1-6_vs_2.dat',
    'glass-0-1-6_vs_5.dat',
    'glass-0-4_vs_5.dat',
    'glass-0-6_vs_5.dat',
    'glass0.dat',
    'glass1.dat',
    'glass2.dat',
    'glass4.dat',
    'glass5.dat',
    'glass6.dat',
    'haberman.dat',
    'iris0.dat',
    'led7digit-0-2-4-5-6-7-8-9_vs_1.dat',
    'new-thyroid1.dat',
    'newthyroid2.dat',
    'page-blocks-1-3_vs_4.dat',
    'page-blocks0.dat',
    'pima.dat',
    'poker-8-9_vs_5.dat',
    'poker-8-9_vs_6.dat',
    'poker-8_vs_6.dat',
    'poker-9_vs_7.dat',
    'segment0.dat',
    'shuttle-2_vs_5.dat',
    'shuttle-6_vs_2-3.dat',
    'shuttle-c0-vs-c4.dat',
    'shuttle-c2-vs-c4.dat',
    'vehicle0.dat',
    'vehicle1.dat',
    'vehicle2.dat',
    'vehicle3.dat',
    'vowel0.dat',
    'winequality-red-3_vs_5.dat',
    'winequality-red-4.dat',
    'winequality-red-8_vs_6-7.dat',
    'winequality-red-8_vs_6.dat',
    'winequality-white-3-9_vs_5.dat',
    'winequality-white-3_vs_7.dat',
    'winequality-white-9_vs_4.dat',
    'wisconsin.dat',
    'yeast-0-2-5-6_vs_3-7-8-9.dat',
    'yeast-0-2-5-7-9_vs_3-6-8.dat',
    'yeast-0-3-5-9_vs_7-8.dat',
    'yeast-0-5-6-7-9_vs_4.dat',
    'yeast-1-2-8-9_vs_7.dat',
    'yeast-1-4-5-8_vs_7.dat',
    'yeast-1_vs_7.dat',
    'yeast-2_vs_4.dat',
    'yeast-2_vs_8.dat',
    'yeast1.dat',
    'yeast3.dat',
    'yeast4.dat',
    'yeast5.dat',
    'yeast6.dat',
]

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    all_methods = ['Original', *[i.__name__ for i in TRADITIONAL_METHODS], 'RGAN-TL']
    result = {
        k: pd.DataFrame(
            {
                kk:
                    {
                        kkk: 0.0 for kkk in DATASETS
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
            # test original classifier
            src.utils.set_random_state()
            training_dataset = src.datasets.FullDataset(training=True)
            test_dataset = src.datasets.FullDataset(training=False)
            classifier = src.classifier.Classifier('Original')
            classifier.fit(training_dataset)
            classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['Original'].append(classifier.metrics[metric_name])
            # test traditional methods
            for METHOD in TRADITIONAL_METHODS:
                try:
                    x, y = training_dataset.samples, training_dataset.labels
                    x = x.numpy()
                    y = y.numpy()
                    x, y = METHOD(random_state=src.config.seed).fit_resample(x, y)
                    balanced_dataset = src.datasets.BasicDataset()
                    balanced_dataset.samples = torch.from_numpy(x)
                    balanced_dataset.labels = torch.from_numpy(y)
                    src.utils.set_random_state()
                    classifier = src.classifier.Classifier(METHOD.__name__)
                    classifier.fit(balanced_dataset)
                    classifier.test(test_dataset)
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(classifier.metrics[metric_name])
                except (RuntimeError, ValueError):
                    for metric_name in METRICS:
                        temp_result[metric_name][METHOD.__name__].append(0)
            # test RGAN-TL
            src.utils.set_random_state()
            rgan_dataset = src.utils.get_rgan_dataset(RGAN())
            esb_classifier = src.tr_ada_boost.TrAdaBoost()
            esb_classifier.fit(rgan_dataset, training_dataset)
            esb_classifier.test(test_dataset)
            for metric_name in METRICS:
                temp_result[metric_name]['RGAN-TL'].append(classifier.metrics[metric_name])
        # calculate final metrics
        for method_name in all_methods:
            for metric_name in METRICS:
                result[metric_name][method_name][dataset_name] = np.mean(temp_result[metric_name][method_name])
        # write down current result
        for metric_name in METRICS:
            result[metric_name].to_excel(
                src.config.path.test_results / f'vs_tm_{metric_name}.xlsx',
                float_format='%.4f'
            )

