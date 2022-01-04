import context

import math

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

import src
from src import utils, Classifier, VAE
from src.dataset import CompleteDataset, MinorityDataset
from src import config

K = 5

TRADITIONAL_METHODS = [
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
]

GAN = src.SNGANHL

DATASETS = [
    # 'dermatology-6.dat',
    'ecoli-0-1-3-7_vs_2-6.dat',
    'ecoli-0-1-4-6_vs_5.dat',
    'ecoli-0-1-4-7_vs_2-3-5-6.dat',
    # 'ecoli-0-1-4-7_vs_5-6.dat',
    # 'ecoli-0-1_vs_2-3-5.dat',
    # 'ecoli-0-1_vs_5.dat',
    'ecoli-0-2-3-4_vs_5.dat',
    # 'ecoli-0-2-6-7_vs_3-5.dat',
    # 'ecoli-0-3-4-6_vs_5.dat',
    # 'ecoli-0-3-4-7_vs_5-6.dat',
    'ecoli-0-3-4_vs_5.dat',
    # 'ecoli-0-4-6_vs_5.dat',
    # 'ecoli-0-6-7_vs_3-5.dat',
    # 'ecoli-0-6-7_vs_5.dat',
    # 'ecoli-0_vs_1.dat',
    'ecoli1.dat',
    'ecoli2.dat',
    'ecoli3.dat',
    # 'ecoli4.dat',
    # 'glass-0-1-2-3_vs_4-5-6.dat',
    'glass-0-1-4-6_vs_2.dat',
    'glass-0-1-5_vs_2.dat',
    'glass-0-1-6_vs_2.dat',
    # 'glass-0-1-6_vs_5.dat',
    # 'glass-0-4_vs_5.dat',
    # 'glass-0-6_vs_5.dat',
    # 'glass0.dat',
    # 'glass1.dat',
    # 'glass2.dat',
    'glass4.dat',
    # 'glass5.dat',
    # 'glass6.dat',
    'haberman.dat',
    # 'iris0.dat',
    # 'led7digit-0-2-4-5-6-7-8-9_vs_1.dat',
    # 'new-thyroid1.dat',
    # 'newthyroid2.dat',
    'page-blocks-1-3_vs_4.dat',
    # 'page-blocks0.dat',
    'pima.dat',
    'poker-8-9_vs_5.dat',
    # 'poker-8-9_vs_6.dat',
    'poker-8_vs_6.dat',
    # 'poker-9_vs_7.dat',
    # 'segment0.dat',
    # 'shuttle-2_vs_5.dat',
    # 'shuttle-6_vs_2-3.dat',
    # 'shuttle-c0-vs-c4.dat',
    # 'shuttle-c2-vs-c4.dat',
    # 'vehicle0.dat',
    # 'vehicle1.dat',
    # 'vehicle2.dat',
    'vehicle3.dat',
    # 'vowel0.dat',
    # 'winequality-red-3_vs_5.dat',
    'winequality-red-4.dat',
    'winequality-red-8_vs_6-7.dat',
    'winequality-red-8_vs_6.dat',
    'winequality-white-3-9_vs_5.dat',
    # 'winequality-white-3_vs_7.dat',
    'winequality-white-9_vs_4.dat',
    'wisconsin.dat',
    'yeast-0-2-5-6_vs_3-7-8-9.dat',
    # 'yeast-0-2-5-7-9_vs_3-6-8.dat',
    'yeast-0-3-5-9_vs_7-8.dat',
    # 'yeast-0-5-6-7-9_vs_4.dat',
    'yeast-1-2-8-9_vs_7.dat',
    'yeast-1-4-5-8_vs_7.dat',
    # 'yeast-1_vs_7.dat',
    'yeast-2_vs_4.dat',
    'yeast-2_vs_8.dat',
    'yeast1.dat',
    'yeast3.dat',
    'yeast4.dat',
    'yeast5.dat',
    'yeast6.dat',
]

WGANHL_DATASETS = [
    # 'dermatology-6.dat',
    'ecoli-0-1-3-7_vs_2-6.dat',
    # 'ecoli-0-1-4-6_vs_5.dat',
    # 'ecoli-0-1-4-7_vs_2-3-5-6.dat',
    'ecoli-0-1-4-7_vs_5-6.dat',
    'ecoli-0-1_vs_2-3-5.dat',
    # 'ecoli-0-1_vs_5.dat',
    'ecoli-0-2-3-4_vs_5.dat',
    # 'ecoli-0-2-6-7_vs_3-5.dat',
    # 'ecoli-0-3-4-6_vs_5.dat',
    'ecoli-0-3-4-7_vs_5-6.dat',
    'ecoli-0-3-4_vs_5.dat',
    # 'ecoli-0-4-6_vs_5.dat',
    # 'ecoli-0-6-7_vs_3-5.dat',
    # 'ecoli-0-6-7_vs_5.dat',
    # 'ecoli-0_vs_1.dat',
    # 'ecoli1.dat',
    # 'ecoli2.dat',
    # 'ecoli3.dat',
    # 'ecoli4.dat',
    # 'glass-0-1-2-3_vs_4-5-6.dat',
    # 'glass-0-1-4-6_vs_2.dat',
    # 'glass-0-1-5_vs_2.dat',
    'glass-0-1-6_vs_2.dat',
    'glass-0-1-6_vs_5.dat',
    # 'glass-0-4_vs_5.dat',
    # 'glass-0-6_vs_5.dat',
    # 'glass0.dat',
    # 'glass1.dat',
    # 'glass2.dat',
    'glass4.dat',
    # 'glass5.dat',
    # 'glass6.dat',
    # 'haberman.dat',
    # 'iris0.dat',
    # 'led7digit-0-2-4-5-6-7-8-9_vs_1.dat',
    # 'new-thyroid1.dat',
    # 'newthyroid2.dat',
    'page-blocks-1-3_vs_4.dat',
    # 'page-blocks0.dat',
    # 'pima.dat',
    'poker-8-9_vs_5.dat',
    # 'poker-8-9_vs_6.dat',
    'poker-8_vs_6.dat',
    # 'poker-9_vs_7.dat',
    # 'segment0.dat',
    'shuttle-2_vs_5.dat',
    'shuttle-6_vs_2-3.dat',
    # 'shuttle-c0-vs-c4.dat',
    # 'shuttle-c2-vs-c4.dat',
    # 'vehicle0.dat',
    # 'vehicle1.dat',
    'vehicle2.dat',
    'vehicle3.dat',
    # 'vowel0.dat',
    # 'winequality-red-3_vs_5.dat',
    # 'winequality-red-4.dat',
    # 'winequality-red-8_vs_6-7.dat',
    'winequality-red-8_vs_6.dat',
    'winequality-white-3-9_vs_5.dat',
    # 'winequality-white-3_vs_7.dat',
    'winequality-white-9_vs_4.dat',
    'wisconsin.dat',
    'yeast-0-2-5-6_vs_3-7-8-9.dat',
    'yeast-0-2-5-7-9_vs_3-6-8.dat',
    # 'yeast-0-3-5-9_vs_7-8.dat',
    # 'yeast-0-5-6-7-9_vs_4.dat',
    'yeast-1-2-8-9_vs_7.dat',
    # 'yeast-1-4-5-8_vs_7.dat',
    # 'yeast-1_vs_7.dat',
    # 'yeast-2_vs_4.dat',
    'yeast-2_vs_8.dat',
    # 'yeast1.dat',
    'yeast3.dat',
    'yeast4.dat',
    # 'yeast5.dat',
    # 'yeast6.dat',
]

METRICS = [
    'Precision',
    'Recall',
    'F1',
    'G-Mean',
    'AUC',
]


def train_all() -> dict:
    metric_ = dict()
    utils.set_x_size()

    # original classifier
    utils.set_random_state()
    classifier = Classifier('Original')
    classifier.train(
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    metric_['Original'] = utils.get_final_test_metrics(classifier.statistics)

    # traditional methods
    for M in TRADITIONAL_METHODS:
        try:
            x, y = CompleteDataset(training=True)[:]
            x = x.numpy()
            y = y.numpy()
            x, y = M(random_state=config.seed).fit_resample(x, y)
            dataset = CompleteDataset()
            dataset.features = torch.from_numpy(x)
            dataset.labels = torch.from_numpy(y)
            utils.set_random_state()
            classifier = Classifier(M.__name__)
            classifier.train(
                training_dataset=dataset,
                test_dateset=CompleteDataset(training=False),
            )
            metric_[M.__name__] = utils.get_final_test_metrics(classifier.statistics)
        except (RuntimeError, ValueError):
            metric_[M.__name__] = {k: math.nan for k in METRICS}

    # prepare encoder
    utils.set_random_state()
    vae = VAE()
    vae.train(MinorityDataset(training=True))

    # GANHL_EGD
    utils.set_random_state()
    gan = GAN()
    gan.train(MinorityDataset(training=True))

    utils.set_random_state()
    classifier = Classifier(f'{GAN.__name__}_EGD')
    classifier.egd_train(
        encoder=vae.encoder,
        generator=gan.generator,
        discriminator=gan.discriminator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
        seed_dataset=MinorityDataset(training=True),
    )
    metric_[f'{GAN.__name__}_EGD'] = utils.get_final_test_metrics(classifier.statistics)

    return metric_


def validate(file_name_: str) -> pd.DataFrame:
    all_methods = ['Original']
    all_methods.extend([M.__name__ for M in TRADITIONAL_METHODS])
    all_methods.append(f'{GAN.__name__}_EGD')

    result_ = {
        k: {
            kk: [] for kk in METRICS
        } for k in all_methods
    }

    # preprocess data
    file_path = config.path.data / file_name_
    skip_rows = 0
    with open(file_path, 'r') as f_:
        while True:
            line = f_.readline()
            if line[0] != '@':
                break
            else:
                skip_rows += 1
    df_ = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df_.to_numpy()
    label = np_array[:, -1].copy()
    feature = np_array[:, :-1].copy()
    for i, _ in enumerate(label):
        label[i] = label[i].strip()
    label[label[:] == 'positive'] = 1
    label[label[:] == 'negative'] = 0
    label = label.astype('int')
    feature = MinMaxScaler().fit_transform(feature)

    # partition data and train all models
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=config.seed)
    for training_indices, test_indices in skf.split(feature, label):
        np.save(str(config.path.data / 'training_label.npy'), label[training_indices])
        np.save(str(config.path.data / 'training_feature.npy'), feature[training_indices])
        np.save(str(config.path.data / 'test_label.npy'), label[test_indices])
        np.save(str(config.path.data / 'test_feature.npy'), feature[test_indices])
        metric = train_all()
        for model_name, v in metric.items():
            for metric_name, vv in v.items():
                if metric_name in METRICS:
                    result_[model_name][metric_name].append(vv)
                else:
                    continue

    for model_name, v in result_.items():
        for metric_name, vv in v.items():
            result_[model_name][metric_name] = sum(vv) / len(vv)

    return pd.DataFrame(result_)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    with open(config.path.data / 'kfcv_n_vs_1_tested_datasets.txt', 'w') as f:
        f.write('')

    result = dict()
    for file_name in DATASETS:
        print(f'{file_name:*^50}')
        result[file_name] = validate(file_name)
        with open(config.path.data / 'kfcv_n_vs_1_tested_datasets.txt', 'a') as f:
            f.write(f'{file_name}\n')
            f.write(f'{result[file_name]}\n\n')

    with pd.ExcelWriter(config.path.data / 'kfcv_n_vs_1_result.xlsx') as writer:
        for filename, df in result.items():
            df.to_excel(writer, filename.split('.')[0])
            for column in df:
                column_width = 15
                col_idx = df.columns.get_loc(column) + 1
                writer.sheets[filename.split('.')[0]].set_column(col_idx, col_idx, column_width)
            df.style.highlight_max(axis=1).to_excel(writer, filename.split('.')[0])

    print('Done!')
