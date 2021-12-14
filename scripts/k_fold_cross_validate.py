import context

import math

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

import src
from src import utils, Classifier, VAE
from src.dataset import CompleteDataset, MinorityDataset
from src import config

K = 5
MODELS = (
    'Original',
    'SMOTE',
    'ADASYN',
    'BorderlineSMOTE',
    'WGAN_G',
    'WGAN_EGD',
    'WGANGP_G',
    'WGANGP_EGD',
    'SNGAN_G',
    'SNGAN_EGD',
)
METRICS = (
    'Precision',
    'Recall',
    'F1',
    'Accuracy',
    'AUC',
)


def highlight_higher_cell(s: pd.Series) -> list[str]:
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

    # SMOTE
    try:
        x, y = CompleteDataset(training=True)[:]
        x = x.numpy()
        y = y.numpy()
        x, y = SMOTE(random_state=config.seed).fit_resample(x, y)
        smote_dataset = CompleteDataset()
        smote_dataset.features = torch.from_numpy(x)
        smote_dataset.labels = torch.from_numpy(y)
        utils.set_random_state()
        classifier = Classifier('SMOTE')
        classifier.train(
            training_dataset=smote_dataset,
            test_dateset=CompleteDataset(training=False),
        )
        metric_['SMOTE'] = utils.get_final_test_metrics(classifier.statistics)
    except ValueError:
        metric_['SMOTE'] = {k: math.nan for k in METRICS}

    # ADASYN
    try:
        x, y = CompleteDataset(training=True)[:]
        x = x.numpy()
        y = y.numpy()
        x, y = ADASYN(random_state=config.seed).fit_resample(x, y)
        adasyn_dataset = CompleteDataset()
        adasyn_dataset.features = torch.from_numpy(x)
        adasyn_dataset.labels = torch.from_numpy(y)
        utils.set_random_state()
        classifier = Classifier('ADASYN')
        classifier.train(
            training_dataset=adasyn_dataset,
            test_dateset=CompleteDataset(training=False),
        )
        metric_['ADASYN'] = utils.get_final_test_metrics(classifier.statistics)
    except (RuntimeError, ValueError):
        metric_['ADASYN'] = {k: math.nan for k in METRICS}

    # BorderlineSMOTE
    try:
        x, y = CompleteDataset(training=True)[:]
        x = x.numpy()
        y = y.numpy()
        x, y = BorderlineSMOTE(random_state=config.seed).fit_resample(x, y)
        smote_enn_dataset = CompleteDataset()
        smote_enn_dataset.features = torch.from_numpy(x)
        smote_enn_dataset.labels = torch.from_numpy(y)
        utils.set_random_state()
        classifier = Classifier('BorderlineSMOTE')
        classifier.train(
            training_dataset=smote_enn_dataset,
            test_dateset=CompleteDataset(training=False),
        )
        metric_['BorderlineSMOTE'] = utils.get_final_test_metrics(classifier.statistics)
    except ValueError:
        metric_['BorderlineSMOTE'] = {k: math.nan for k in METRICS}

    # prepare encoder
    utils.set_random_state()
    vae = VAE()
    vae.train(MinorityDataset(training=True))
    vae.load_model()

    # WGAN_G
    utils.set_random_state()
    wgan = src.WGAN()
    wgan.train(MinorityDataset(training=True))
    wgan.load_model()
    utils.set_random_state()
    classifier = Classifier('WGAN_G')
    classifier.g_train(
        generator=wgan.generator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    metric_['WGAN_G'] = utils.get_final_test_metrics(classifier.statistics)

    # WGAN_EGD
    utils.set_random_state()
    classifier = Classifier('WGAN_EGD')
    classifier.egd_train(
        encoder=vae.encoder,
        generator=wgan.generator,
        discriminator=wgan.discriminator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
        seed_dataset=MinorityDataset(training=True),
    )
    metric_['WGAN_EGD'] = utils.get_final_test_metrics(classifier.statistics)

    # SNGAN_G
    utils.set_random_state()
    sngan = src.SNGAN()
    sngan.train(MinorityDataset(training=True))
    sngan.load_model()
    utils.set_random_state()
    classifier = Classifier('SNGAN_G')
    classifier.g_train(
        generator=sngan.generator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    metric_['SNGAN_G'] = utils.get_final_test_metrics(classifier.statistics)

    # SNGAN_EGD
    utils.set_random_state()
    classifier = Classifier('SNGAN_EGD')
    classifier.egd_train(
        encoder=vae.encoder,
        generator=sngan.generator,
        discriminator=sngan.discriminator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
        seed_dataset=MinorityDataset(training=True),
    )
    metric_['SNGAN_EGD'] = utils.get_final_test_metrics(classifier.statistics)

    # WGANGP_G
    utils.set_random_state()
    wgangp = src.WGANGP()
    wgangp.train(MinorityDataset(training=True))
    wgangp.load_model()
    utils.set_random_state()
    classifier = Classifier('WGANGP_G')
    classifier.g_train(
        generator=wgangp.generator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )
    metric_['WGANGP_G'] = utils.get_final_test_metrics(classifier.statistics)

    # WGANGP_EGD
    utils.set_random_state()
    classifier = Classifier('WGANGP_EGD')
    classifier.egd_train(
        encoder=vae.encoder,
        generator=wgangp.generator,
        discriminator=wgangp.discriminator,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
        seed_dataset=MinorityDataset(training=True),
    )
    metric_['WGANGP_EGD'] = utils.get_final_test_metrics(classifier.statistics)

    return metric_


def validate(file_name_: str) -> pd.DataFrame:
    result_ = {
        k: {
            kk: [] for kk in METRICS
        } for k in MODELS
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
                result_[model_name][metric_name].append(vv)

    for model_name, v in result_.items():
        for metric_name, vv in v.items():
            result_[model_name][metric_name] = sum(vv) / len(vv)

    return pd.DataFrame(result_)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    all_datasets = src.utils.get_all_datasets()

    with open(config.path.data / 'tested_datasets.txt', 'w') as f:
        f.write('')

    result = dict()
    for file_name in all_datasets:
        print(f'{file_name:*^50}')
        result[file_name] = validate(file_name)
        with open(config.path.data / 'tested_datasets.txt', 'a') as f:
            f.write(f'{file_name}\n')
            f.write(f'{result[file_name]}\n\n')

    with pd.ExcelWriter(config.path.data / 'validation.xlsx') as writer:
        for filename, df in result.items():
            df.to_excel(writer, filename.split('.')[0])
            for column in df:
                column_width = 15
                col_idx = df.columns.get_loc(column) + 1
                writer.sheets[filename.split('.')[0]].set_column(col_idx, col_idx, column_width)
            df.style.highlight_max(axis=1).to_excel(writer, filename.split('.')[0])
    print('done!')
