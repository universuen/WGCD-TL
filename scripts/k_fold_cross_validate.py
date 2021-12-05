import context

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import src
from src import utils, Classifier, VAE
from src.dataset import CompleteDataset, MinorityDataset
from src import config

K = 10
MODELS = (
    'SNGAN_G',
    'SNGAN_EGD',
    'WGANGP_G',
    'WGANGP_EGD',
)
METRICS = (
    'Precision',
    'Recall',
    'F1',
    'Accuracy',
    'AUC',
)


def train_all() -> dict:
    metric_ = dict()
    utils.set_x_size()

    # prepare encoder
    utils.set_random_state()
    vae = VAE()
    vae.train(MinorityDataset(training=True))
    vae.load_model()

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
    feature = StandardScaler().fit_transform(feature)

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
    all_datasets = (
        'page-blocks0.dat',
        'segment0.dat',
        'yeast4.dat',
        'yeast5.dat',
        'yeast6.dat',
        'ecoli3.dat',
        'ecoli4.dat',
        'glass0.dat',
        'glass2.dat',
        'glass4.dat',
        'vowel0.dat',
        'haberman.dat',
        'iris0.dat',
        'vehicle0.dat',
        'vehicle1.dat',
        'vehicle2.dat',
        'wisconsin.dat',
        'poker-8_vs_6.dat',
        'shuttle-c0-vs-c4.dat',
        'winequality-white-3-9_vs_5.dat',
    )

    with open(config.path.data / 'tested_datasets.txt', 'w') as f:
        f.write('')

    result = dict()
    for file_name in all_datasets:
        result[file_name] = validate(file_name)
        with open(config.path.data / 'tested_datasets.txt', 'a') as f:
            f.write(f'{file_name}\n')
            f.write(f'{result[file_name]}\n\n')

    with pd.ExcelWriter(config.path.data / 'validation.xlsx') as writer:
        for filename, df in result.items():
            df.to_excel(writer, filename.split('.')[0])
    print('done!')
