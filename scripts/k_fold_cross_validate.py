import context
from time import sleep

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

import config
from src import EGANClassifier, Classifier, VAE, EGAN
from src.dataset import CompleteDataset
from train_sngan_classifier import SNGAN
from train_wgan_gp_classifier import WGAN
from config import path

K = 10


def train_models():
    metrics = dict()
    # collect EGAN Classifier metrics
    VAE().train()
    EGAN().train()
    metrics['EGAN'] = EGANClassifier().train()
    # collect WGAN-GP Classifier metrics
    wgan = WGAN()
    wgan.train()
    wgan.generator.eval()
    training_dataset = CompleteDataset(training=True)
    x_hat_num = len(training_dataset) - 2 * int(training_dataset.labels.sum().item())
    z = torch.randn(x_hat_num, config.data.z_size).to(config.device)
    x_hat = wgan.generator(z).cpu().detach()
    training_dataset.features = torch.cat([training_dataset.features, x_hat])
    training_dataset.labels = torch.cat([training_dataset.labels, torch.ones(x_hat_num)])
    metrics['WGAN-GP'] = Classifier('WGAN_GP_Classifier').train(training_dataset)
    # collect SNGAN Classifier metrics
    sngan = SNGAN()
    sngan.train()
    sngan.generator.eval()
    training_dataset = CompleteDataset(training=True)
    x_hat_num = len(training_dataset) - 2 * int(training_dataset.labels.sum().item())
    z = torch.randn(x_hat_num, config.data.z_size).to(config.device)
    x_hat = sngan.generator(z).cpu().detach()
    training_dataset.features = torch.cat([training_dataset.features, x_hat])
    training_dataset.labels = torch.cat([training_dataset.labels, torch.ones(x_hat_num)])
    metrics['SNGAN'] = Classifier('SNGAN_Classifier').train(training_dataset)
    # collect Simple Classifier metrics
    metrics['Simple'] = Classifier().train()
    # collect SMOTE Classifier metrics
    x, y = CompleteDataset(training=True)[:]
    x = x.numpy()
    y = y.numpy()
    x, y = SMOTE().fit_resample(x, y)
    smote_dataset = CompleteDataset()
    smote_dataset.features = torch.from_numpy(x)
    smote_dataset.labels = torch.from_numpy(y)
    metrics['SMOTE'] = Classifier('SMOTE_Classifier').train(smote_dataset)
    return metrics


def validate(filename: str, skip_rows: int):
    file_path = path.data / filename
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    labels = np_array[:, -1].copy()
    labels = np.array([i.strip() for i in labels])
    features = np_array[:, :-1].copy()
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')
    features = StandardScaler().fit_transform(features)
    config.data.x_size = len(features[0])
    config.training.gan.batch_size = int(labels.sum()) // 2

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=config.seed)

    model_names = ['EGAN', 'SNGAN', 'WGAN-GP', 'SMOTE', 'Simple']
    metric_names = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']
    result = {
        k: {
            kk: [] for kk in metric_names
        } for k in model_names
    }

    for training_indices, test_indices in skf.split(features, labels):
        np.save(str(path.data / 'training_labels.npy'), labels[training_indices])
        np.save(str(path.data / 'training_features.npy'), features[training_indices])
        np.save(str(path.data / 'test_labels.npy'), labels[test_indices])
        np.save(str(path.data / 'test_features.npy'), features[test_indices])

        metrics = train_models()
        for model_name, v in metrics.items():
            for metric_name, vv in v.items():
                try:
                    result[model_name][metric_name].append(vv[-1])
                except KeyError:
                    pass

    for model_name, v in result.items():
        for metric_name, vv in v.items():
            v[metric_name] = sum(vv) / len(vv)

    return pd.DataFrame(result)


if __name__ == '__main__':
    result = dict()
    filename_and_skip_rows = {
        'page-blocks0.dat': 15,
        'segment0.dat': 24,
        'yeast4.dat': 13,
        'yeast5.dat': 13,
        'yeast6.dat': 13,
        'ecoli3.dat': 12,
        'ecoli4.dat': 12,
        'glass0.dat': 14,
        'glass2.dat': 14,
        'glass4.dat': 14,
        'vowel0.dat': 18,
        'haberman.dat': 8,
        'iris0.dat': 9,
        'vehicle0.dat': 23,
        'vehicle1.dat': 23,
        'vehicle2.dat': 23,
        'wisconsin.dat': 14,
        'poker-8_vs_6.dat': 15,
        'shuttle-c0-vs-c4.dat': 14,
        'winequality-white-3-9_vs_5.dat': 16,
    }
    with open(path.data / 'tested_datasets.txt', 'w') as f:
        f.write('')
    for filename, skip_rows in filename_and_skip_rows.items():
        result[filename] = validate(filename, skip_rows)
        with open(path.data / 'tested_datasets.txt', 'a') as f:
            f.write(f'{filename}\n')
            f.write(f'{result[filename]}\n\n')

    with pd.ExcelWriter(config.path.data / 'validation.xlsx') as writer:
        for filename, df in result.items():
            df.to_excel(writer, filename.split('.')[0])
    print('done!')
