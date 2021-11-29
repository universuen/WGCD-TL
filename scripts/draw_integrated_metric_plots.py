import context

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import config
from src import EGANClassifier, Classifier, VAE, EGAN
from src.dataset import CompleteDataset
from train_gan_classifier import GAN

if __name__ == '__main__':
    metrics = dict()
    # collect EGAN Classifier metrics
    VAE().train()
    EGAN().train()
    metrics['EGAN'] = EGANClassifier().train()
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
    metrics['SMOTE'] = Classifier('SMOTE_Classifier').train(training_dataset=smote_dataset)
    # collect GAN Classifier metrics
    gan = GAN()
    gan.train()
    gan.generator.eval()
    training_dataset = CompleteDataset(training=True)
    x_hat_num = len(training_dataset) - 2 * int(training_dataset.labels.sum().item())
    z = torch.randn(x_hat_num, config.data.z_size).to(config.device)
    x_hat = gan.generator(z).cpu().detach()
    training_dataset.features = torch.cat([training_dataset.features, x_hat])
    training_dataset.labels = torch.cat([training_dataset.labels, torch.ones(x_hat_num)])
    metrics['GAN'] = Classifier('GAN_Classifier').train(training_dataset)

    # draw precision
    sns.set()
    plt.title("Precision")
    for label, (precision, _, _) in metrics.items():
        plt.plot(precision, label=label)
    plt.xlabel("iterations")
    plt.ylabel("Percentage Value")
    plt.legend()
    plt.savefig(fname=str(config.path.plots / 'integrated_precision.png'))
    plt.clf()

    # draw recall
    plt.title("Recall")
    for label, (_, recall, _) in metrics.items():
        plt.plot(recall, label=label)
    plt.xlabel("iterations")
    plt.ylabel("Percentage Value")
    plt.legend()
    plt.savefig(fname=str(config.path.plots / 'integrated_recall.png'))
    plt.clf()

    # draw F1
    plt.title("F1")
    for label, (_, _, f1) in metrics.items():
        plt.plot(f1, label=label)
    plt.xlabel("iterations")
    plt.ylabel("Percentage Value")
    plt.legend()
    plt.savefig(fname=str(config.path.plots / 'integrated_F1.png'))
    plt.clf()
