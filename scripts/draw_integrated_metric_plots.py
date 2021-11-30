import context

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import config
from src import EGANClassifier, Classifier, VAE, EGAN
from src.dataset import CompleteDataset
from train_sngan_classifier import SNGAN
from train_wgan_gp_classifier import WGAN

if __name__ == '__main__':
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

    # draw plots
    sns.set()
    metric_names = [
        'Precision',
        'Recall',
        'F1',
        'Accuracy',
        'AUC',
        'ROC',
    ]
    for metric_name in metric_names:
        plt.title(metric_name)
        for model_name, metric_lists in metrics.items():
            target_metric = metric_lists[metric_name]
            if metric_name == 'ROC':
                sns.lineplot(x=target_metric[0], y=target_metric[1], label=model_name)
            else:
                plt.plot(target_metric, label=model_name)
        if metric_name == 'ROC':
            sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
        else:
            plt.xlabel('Iterations')
            plt.ylabel('Value')
        plt.legend()
        plt.savefig(fname=str(config.path.plots / f'integrated_{metric_name}.png'))
        plt.clf()
