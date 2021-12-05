import random

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

from .classifier_model import ClassifierModel
from src.logger import Logger
from src import config


class Classifier:

    def __init__(
            self,
            name: str,
    ):
        self.name = name
        self.model = ClassifierModel().to(config.device)
        self.logger = Logger(name)
        self.statistics = {
            'Loss': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'Accuracy': [],
            'AUC': [],
        }

    def train(
            self,
            training_dataset: Dataset,
            test_dateset: Dataset,
    ) -> None:

        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)
                label = label.to(config.device)
                prediction = self.model(x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=label,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')

    def g_train(
            self,
            generator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
    ) -> None:

        self.logger.info('Started training with generator')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)
                label = label.to(config.device)
                real_minority_num = int(label.sum().item())
                fake_minority_num = config.training.classifier.batch_size - real_minority_num
                z = torch.randn(fake_minority_num, config.data.z_size, device=config.device)
                supplement_x = generator(z).detach()
                balanced_x = torch.cat([x, supplement_x])
                balanced_label = torch.cat([label, torch.ones(fake_minority_num, device=config.device)])
                prediction = self.model(balanced_x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=balanced_label,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')

    def egw_train(
            self,
            encoder: nn.Module,
            generator: nn.Module,
            discriminator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
            seed_dataset: Dataset,
    ) -> None:

        self.logger.info('Started training with encoder, generator and weighted loss')
        self.logger.debug(f'Using device: {config.device}')

        dl = DataLoader(
            dataset=training_dataset,
            batch_size=config.training.classifier.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.training.classifier.lr,
            betas=(0.5, 0.9),
        )

        for _ in tqdm(range(config.training.classifier.epochs)):
            for x, label in dl:
                self.model.zero_grad()
                x = x.to(config.device)
                label = label.to(config.device)
                real_minority_num = int(label.sum().item())
                fake_minority_num = config.training.classifier.batch_size - real_minority_num
                seed = random.choice(seed_dataset[:][0]).to(config.device)

                z = encoder(seed).detach()
                supplement_x = generator(z).detach()

                score = discriminator(x).detach()
                weight = ((score - score.min()) / (score.max() - score.min())).squeeze(dim=1)
                weight = torch.cat([torch.ones(real_minority_num, device=config.device), weight])
                weight = (fake_minority_num / weight.sum()) * weight

                balanced_x = torch.cat([x, supplement_x])
                balanced_label = torch.cat([label, torch.ones(fake_minority_num, device=config.device)])
                prediction = self.model(balanced_x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=balanced_label,
                    weight=weight,
                )
                loss.backward()
                optimizer.step()
                self.statistics['Loss'].append(loss.item())
            self._test(test_dateset)

        self._plot()
        self.logger.info('Finished training')

    def _test(self, test_dataset):
        with torch.no_grad():
            self.model.eval()
            x, label = test_dataset[:]
            x = x.to(config.device)
            prob = self.model(x)
            predicted_label = self._prob2label(prob)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=label,
                y_pred=predicted_label,
                pos_label=1,
                average='binary',
                zero_division=0,
            )
            self.statistics['Precision'].append(precision)
            self.statistics['Recall'].append(recall)
            self.statistics['F1'].append(f1)
            self.model.train()

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels)

    def _plot(self):
        sns.set()
        plt.title(f"{self.name} Classifier Loss During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        sns.lineplot(data=self.statistics['Loss'])
        loss_plot_path = config.path.plots / f'{self.name}_Classifier_Loss.png'
        plt.savefig(loss_plot_path)
        plt.clf()
        self.logger.debug(f'Saved loss plot at {loss_plot_path}')

        sns.set()
        plt.title(f"{self.name} Classifier Test Metrics During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        for name, value in self.statistics.items():
            if name == 'Loss':
                continue
            plt.plot(value, label=name)
        plt.legend()
        test_metrics_plot_path = config.path.plots / f'{self.name}_Classifier_Test_Metrics.png'
        plt.savefig(test_metrics_plot_path)
        plt.clf()
        self.logger.debug(f'Saved test metrics plot at {test_metrics_plot_path}')
