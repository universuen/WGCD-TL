from math import sqrt

import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, Optimizer
from sklearn.metrics import roc_auc_score, confusion_matrix

from src import config, logger, models
from src.types import Dataset, GAN


class Classifier:
    def __init__(self, name: str):
        self.name = name
        self.model = models.ClassifierModel().to(config.device)
        self.logger = logger.Logger(name)
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
        }

    def fit(
            self,
            dataset: Dataset,
            optimizer: Optimizer = None,
            gan: GAN = None,
    ):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        if optimizer is None:
            optimizer = Adam(
                params=self.model.parameters(),
                lr=config.classifier_config.lr,
                betas=(0.5, 0.9),
            )
        dataset.to(config.device)
        if gan is None:
            x, labels = dataset.samples, dataset.labels
            weights = torch.ones(len(x), device=config.device)
        else:
            real_x, real_labels = dataset.samples, dataset.labels
            real_x_weights = torch.ones(len(real_x), device=config.device)
            # get positive indices and negative indices
            pos_indices, neg_indices = [], []
            for idx, item in enumerate(real_labels):
                if item == 1:
                    pos_indices.append(idx)
                elif item == 0:
                    neg_indices.append(idx)
                else:
                    raise ValueError(f"Invalid value found in labels: {item}")
            # count positive samples and negative samples
            pos_num = len(pos_indices)
            neg_num = len(neg_indices)
            assert pos_num < neg_num
            # calculate weights
            generated_x_num = neg_num - pos_num
            generated_x = gan.generate_samples(generated_x_num)
            scores = gan.d(generated_x).squeeze(dim=1).detach()
            generated_x_weights = (scores - scores.min()) / (scores.max() - scores.min())
            # eta = neg_num / (sum(generated_x_weights) + pos_num)
            # assert eta >= 1
            # real_x_weights[pos_indices] *= eta
            # generated_x_weights *= eta
            delta = (neg_num - (pos_num + sum(generated_x_weights))) / neg_num
            real_x_weights[pos_indices] += delta
            generated_x_weights += delta
            x = torch.cat([real_x, generated_x])
            labels = torch.cat([real_labels, torch.ones(len(generated_x), device=config.device)])
            weights = torch.cat([real_x_weights, generated_x_weights])

        for _ in range(config.classifier_config.epochs):
            self.model.zero_grad()
            prediction = self.model(x).squeeze()
            loss = binary_cross_entropy(
                input=prediction,
                target=labels,
                weight=weights,
            )
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(config.device)
        prob = self.model(x)
        return self._prob2label(prob)

    def test(self, test_dataset: Dataset):
        with torch.no_grad():
            x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
            predicted_label = self.predict(x).cpu()
            tn, fp, fn, tp = confusion_matrix(
                y_true=label,
                y_pred=predicted_label,
            ).ravel()

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0

            f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            g_mean = sqrt(recall * specificity)

            auc = roc_auc_score(
                y_true=label,
                y_score=predicted_label,
            )

            self.metrics['F1'] = f1
            self.metrics['G-Mean'] = g_mean
            self.metrics['AUC'] = auc

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels).to(config.device)
