from math import sqrt, log, ceil

import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

from src.config.tr_ada_boost import classifiers as n_classifier
from src.datasets import BasicDataset
from src.classifier import Classifier


class TrAdaBoost:
    def __init__(self):
        self.classifiers = [Classifier(f'boosted_{i}') for i in range(n_classifier)]
        self.betas: torch.Tensor = None
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
        }

    def fit(self, src_dataset: Dataset, tgt_dataset: Dataset):
        final_betas = []
        combined_dataset = BasicDataset()
        combined_dataset.samples = torch.cat([src_dataset.samples, tgt_dataset.samples])
        combined_dataset.labels = torch.cat([src_dataset.labels, tgt_dataset.labels])
        weights = torch.ones(len(combined_dataset))
        n = len(src_dataset)
        m = len(tgt_dataset)
        beta = 1 / (1 + sqrt(2 * log(n) / n_classifier))
        for i in tqdm(range(n_classifier)):
            p = weights / sum(weights)
            classifier = self.classifiers[i]
            classifier.logger.setLevel('FATAL')
            classifier.fit(combined_dataset, p)
            error_tgt = sum(
                weights[n:n + m] * abs(classifier.predict(tgt_dataset[:][0]) - tgt_dataset[:][1])
            ) / sum(weights[n:n + m])
            if error_tgt >= 0.5:
                error_tgt = 0.5 - 1e-3
            beta_t = (error_tgt / (1 - error_tgt))
            betas = torch.tensor([beta if j < n else beta_t for j in range(n + m)])
            signs = torch.tensor([1 if j < n else -1 for j in range(n + m)])
            exponents = torch.cat(
                [
                    abs(classifier.predict(src_dataset[:][0]) - src_dataset[:][1]),
                    abs(classifier.predict(tgt_dataset[:][0]) - tgt_dataset[:][1]),
                ]
            )
            weights = weights * (betas ** (signs * exponents))
            final_betas.append(beta_t)

        # remove extra classifiers and coefficients
        self.classifiers = self.classifiers[ceil(n_classifier / 2):n_classifier]
        self.betas = torch.tensor(final_betas[ceil(n_classifier / 2):n_classifier])

    def predict(self, x):
        prediction = torch.stack([i.predict(x) for i in self.classifiers]).T
        result = [
            1 if i >= 0 else 0 for i in
            torch.prod(self.betas ** -prediction, dim=1) - torch.prod(self.betas ** -0.5)
        ]
        return torch.tensor(result)

    def test(self, test_dataset: Dataset):
        with torch.no_grad():
            x, label = test_dataset[:]
            predicted_label = self.predict(x)
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