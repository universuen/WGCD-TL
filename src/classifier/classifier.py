import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .classifier_model import ClassifierModel
from src.logger import Logger
from src import config


class Classifier:

    def __init__(
            self,
            name: str,
    ):
        self.model = ClassifierModel()
        self.logger = Logger(name)
        self.metrics = {
            'Loss': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'Accuracy': [],
            'AUC': [],
        }

    def train(
            self,
            generator: nn.Module,
            training_dataset: Dataset,
            test_dateset: Dataset,
    ) -> dict[str:[float]]:

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
                x.to(config.device)
                label.to(config.device)
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
                self.metrics['Loss'].append(loss)
            self._test(test_dateset)

    def _test(self, test_dataset):
        self.model.eval()
        x, label = test_dataset[:]
        prob = self.model(x)
        predicted_label = self._prob2label(prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=label,
            y_pred=predicted_label,
        )
        self.metrics['precision'].appened(precision)
        self.metrics['recall'].appened(recall)
        self.metrics['F1'].append(f1)
        print(self.metrics)

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels)

    def optimized_train(
            self,
            encoder: nn.Module,
            generator: nn.Module,
            discriminator: nn.Module,
    ) -> dict[str:[float]]:
        pass
