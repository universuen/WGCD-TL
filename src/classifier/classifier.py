import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import config
from config.training.classifier import epochs, batch_size, learning_rate
from src.dataset import MajorityDataset, MinorityDataset, CompleteDataset
from src.classifier.model import ClassifierModel
from src.logger import Logger


class Classifier:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.model = ClassifierModel().to(config.device)

    def train(self):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        training_dataset = CompleteDataset(training=True)

        data_loader = DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        losses = []
        precision_list = []
        recall_list = []
        f1_list = []

        for e in range(epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, label) in enumerate(data_loader):
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                x = x.to(config.device)
                label = label.to(config.device)
                self.model.zero_grad()
                # train
                prediction = self.model(x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=label,
                )
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print('\n')
            precision, recall, f1 = self.test(in_training=True)
            precision_list.append(precision * 100)
            recall_list.append(recall * 100)
            f1_list.append(f1 * 100)
            sns.set()
            plt.title("Classifier Test Metrics During Training")
            plt.xlabel("Iterations")
            plt.ylabel("Percentage value")
            plt.plot(precision_list, label='precision')
            plt.plot(recall_list, label='recall')
            plt.plot(f1_list, label='f1')
            plt.legend()
            plt.savefig(config.path.plots / 'Classifier_test_metrics.png')
            plt.clf()

            print(f'current loss: {losses[-1]}')
            plt.title("Classifier Loss During Training")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            sns.lineplot(data=losses)
            plt.savefig(config.path.plots / 'Classifier_loss.png')
            plt.clf()
        self.logger.debug('finished training')

    def test(self, in_training: bool = False):
        if not in_training:
            self.model.load_state_dict(
                torch.load(
                    config.path.data / 'classifier.pt'
                )
            )
        self.model.eval()
        majority_test_dataset = MajorityDataset(training=False)
        minority_test_dataset = MinorityDataset(training=False)
        majority_x = majority_test_dataset[:][0].to(config.device)
        minority_x = minority_test_dataset[:][0].to(config.device)
        majority_prediction = self._probability2label(self.model(majority_x))
        minority_prediction = self._probability2label(self.model(minority_x))
        tp = minority_prediction.sum().item()
        fp = len(minority_test_dataset) - tp
        fn = majority_prediction.sum().item()
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        print(f'precision: {precision: %}')
        print(f'recall: {recall: %}')
        print(f'f1: {f1: %}')
        if in_training:
            self.model.train()
        return precision, recall, f1

    @staticmethod
    def _probability2label(probabilities: torch.Tensor):
        probabilities = probabilities.squeeze()
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels)
