import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

import config
from config.training.classifier import epochs, batch_size, learning_rate
from src.dataset import MajorityDataset, MinorityDataset, CompleteDataset
from src.classifier.model import ClassifierModel
from src.logger import Logger


class Classifier:
    def __init__(self, name='Simple_Classifier'):
        self.logger = Logger(name)
        self.model = ClassifierModel().to(config.device)
        self.name = name
        self.data_loader = None

    def train(self, training_dataset=None):
        if training_dataset is None:
            training_dataset = CompleteDataset(training=True)
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        self.logger.debug(f'loaded {len(training_dataset)} samples from training dataset')

        data_loader = DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_data_loader_workers,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []
        auc_list = []

        for e in tqdm(range(epochs)):
            # print(f'\nepoch: {e + 1}')
            # train
            single_loss_list = self._single_epoch_train(data_loader, optimizer)
            loss_list.extend(single_loss_list)

            # estimate
            precision, recall, f1, accuracy, auc = self.test(in_training=True)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            auc_list.append(auc)

        # visualize metrics
        sns.set()
        plt.title(f"{self.name} Test Metrics During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Value")
        plt.plot(precision_list, label='Precision')
        plt.plot(recall_list, label='Recall')
        plt.plot(f1_list, label='F1')
        plt.plot(accuracy_list, label='Accuracy')
        plt.plot(auc_list, label='AUC')
        plt.legend()
        plt.savefig(config.path.plots / f'{self.name}_test_metrics.png')
        plt.clf()

        # visualize training loss
        # print(f'current loss: {loss_list[-1]: .3f}')
        plt.title(f"{self.name} Loss During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        sns.lineplot(data=loss_list)
        plt.savefig(config.path.plots / f'{self.name}_Loss.png')
        plt.clf()

        # visualize roc curve
        roc_curve_ = self._cal_roc_curve()
        plt.title(f"{self.name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        sns.lineplot(x=roc_curve_[0], y=roc_curve_[1])
        sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--')
        plt.savefig(config.path.plots / f'{self.name}_ROC.png')
        plt.clf()

        self.logger.debug('finished training')
        torch.save(self.model.state_dict(), config.path.data / f'{self.name}_model.pt')
        self.logger.info(f"saved model at {config.path.data / f'{self.name}_model.pt'}")
        return {
            'Precision': precision_list,
            'Recall': recall_list,
            'F1': f1_list,
            'Accuracy': accuracy_list,
            'AUC': auc_list,
            'ROC': roc_curve_,
        }

    def _single_epoch_train(self, data_loader, optimizer):
        single_loss_list = []
        for idx, (x, label) in enumerate(data_loader):
            x = x.to(config.device)
            label = label.to(config.device)
            self.model.zero_grad()
            prediction = self.model(x).squeeze()
            loss = binary_cross_entropy(
                input=prediction,
                target=label,
            )
            loss.backward()
            optimizer.step()
            single_loss_list.append(loss.item())
        return single_loss_list

    def test(self, in_training: bool = False):
        if not in_training:
            self.model.load_state_dict(
                torch.load(
                    config.path.data / f'{self.name}_model.pt'
                )
            )
        self.model.eval()
        majority_test_dataset = MajorityDataset(training=False)
        minority_test_dataset = MinorityDataset(training=False)
        majority_len = len(majority_test_dataset)
        minority_len = len(minority_test_dataset)
        majority_x = majority_test_dataset[:][0].to(config.device)
        minority_x = minority_test_dataset[:][0].to(config.device)
        majority_predicted_values = self.model(majority_x)
        minority_predicted_values = self.model(minority_x)
        majority_predicted_labels = self._probability2label(majority_predicted_values)
        minority_predicted_labels = self._probability2label(minority_predicted_values)
        tp = minority_predicted_labels.sum().item()
        fp = minority_len - tp
        fn = majority_predicted_labels.sum().item()
        tn = majority_len - fn
        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (majority_len + minority_len)
        auc = roc_auc_score(
            np.concatenate(
                [
                    np.zeros(majority_len),
                    np.ones(minority_len),
                ]
            ),
            np.concatenate(
                [
                    majority_predicted_values.squeeze(dim=1).cpu().detach().numpy(),
                    minority_predicted_values.squeeze(dim=1).cpu().detach().numpy(),
                ]
            )
        )
        if in_training:
            self.model.train()
        else:
            print(f"{'Precision':<10}: {precision:.5f}")
            print(f"{'Recall':<10}: {recall:.5f}")
            print(f"{'F1':<10}: {f1:.5f}")
            print(f"{'Accuracy':<10}: {accuracy:.5f}")
            print(f"{'AUC':<10}: {auc:.5f}")
        return precision, recall, f1, accuracy, auc

    def _cal_roc_curve(self):
        self.model.eval()
        majority_test_dataset = MajorityDataset(training=False)
        minority_test_dataset = MinorityDataset(training=False)
        majority_len = len(majority_test_dataset)
        minority_len = len(minority_test_dataset)
        majority_x = majority_test_dataset[:][0].to(config.device)
        minority_x = minority_test_dataset[:][0].to(config.device)
        majority_predicted_values = self.model(majority_x)
        minority_predicted_values = self.model(minority_x)
        curve = roc_curve(
            np.concatenate(
                [
                    np.zeros(majority_len),
                    np.ones(minority_len),
                ]
            ),
            np.concatenate(
                [
                    majority_predicted_values.squeeze(dim=1).cpu().detach().numpy(),
                    minority_predicted_values.squeeze(dim=1).cpu().detach().numpy(),
                ]
            )
        )[:2]
        return curve

    @staticmethod
    def _probability2label(probabilities: torch.Tensor):
        probabilities = probabilities.squeeze(dim=1)
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels)
