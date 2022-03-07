import torch
from torch import nn
from torch.optim import Adam

from src.classifier import Classifier
from src.types import Dataset, GAN
from src.logger import Logger
from src import config


class TransferLearner:
    def __init__(self):
        self.classifier = Classifier(self.__class__.__name__)
        self.classifier.logger.turn_off()
        self.logger = Logger(self.__class__.__name__)
        self.metrics = self.classifier.metrics

    def fit(self, dataset: Dataset, gan: GAN):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        self.classifier.fit(dataset, gan=gan)
        for i in self.classifier.model.main_model.parameters():
            i.requires_grad = False
        optimizer = Adam(
            params=self.classifier.model.last_layer.parameters(),
            lr=config.classifier_config.lr,
            betas=(0.5, 0.9),
        )
        self.classifier.fit(dataset, optimizer=optimizer)
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.predict(x)

    def test(self, test_dataset: Dataset):
        self.classifier.test(test_dataset)
