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
        for i in self.classifier.model.parameters():
            i.requires_grad = False
        self.classifier.model.last_layer = nn.Linear(
            in_features=self.classifier.model.last_layer.in_features,
            out_features=self.classifier.model.last_layer.out_features,
        )

        def init_weights(layer: nn.Module):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)

        self.classifier.model.last_layer.apply(init_weights)
        self.classifier.model.last_layer.to(config.device)
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
