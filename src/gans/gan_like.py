from abc import abstractmethod

import torch

from src import config
from src.logger import Logger
from src.models.model_like import ModelLike


class GANLike:
    def __init__(
            self,
            g: ModelLike,
            d: ModelLike,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g.to(config.device)
        self.d = d.to(config.device)

    def fit(self):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        self._fit()
        self.g.eval()
        self.d.eval()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _fit(self):
        pass

    def generate_samples(self, z: torch.Tensor):
        return self.g(z)
