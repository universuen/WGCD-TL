from abc import abstractmethod

import torch
from torch.nn import Module

from src import config
from src.logger import Logger


class Base:
    def __init__(
            self,
            g: Module,
            d: Module,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g.to(config.device)
        self.d = d.to(config.device)

    def train(self):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        self._train()
        self.g.eval()
        self.d.eval()
        self.save()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _train(self):
        pass

    def save(self):
        g_path = config.path.data / f'{self.__class__.__name__}_g.pt'
        torch.save(self.g.state_dict(), g_path)
        self.logger.debug(f'Saved generator model at {g_path}')

        d_path = config.path.data / f'{self.__class__.__name__}_d.pt'
        torch.save(self.d.state_dict(), d_path)
        self.logger.debug(f'Saved discriminator model at {d_path}')

    def load(self):
        g_path = config.path.data / f'{self.__class__.__name__}_g.pt'
        self.g.load_state_dict(
            torch.load(g_path)
        )
        self.g.to(config.device)
        self.g.eval()
        self.logger.debug(f'Loaded generator model from {g_path}')

        d_path = config.path.data / f'{self.__class__.__name__}_d.pt'
        self.d.load_state_dict(
            torch.load(d_path)
        )
        self.d.to(config.device)
        self.d.eval()
        self.logger.debug(f'Loaded discriminator model from {d_path}')
