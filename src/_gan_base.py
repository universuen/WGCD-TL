from abc import abstractmethod

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src.logger import Logger
from src import config


class GANBase:

    def __init__(
            self,
            g: Module,
            d: Module,
            g_optimizer: Optimizer,
            d_optimizer: Optimizer,
            training_config,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g
        self.d = d
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.training_config = training_config
        self.metrics = {
            'd_loss': [],
            'g_loss': [],
        }

    def train(self, dataset: Dataset):

        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        for _ in tqdm(range(self.training_config.epochs)):
            x = dataset[:][0].to(config.device)
            loss = 0
            for _ in range(self.training_config.d_n_loop):
                loss = self._train_d(x)
            self.metrics['d_loss'].append(loss)
            for _ in range(self.training_config.g_n_loop):
                loss = self._train_g(len(x))
            self.metrics['g_loss'].append(loss)

        self._save_model()
        self._plot()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _train_d(self, x: torch.Tensor) -> float:

        pass

    @abstractmethod
    def _train_g(self, x_len: int) -> float:

        pass

    def _save_model(self):

        g_path = config.path.data / f'{self.__class__.__name__}_g.pt'
        torch.save(self.g.state_dict(), g_path)
        self.logger.debug(f'Saved generator model at {g_path}')

        d_path = config.path.data / f'{self.__class__.__name__}_d.pt'
        torch.save(self.d.state_dict(), d_path)
        self.logger.debug(f'Saved discriminator model at {d_path}')

    def _plot(self):

        sns.set()
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.metrics['g_loss'], label="Generator")
        plt.plot(self.metrics['d_loss'], label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = config.path.plots / f'{self.__class__.__name__}_Loss.png'
        plt.savefig(fname=str(plot_path))
        plt.clf()
        self.logger.debug(f'Saved plot at {plot_path}')

    def load_model(self):

        g_path = config.path.data / f'{self.__class__.__name__}_g.pt'
        self.g.load_state_dict(
            torch.load(g_path)
        )
        self.g.to(config.device)
        self.g.eval()
        self.logger.debug(f'Loaded generator model at {g_path}')

        d_path = config.path.data / f'{self.__class__.__name__}_d.pt'
        self.d.load_state_dict(
            torch.load(d_path)
        )
        self.d.to(config.device)
        self.d.eval()
        self.logger.debug(f'Loaded discriminator model at {d_path}')
