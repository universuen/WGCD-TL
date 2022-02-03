import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm

from src import config
from src.logger import Logger
from src.models import VAEEModel, VAEDModel


class VAE:

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.e = VAEEModel().to(config.device)
        self.d = VAEDModel().to(config.device)

    def train(self, dataset):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        e_optimizer = torch.optim.Adam(
            params=self.e.parameters(),
            lr=config.vae.e_lr,
        )
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.vae.d_lr,
        )
        x = dataset[:][0]
        x = x.to(config.device)
        for _ in tqdm(range(config.vae.epochs)):
            # clear gradients
            self.e.zero_grad()
            self.d.zero_grad()
            # calculate z, mu and sigma
            z, mu, sigma = self.e(x)
            # calculate x_hat
            x_hat = self.d(z)
            # calculate loss
            divergence = - 0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
            loss = divergence + mse_loss(x_hat, x)
            # calculate gradients
            loss.backward()
            # optimize models
            e_optimizer.step()
            d_optimizer.step()

        self.e.eval()
        self.d.eval()
        self.save()
        self.logger.info("Finished training")

    def save(self):
        e_path = config.path.models / f'{self.__class__.__name__}_e.pt'
        torch.save(self.e.state_dict(), e_path)
        self.logger.debug(f'Saved e model at {e_path}')

        d_path = config.path.models / f'{self.__class__.__name__}_d.pt'
        torch.save(self.d.state_dict(), d_path)
        self.logger.debug(f'Saved d model at {d_path}')

    def load(self):
        e_path = config.path.models / f'{self.__class__.__name__}_e.pt'
        self.e.load_state_dict(
            torch.load(e_path)
        )
        self.e.to(config.device)
        self.e.eval()
        self.logger.debug(f'Loaded encoder model from {e_path}')

        d_path = config.path.models / f'{self.__class__.__name__}_d.pt'
        self.d.load_state_dict(
            torch.load(d_path)
        )
        self.d.to(config.device)
        self.d.eval()
        self.logger.debug(f'Loaded decoder model from {d_path}')
