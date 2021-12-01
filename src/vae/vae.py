import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from tqdm import tqdm

import config
from src.vae.models import EncoderModel, DecoderModel
from src.logger import Logger
from src.dataset import MinorityDataset


class VAE:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.encoder = EncoderModel().to(config.device)
        self.decoder = DecoderModel().to(config.device)

    def train(self):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        dataset = MinorityDataset()
        self.logger.debug(f'loaded {len(dataset)} data')
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.training.vae.learning_rate,
        )
        decoder_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=config.training.vae.learning_rate,
        )
        losses = []
        for e in tqdm(range(config.training.vae.epochs)):
            # clear gradients
            x = dataset[:][0]
            x = x.to(config.device)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # calculate z, mu and sigma
            z, mu, sigma = self.encoder(x)
            # calculate x_hat
            x_hat = self.decoder(z)
            # calculate loss
            divergence = - 0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
            loss = divergence + mse_loss(x_hat, x)
            # calculate gradients
            loss.backward()
            losses.append(loss.item())
            # optimize models
            encoder_optimizer.step()
            decoder_optimizer.step()

        sns.set()
        plt.title("AutoEncoder Loss During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        sns.lineplot(data=losses)
        plt.savefig(config.path.plots / 'VAE_loss.png')
        plt.clf()

        self.logger.info("finished training")
        torch.save(self.encoder.state_dict(), config.path.data / 'encoder.pt')
        self.logger.info(f"saved encoder model at {config.path.data / 'encoder.pt'}")
