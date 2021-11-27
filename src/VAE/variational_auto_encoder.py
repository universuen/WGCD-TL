import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

import config
from src.VAE.models import Encoder, Decoder
from src.logger import Logger
from src.dataset import MinorityTrainingDataset


class VariationalAutoEncoder:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)

    def train(self):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        dataset = MinorityTrainingDataset()
        self.logger.debug(f'loaded {len(dataset)} data')
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.VAE.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.training.VAE.learning_rate,
        )
        decoder_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=config.training.VAE.learning_rate,
        )
        losses = []
        for e in range(config.training.VAE.epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, _) in enumerate(data_loader):
                x = x.to(config.device)
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                # clear gradients
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
            print(f'\ncurrent loss: {losses[-1]}')

            sns.set()
            plt.title("AutoEncoder Loss During Training")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            sns.lineplot(data=losses)
            plt.savefig(config.path.plots / 'VAE_loss.png')
            plt.clf()

        self.logger.debug("training finished.")
        torch.save(self.encoder.state_dict(), config.path.data/'encoder.pt')
        self.logger.info(f"saved encoder model at {config.path.data/'encoder.pt'}")

