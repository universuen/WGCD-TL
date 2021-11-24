import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from AEGAN.AE.models import Encoder, Decoder
from AEGAN.logger import Logger
from AEGAN.dataset import MinorityDataset


class AutoEncoder:
    def __init__(
            self,
            x_size: int = config.data.x_size,
            z_size: int = config.data.z_size,
    ):
        self.logger = Logger(self.__class__.__name__)

        self.encoder = Encoder(
            in_size=x_size,
            out_size=z_size,
            hidden_sizes=[
                128, 128, 128,
            ]
        ).to(config.device)

        self.decoder = Decoder(
            in_size=z_size,
            out_size=x_size,
            hidden_sizes=[
                128, 128, 128,
            ]
        ).to(config.device)

    def train(self):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        dataset = MinorityDataset(
            features_path=config.path.data / 'features.npy',
            labels_path=config.path.data / 'labels.npy',
        )
        self.logger.debug(f'loaded {len(dataset)} data')
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.AE.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.training.AE.learning_rate,
        )
        decoder_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=config.training.AE.learning_rate,
        )
        losses = []
        for e in range(config.training.AE.epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, _) in enumerate(data_loader):
                x = x.to(config.device)
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                # clear gradients
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                # feed data
                z = self.encoder(x)
                x_hat = self.decoder(z)
                # calculate gradients
                loss = torch.norm(x - x_hat, 2, dim=1).mean() + torch.norm(z.to(config.device), 2, dim=1).mean()
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
            plt.savefig(config.path.plots / 'AE_loss.png')
            plt.clf()
