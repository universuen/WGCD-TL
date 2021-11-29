import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from src.vae.models import EncoderModel
from src.egan.models import GeneratorModel, DiscriminatorModel
from src.logger import Logger
from src.dataset import MinorityDataset


class EGAN:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

        self.encoder = EncoderModel()
        self.encoder.load_state_dict(torch.load(
            config.path.data / 'encoder.pt'))
        self.encoder = self.encoder.to(config.device)
        self.encoder.eval()

        self.generator = GeneratorModel().to(config.device)
        self.discriminator = DiscriminatorModel().to(config.device)

        self.generator_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=config.training.gan.g_learning_rate,
            betas=(0.5, 0.9)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=config.training.gan.d_learning_rate,
            betas=(0.5, 0.9)
        )

    def train(self):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        dataset = MinorityDataset()
        self.logger.debug(f'loaded {len(dataset)} data')
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.gan.batch_size * 2,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_data_loader_workers,
        )
        g_losses = []
        d_losses = []

        for e in range(config.training.gan.epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, _) in enumerate(data_loader):

                x = x.to(config.device)
                x_1, x_2 = x.split(config.training.gan.batch_size)
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                loss = 0

                for _ in range(config.training.gan.d_n_loop):
                    loss = self._train_d(x_1, x_2)
                d_losses.append(loss)
                for _ in range(config.training.gan.g_n_loop):
                    loss = self._train_g(x_1)
                g_losses.append(loss)

            print(
                f"\n"
                f"Discriminator loss: {d_losses[-1]}\n"
                f"Generator loss: {g_losses[-1]}\n"
            )
            sns.set()
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses, label="generator")
            plt.plot(d_losses, label="discriminator")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(fname=str(config.path.plots / 'EGAN_loss.png'))
            plt.clf()

        self.logger.info("finished training")
        torch.save(self.generator.state_dict(), config.path.data / 'EGAN_generator.pt')
        self.logger.info(f"saved generator model at {config.path.data / 'EGAN_generator.pt'}")
        torch.save(self.discriminator.state_dict(), config.path.data / 'EGAN_discriminator.pt')
        self.logger.info(f"saved discriminator model at {config.path.data / 'EGAN_discriminator.pt'}")

    def _train_d(self, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x_2)
        loss_real = - prediction_real.mean()
        z, _, _ = self.encoder(x_1)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_g(self, x_1: torch.Tensor) -> float:
        self.generator.zero_grad()
        z, _, _ = self.encoder(x_1)
        fake_x = self.generator(z)
        prediction_fake = self.discriminator(fake_x)
        loss = - prediction_fake.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()
