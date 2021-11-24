import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from AEGAN.GAN.models import Generator, Discriminator
from AEGAN.logger import Logger
from AEGAN.dataset import MinorityDataset
from AEGAN.utils import cal_gradient_penalty


class GenerativeAdversarialNetwork:
    def __init__(
            self,
            x_size: int = config.data.x_size,
            z_size: int = config.data.z_size,
    ):
        self.logger = Logger(self.__class__.__name__)

        self.generator = Generator(
            in_size=z_size,
            out_size=x_size,
            hidden_sizes=[
                128, 128, 128,
            ]
        ).to(config.device)
        self.discriminator = Discriminator(
            in_size=x_size,
            out_size=1,
            hidden_sizes=[
                128, 128, 128,
            ]
        ).to(config.device)

        self.generator_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=config.training.GAN.g_learning_rate,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=config.training.GAN.d_learning_rate,
        )

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
            batch_size=config.training.GAN.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        g_losses = []
        d_losses = []

        for e in range(config.training.GAN.epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, _) in enumerate(data_loader):
                x = x.to(config.device)
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                loss = 0
                for _ in range(config.training.GAN.d_n_loop):
                    loss = self._train_d(x)
                d_losses.append(loss)
                for _ in range(config.training.GAN.g_n_loop):
                    loss = self._train_g()
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
            plt.savefig(fname=str(config.path.plots / 'GAN_loss.jpg'))
            plt.clf()

    def _train_d(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(
            config.training.GAN.batch_size,
            config.data.z_size,
            device=config.device,
        )
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        gradient_penalty = cal_gradient_penalty(
            d_model=self.discriminator,
            real_x=x,
            fake_x=fake_x,
        )
        loss = loss_real + loss_fake + gradient_penalty
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_g(self) -> float:
        self.generator.zero_grad()
        z = torch.randn(
            config.training.GAN.batch_size,
            config.data.z_size,
            device=config.device,
        )
        fake_x = self.generator(z)
        prediction_fake = self.discriminator(fake_x)
        loss = - prediction_fake.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()
