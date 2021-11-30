import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.egan.models import GeneratorModel, DiscriminatorModel
from src.logger import Logger
from src.dataset import MinorityDataset, CompleteDataset
from src import Classifier


class SNGAN:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
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

        for e in tqdm(range(config.training.gan.epochs)):
            # print(f'\nepoch: {e + 1}')
            for idx, (x, _) in enumerate(data_loader):

                x = x.to(config.device)
                x = torch.split(x, config.training.gan.batch_size)[1]
                # print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                loss = 0

                for _ in range(config.training.gan.d_n_loop):
                    loss = self._train_d(x)
                d_losses.append(loss)
                for _ in range(config.training.gan.g_n_loop):
                    loss = self._train_g()
                g_losses.append(loss)

            # print(
            #     f"\n"
            #     f"Discriminator loss: {d_losses[-1]}\n"
            #     f"Generator loss: {g_losses[-1]}\n"
            # )
            sns.set()
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses, label="generator")
            plt.plot(d_losses, label="discriminator")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(fname=str(config.path.plots / 'SNGAN_loss.png'))
            plt.clf()

        self.logger.info("finished training")

    def _train_d(self, x) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(config.training.gan.batch_size, config.data.z_size).to(config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_g(self) -> float:
        self.generator.zero_grad()
        z = torch.randn(config.training.gan.batch_size, config.data.z_size).to(config.device)
        fake_x = self.generator(z)
        prediction_fake = self.discriminator(fake_x)
        loss = - prediction_fake.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()


if __name__ == '__main__':
    sngan = SNGAN()
    sngan.train()
    sngan.generator.eval()
    training_dataset = CompleteDataset(training=True)
    x_hat_num = len(training_dataset) - 2 * int(training_dataset.labels.sum().item())
    z = torch.randn(x_hat_num, config.data.z_size).to(config.device)
    x_hat = sngan.generator(z).cpu().detach()
    training_dataset.features = torch.cat([training_dataset.features, x_hat])
    training_dataset.labels = torch.cat([training_dataset.labels, torch.ones(x_hat_num)])
    Classifier('SNGAN_Classifier').train(training_dataset)
