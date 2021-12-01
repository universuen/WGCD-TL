import context

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from src.egan.models import GeneratorModel
from src.logger import Logger
from src.dataset import MinorityDataset, CompleteDataset
from src import Classifier

import torch
from torch import nn

from src.utils import init_weights


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(config.data.x_size, 32, bias=False),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64, bias=False),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8, bias=False),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 4, bias=False),
            nn.LayerNorm(4),
            nn.LeakyReLU(0.2),
            nn.Linear(4, 2, bias=False),
            nn.LayerNorm(2),
            nn.LeakyReLU(0.2),
            nn.Linear(2, 1),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat


gp_lambda = 10


def cal_gradient_penalty(
        d_model: torch.nn.Module,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
):
    alpha = torch.rand(len(real_x), 1, 1, 1).to(config.device)

    interpolates = alpha * real_x + (1 - alpha) * fake_x
    interpolates.requires_grad = True

    disc_interpolates = d_model(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(config.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size()[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


class WGAN:
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
        g_losses = []
        d_losses = []

        for e in tqdm(range(config.training.gan.epochs)):
            x = dataset[:][0].to(config.device)
            x = torch.split(x, len(x) // 2)[0]
            loss = 0
            for _ in range(config.training.gan.d_n_loop):
                loss = self._train_d(x)
            d_losses.append(loss)
            for _ in range(config.training.gan.g_n_loop):
                loss = self._train_g(len(x))
            g_losses.append(loss)

        sns.set()
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="generator")
        plt.plot(d_losses, label="discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(fname=str(config.path.plots / 'WGAN_GP_loss.png'))
        plt.clf()

        self.logger.info("finished training")

    def _train_d(self, x) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size).to(config.device)
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

    def _train_g(self, x_len) -> float:
        self.generator.zero_grad()
        z = torch.randn(x_len, config.data.z_size).to(config.device)
        fake_x = self.generator(z)
        prediction_fake = self.discriminator(fake_x)
        loss = - prediction_fake.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train()
    wgan.generator.eval()
    training_dataset = CompleteDataset(training=True)
    x_hat_num = len(training_dataset) - 2 * int(training_dataset.labels.sum().item())
    z = torch.randn(x_hat_num, config.data.z_size).to(config.device)
    x_hat = wgan.generator(z).cpu().detach()
    training_dataset.features = torch.cat([training_dataset.features, x_hat])
    training_dataset.labels = torch.cat([training_dataset.labels, torch.ones(x_hat_num)])
    Classifier('WGAN_GP_Classifier').train(training_dataset)
