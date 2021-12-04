import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src import config
from src._gan_base import GANBase
from .model import DiscriminatorModel, GeneratorModel


class SNGAN(GANBase):

    def __init__(self):
        g = GeneratorModel().to(config.device)
        d = DiscriminatorModel().to(config.device)
        super(SNGAN, self).__init__(
            g=g,
            d=d,
            g_optimizer=torch.optim.Adam(
                params=g.parameters(),
                lr=config.training.sngan.g_lr,
                betas=(0.5, 0.9),
            ),
            d_optimizer=torch.optim.Adam(
                params=d.parameters(),
                lr=config.training.sngan.d_lr,
                betas=(0.5, 0.9),
            ),
            training_config=config.training.sngan,
        )
        self.metrics = {
            'd_loss': [],
            'g_loss': [],
        }

    def _train_d(self, x: torch.Tensor) -> float:
        self.d.zero_grad()
        prediction_real = self.d(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.g(z).detach()
        prediction_fake = self.d(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

    def _train_g(self, x_len: int) -> float:
        self.g.zero_grad()
        z = torch.randn(x_len, config.data.z_size, device=config.device)
        fake_x = self.g(z)
        prediction_fake = self.d(fake_x)
        loss = - prediction_fake.mean()
        loss.backward()
        self.g_optimizer.step()
        return loss.item()
