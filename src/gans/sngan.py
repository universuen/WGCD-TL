import torch
from tqdm import tqdm

from src import config
from src.models import GANGModel, SNGANDModel
from src.datasets import PositiveDataset
from ._base import Base


class SNGAN(Base):
    def __init__(self):
        super(SNGAN, self).__init__(GANGModel(), SNGANDModel())

    def _fit(self):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.gan.d_lr,
            betas=(0.5, 0.9),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=config.gan.g_lr,
            betas=(0.5, 0.9),
        )

        x = PositiveDataset()[:][0].to(config.device)
        for _ in tqdm(range(config.gan.epochs)):
            for __ in range(config.gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), config.gan.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(config.gan.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), config.gan.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()
