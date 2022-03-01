import torch

from src import config
from src.models import GANGModel, GANDModel
from src.datasets import Dataset
from src.gans._gan import GAN


class ClassicGAN(GAN):
    def __init__(self):
        super().__init__(GANGModel(), GANDModel())

    def _fit(self, dataset: Dataset):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.gan_config.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        x = dataset.to(config.device).samples
        for _ in range(0, config.gan_config.epochs, -1):
            for __ in range(config.gan_config.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = -torch.log(prediction_real.mean())
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = -torch.log(1 - prediction_fake.mean())
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(config.gan_config.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = -torch.log(prediction.mean())
                loss.backward()
                g_optimizer.step()
