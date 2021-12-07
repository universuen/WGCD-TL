import torch

from src import config
from src._gan_base import GANBase
from .model import DiscriminatorModel, GeneratorModel


class GAN(GANBase):

    def __init__(self):
        generator = GeneratorModel().to(config.device)
        discriminator = DiscriminatorModel().to(config.device)
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=torch.optim.Adam(
                params=generator.parameters(),
                lr=config.training.gan.generator_lr,
                betas=(0.5, 0.9),
            ),
            discriminator_optimizer=torch.optim.Adam(
                params=discriminator.parameters(),
                lr=config.training.gan.discriminator_lr,
                betas=(0.5, 0.9),
            ),
            training_config=config.training.gan,
        )
        self.criterion = torch.nn.BCELoss()

    def _train_discriminator(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()

        prediction_real = self.discriminator(x).squeeze(dim=1)
        label_real = torch.ones(len(x), device=config.device)
        loss_real = self.criterion(prediction_real, label_real)

        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x).squeeze(dim=1)
        label_fake = torch.zeros(len(x), device=config.device)
        loss_fake = self.criterion(prediction_fake, label_fake)

        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()

        return loss.item()

    def _train_generator(self, x_len: int) -> float:
        self.generator.zero_grad()

        z = torch.randn(x_len, config.data.z_size, device=config.device)
        fake_x = self.generator(z)
        prediction = self.discriminator(fake_x).squeeze(dim=1)
        label = torch.ones(x_len, device=config.device)
        loss = self.criterion(prediction, label)
        loss.backward()
        self.generator_optimizer.step()

        return loss.item()

