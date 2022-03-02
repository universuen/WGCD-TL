import torch

from src import config
from src.models import WGANGPGModel, WGANGPDModel
from src.gans._gan import GAN


class WGANGP(GAN):
    def __init__(self):
        super().__init__(WGANGPGModel(), WGANGPDModel())

    def _fit(self, x: torch.Tensor):
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

        for _ in range(config.gan_config.epochs):
            for __ in range(config.gan_config.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                gradient_penalty = self._cal_gradient_penalty(x, fake_x)
                loss = loss_real + loss_fake + gradient_penalty
                loss.backward()
                d_optimizer.step()
            for __ in range(config.gan_config.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()

    def _cal_gradient_penalty(
            self,
            x: torch.Tensor,
            fake_x: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(len(x), 1).to(config.device)
        interpolates = alpha * x + (1 - alpha) * fake_x
        interpolates.requires_grad = True
        disc_interpolates = self.d(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(config.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.gan_config.wgangp_lambda
        return gradient_penalty
