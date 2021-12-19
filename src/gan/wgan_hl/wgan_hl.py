import torch
import seaborn as sns
import matplotlib.pyplot as plt

from src import config, utils
from src.gan._gan_base import GANBase
from .models import DiscriminatorModel, GeneratorModel


class WGANHL(GANBase):

    def __init__(self):
        generator = GeneratorModel().to(config.device)
        discriminator = DiscriminatorModel().to(config.device)
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=torch.optim.RMSprop(
                params=generator.parameters(),
                lr=config.training.wgan_hl.generator_lr,
            ),
            discriminator_optimizer=torch.optim.RMSprop(
                params=discriminator.parameters(),
                lr=config.training.wgan_hl.discriminator_lr,
            ),
            training_config=config.training.wgan_hl,
        )
        self.statistics['hidden_loss'] = []

    def _train_discriminator(self, x: torch.Tensor) -> float:
        self.discriminator.zero_grad()
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        for p in self.discriminator.parameters():
            p.data.clamp_(config.training.wgan_hl.lower_clamp, config.training.wgan_hl.upper_clamp)
        return loss.item()

    def _train_generator(self, x_len: int) -> float:
        self.generator.zero_grad()
        # get the hidden output of real x
        real_x_hidden_output = self.discriminator.hidden_output.detach()

        # get the final output and hidden output of fake x
        z = torch.randn(x_len, config.data.z_size, device=config.device)
        fake_x = self.generator(z)
        final_output = self.discriminator(fake_x)
        fake_x_hidden_output = self.discriminator.hidden_output

        cal_kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        real_x_hidden_distribution = utils.normalize(real_x_hidden_output)
        fake_x_hidden_distribution = utils.normalize(fake_x_hidden_output)
        hidden_loss = cal_kl_div(
            input=fake_x_hidden_distribution,
            target=real_x_hidden_distribution,
        ) * config.training.wgan_hl.hl_lambda

        # hidden_loss = torch.norm(
        #     input=real_x_hidden_output - fake_x_hidden_output,
        #     p=2,
        # ) * config.training.wgan_hl.hl_lambda

        # hidden_loss = torch.cosine_similarity(
        #     fake_x_hidden_output,
        #     real_x_hidden_output,
        # ).mean() * config.training.wgan_hl.hl_lambda

        self.statistics['hidden_loss'].append(hidden_loss.item())
        loss = -final_output.mean() + hidden_loss
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def _plot(self):
        sns.set()
        plt.figure(self.__class__.__name__)
        plt.clf()
        plt.title(f"{self.__class__.__name__} Generator and Discriminator Loss During Training")
        plt.plot(self.statistics['generator_loss'], label="Generator")
        plt.plot(self.statistics['discriminator_loss'], label="Discriminator")
        plt.plot(self.statistics['hidden_loss'], label="Hidden")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = config.path.plots / f'{self.__class__.__name__}_Loss.png'
        plt.savefig(fname=str(plot_path))
        plt.clf()
        self.logger.debug(f'Saved plot at {plot_path}')
