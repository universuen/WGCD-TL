import torch
from torch import nn

import config


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def cal_gradient_penalty(
        d_model: torch.nn.Module,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
):
    alpha = torch.rand(config.training.GAN.batch_size, 1, 1, 1).to(config.device)

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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.training.GAN.gp_lambda
    return gradient_penalty
