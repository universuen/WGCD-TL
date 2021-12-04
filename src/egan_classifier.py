from random import choice

import torch
from torch.nn.functional import binary_cross_entropy

import config
from src.dataset import MinorityDataset
from src.classifier import Classifier
from src.vae.models import EncoderModel
from src.egan.models import GeneratorModel, DiscriminatorModel


class EGANClassifier(Classifier):
    def __init__(self, weight_optimization: bool, name: str):
        super().__init__(name)
        self.weight_optimization = weight_optimization

        self.generator = GeneratorModel()
        self.generator.load_state_dict(
            torch.load(
                config.path.data / 'EGAN_generator.pt'
            )
        )
        self.generator.to(config.device)
        self.generator.eval()

        self.discriminator = DiscriminatorModel()
        self.discriminator.load_state_dict(
            torch.load(
                config.path.data / 'EGAN_discriminator.pt'
            )
        )
        self.discriminator.to(config.device)
        self.discriminator.eval()

    def _single_epoch_train(self, data_loader, optimizer):
        single_loss_list = []
        for idx, (x, labels) in enumerate(data_loader):
            x = x.to(config.device)
            labels = labels.to(config.device)
            real_minority_num = int(labels.sum().item())
            fake_minority_num = config.training.classifier.batch_size - real_minority_num
            z = torch.randn(fake_minority_num, config.data.z_size, device=config.device)
            supplement_x = self.generator(z).detach()
            balanced_x = torch.cat([x, supplement_x])
            if self.weight_optimization:
                scores = self.discriminator(supplement_x).detach()
                supplement_weight = ((scores - scores.min()) / (scores.max() - scores.min())).squeeze()
                weights = torch.cat(
                    [torch.ones(config.training.classifier.batch_size, device=config.device), supplement_weight]
                )
                weights = weights + (1 - weights.mean())
            else:
                weights = torch.ones(len(balanced_x), device=config.device)
            balanced_labels = torch.cat([labels, torch.ones(fake_minority_num, device=config.device)])
            predictions = self.model(balanced_x).squeeze()
            loss = binary_cross_entropy(
                input=predictions,
                target=balanced_labels,
                weight=weights,
            )
            loss.backward()
            optimizer.step()
            single_loss_list.append(loss.item())
        return single_loss_list
