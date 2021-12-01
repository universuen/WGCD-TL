from random import choice

import torch
from torch.nn.functional import binary_cross_entropy

import config
from src.dataset import MinorityDataset
from src.classifier import Classifier
from src.vae.models import EncoderModel
from src.egan.models import GeneratorModel, DiscriminatorModel


class EGANClassifier(Classifier):
    def __init__(self):
        super().__init__('EGAN_Classifier')
        self.encoder = EncoderModel()
        self.encoder.load_state_dict(
            torch.load(
                config.path.data / 'encoder.pt'
            )
        )
        self.encoder.to(config.device)
        self.encoder.eval()

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
        minority_training_dataset = MinorityDataset(training=True)
        single_loss_list = []
        for idx, (x, label) in enumerate(data_loader):
            x = x.to(config.device)
            label = label.to(config.device)
            self.model.zero_grad()

            # balance samples
            real_minority_num = int(label.sum().item())
            fake_minority_num = config.training.classifier.batch_size - real_minority_num
            seed = choice(list(minority_training_dataset))[0]
            seed = torch.stack([seed for _ in range(fake_minority_num)]).to(config.device)
            z, _, _ = self.encoder(seed)
            supplement_x = self.generator(z).detach()
            score = self.discriminator(supplement_x).detach()
            supplement_weight = ((score - score.min()) / (score.max() - score.min())).squeeze()
            balanced_x = torch.cat([x, supplement_x])
            balanced_weight = torch.cat(
                [torch.ones(config.training.classifier.batch_size).to(config.device), supplement_weight]
            )
            balanced_label = torch.cat([label, torch.ones(fake_minority_num).to(config.device)])

            # train
            prediction = self.model(balanced_x).squeeze()
            loss = binary_cross_entropy(
                input=prediction,
                target=balanced_label,
                weight=balanced_weight,
            )
            loss.backward()
            optimizer.step()
            single_loss_list.append(loss.item())
        return single_loss_list
