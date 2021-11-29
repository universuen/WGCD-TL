from random import sample

import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

import config
from config.training.classifier import epochs, batch_size, learning_rate
from src.dataset import MinorityDataset, CompleteDataset
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

    def train(self, training_dataset = CompleteDataset(training=True)):
        self.logger.info('started training')
        self.logger.debug(f'using device: {config.device}')
        self.logger.debug(f'loaded {len(training_dataset)} samples from training dataset')
        minority_training_dataset = MinorityDataset(training=True)

        data_loader = DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        losses = []
        precision_list = []
        recall_list = []
        f1_list = []

        for e in range(epochs):
            print(f'\nepoch: {e + 1}')
            for idx, (x, label) in enumerate(data_loader):
                print(f'\rprocess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')
                x = x.to(config.device)
                label = label.to(config.device)
                self.model.zero_grad()

                # balance samples
                real_minority_num = int(label.sum().item())
                fake_minority_num = batch_size - real_minority_num
                seed = sample(list(minority_training_dataset),
                              fake_minority_num)
                seed = torch.stack([i[0].to(config.device) for i in seed])
                z, _, _ = self.encoder(seed)
                supplement_x = self.generator(z).detach()
                score = self.discriminator(supplement_x).detach()
                supplement_weight = ((score - score.min()) / (score.max() - score.min())).squeeze()
                balanced_x = torch.cat([x, supplement_x])
                balanced_weight = torch.cat([torch.ones(batch_size).to(config.device), supplement_weight])
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
                losses.append(loss.item())

            print('\n')
            precision, recall, f1 = self.test(in_training=True)
            precision_list.append(precision * 100)
            recall_list.append(recall * 100)
            f1_list.append(f1 * 100)
            sns.set()
            plt.title("EGAN-Classifier Test Metrics During Training")
            plt.xlabel("Iterations")
            plt.ylabel("Percentage value")
            plt.plot(precision_list, label='precision')
            plt.plot(recall_list, label='recall')
            plt.plot(f1_list, label='f1')
            plt.legend()
            plt.savefig(config.path.plots / 'EGAN-Classifier_test_metrics.png')
            plt.clf()

            print(f'current loss: {losses[-1]}')
            plt.title("EGAN-Classifier Loss During Training")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            sns.lineplot(data=losses)
            plt.savefig(config.path.plots / 'EGAN-Classifier_loss.png')
            plt.clf()

        self.logger.debug('finished training')
        torch.save(self.model.state_dict(), config.path.data / f'{self.name}_model.pt')
        self.logger.info(f"saved encoder model at {config.path.data / f'{self.name}_model.pt'}")
        return precision_list, recall_list, f1_list
