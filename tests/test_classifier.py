import context
import src.config.gan_config

from src import utils
from src.datasets import FullDataset, WeightedPositiveDataset
from src.classifier import Classifier
from src.gans import SNGAN

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normal_Train')
    classifier.fit(
        dataset=FullDataset(),
    )
    classifier.test(FullDataset(test=True))
    for name, value in classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')

    # train with GAN
    utils.set_random_state()
    gan = SNGAN()
    gan.fit(WeightedPositiveDataset())
    classifier = Classifier('Test_Normal_Train')
    classifier.fit(
        dataset=FullDataset(),
        gan=gan,
    )
    classifier.test(FullDataset(test=True))
    for name, value in classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
