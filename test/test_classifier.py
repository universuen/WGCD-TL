import context

import torch

from src import utils
from src.dataset import CompleteDataset, MinorityDataset
from src import Classifier, SNGAN


FILE_NAME = 'page-blocks0.dat'

if __name__ == '__main__':

    # prepare dataset
    utils.set_random_state()
    utils.prepare_dataset(FILE_NAME)

    # set config
    utils.set_x_size()

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normally_Train')
    classifier.train(
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )

    # train with generator
    utils.set_random_state()
    sn_gan = SNGAN()
    sn_gan.train(MinorityDataset(training=True))
    sn_gan.load_model()

    utils.set_random_state()
    classifier = Classifier('Test_G_Train')
    classifier.g_train(
        generator=sn_gan.g,
        training_dataset=CompleteDataset(training=True),
        test_dateset=CompleteDataset(training=False),
    )

