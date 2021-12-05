import context

import torch

from src import utils, config
from src.dataset import MinorityDataset
from src import SNGAN


FILE_NAME = 'page-blocks0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.set_random_state()
    utils.prepare_dataset(FILE_NAME)
    # set config
    utils.set_x_size()
    # train
    dataset = MinorityDataset(training=True)
    utils.set_random_state()
    sn_gan = SNGAN()
    sn_gan.train(dataset=dataset)
    # test
    sn_gan.load_model()
    z = torch.randn(1, config.data.z_size, device=config.device)
    print(sn_gan.generator(z))
