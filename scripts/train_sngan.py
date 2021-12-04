from src import utils
from src.dataset import MinorityDataset
from src.config import data
from src.sngan import SNGAN


FILE_NAME = 'page-blocks0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.set_random_state()
    utils.prepare_dataset(FILE_NAME)
    dataset = MinorityDataset(training=True)
    # set config
    data.x_size = len(dataset[0][0])
    # train
    utils.set_random_state()
    SNGAN().train(dataset=dataset)
