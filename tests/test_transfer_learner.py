import context

from src import utils
from src.datasets import FullDataset, WeightedPositiveDataset
from src.transfer_learner import TransferLearner
from src.gans import SNGAN

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    gan = SNGAN()
    gan.fit(WeightedPositiveDataset())
    tl_classifier = TransferLearner()
    tl_classifier.fit(
        dataset=FullDataset(),
        gan=gan,
    )
    tl_classifier.test(FullDataset(test=True))
    for name, value in tl_classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
