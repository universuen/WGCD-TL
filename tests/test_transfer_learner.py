import context

from src import utils
from src.datasets import FullDataset, WeightedPositiveDataset
from src.transfer_learner import TransferLearner

FILE_NAME = 'pima.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    tl_classifier = TransferLearner()
    tl_classifier.fit(
        src_domain=WeightedPositiveDataset(),
        dst_domain=FullDataset(),
    )
    tl_classifier.test(FullDataset(test=True))
    for name, value in tl_classifier.metrics.items():
        print(f'{name:<15}:{value:>10.4f}')
